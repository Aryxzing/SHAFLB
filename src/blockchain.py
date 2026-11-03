"""
Hash-chained audit ledger for federated training rounds.

Every block stores round metadata (contributions, metrics, a SHA-256 fingerprint of
the global model) and the previous block's hash, so editing any historical block
breaks the chain. Each hospital keeps its own replica; comparing replicas catches
an attacker who edits a block *and* recomputes every downstream hash.
No patient data is ever written to the ledger.
"""

import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

ZERO_HASH = "0" * 64


def sha256_json(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def model_fingerprint(weights, bias) -> str:
    h = hashlib.sha256(np.asarray(weights, dtype=np.float64).tobytes())
    h.update(np.float64(bias).tobytes())
    return h.hexdigest()


@dataclass
class Block:
    index: int
    timestamp: str
    fl_round: int
    contributions: dict
    total_samples: int
    metrics: dict
    model_hash: str
    previous_hash: str
    block_hash: str = field(default="")

    def compute_hash(self) -> str:
        payload = asdict(self)
        payload.pop("block_hash")
        return sha256_json(payload)

    def __post_init__(self):
        if not self.block_hash:
            self.block_hash = self.compute_hash()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuditLedger:
    def __init__(self, run_config: Optional[dict] = None):
        genesis = Block(
            index=0,
            timestamp=_now(),
            fl_round=0,
            contributions={},
            total_samples=0,
            metrics={"run_config": run_config or {}},
            model_hash=ZERO_HASH,
            previous_hash=ZERO_HASH,
        )
        self.chain: list[Block] = [genesis]

    def __len__(self) -> int:
        return len(self.chain)

    @property
    def head(self) -> str:
        return self.chain[-1].block_hash

    def add_round(self, record: dict, weights, bias) -> Block:
        """Append a block for a finished training round (``record`` from run_federated)."""
        metrics = {k: v for k, v in record["metrics"].items() if isinstance(v, (int, float))}
        block = Block(
            index=len(self.chain),
            timestamp=_now(),
            fl_round=record["round"],
            contributions=dict(record["contributions"]),
            total_samples=record["total_samples"],
            metrics=metrics,
            model_hash=model_fingerprint(weights, bias),
            previous_hash=self.head,
        )
        self.chain.append(block)
        return block

    def verify(self) -> tuple[bool, str, Optional[int]]:
        """Recompute every hash and link. Returns (valid, message, first_bad_index)."""
        for i, block in enumerate(self.chain):
            expected_prev = ZERO_HASH if i == 0 else self.chain[i - 1].block_hash
            if block.index != i:
                return False, f"Block {i} has an out-of-sequence index ({block.index}).", i
            if block.previous_hash != expected_prev:
                return False, f"Block {i} does not link to block {i - 1}.", i
            if block.compute_hash() != block.block_hash:
                return False, f"Block {i} contents do not match its hash (tampered).", i
        return True, f"All {len(self.chain)} blocks verified.", None

    def tamper(self, index: int, field_name: str, value) -> None:
        """Demo only: silently edit a block field or metric without updating its hash."""
        block = self.chain[index]
        if hasattr(block, field_name):
            setattr(block, field_name, value)
        else:
            block.metrics[field_name] = value

    def rehash_from(self, index: int) -> None:
        """Demo only: an attacker recomputes hashes so the local replica verifies again."""
        for i in range(index, len(self.chain)):
            if i:
                self.chain[i].previous_hash = self.chain[i - 1].block_hash
            self.chain[i].block_hash = self.chain[i].compute_hash()

    def copy(self) -> "AuditLedger":
        return copy.deepcopy(self)

    def to_list(self) -> list[dict]:
        return [asdict(b) for b in self.chain]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_list(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "AuditLedger":
        ledger = cls.__new__(cls)
        ledger.chain = [Block(**b) for b in json.loads(Path(path).read_text())]
        return ledger


def consensus(replicas: dict[str, AuditLedger]) -> dict:
    """
    Compare replicas held by each node. Only replicas that verify locally get a vote; a replica
    is trusted if it verifies and its head hash matches the majority of votes.
    """
    checks = {node: ledger.verify() for node, ledger in replicas.items()}
    heads = {node: replicas[node].head for node, (valid, _, _) in checks.items() if valid}
    majority_head, votes = Counter(heads.values()).most_common(1)[0] if heads else (None, 0)
    status = {}
    for node, (valid, message, bad_index) in checks.items():
        agrees = valid and heads[node] == majority_head
        status[node] = {"valid": valid, "agrees": agrees, "message": message, "bad_index": bad_index}
    return {
        "majority_head": majority_head,
        "votes": votes,
        "nodes": len(replicas),
        "quorum": votes > len(replicas) / 2,
        "status": status,
    }
