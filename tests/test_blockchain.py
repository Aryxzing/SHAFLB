import numpy as np

from src.blockchain import AuditLedger, consensus


def make_ledger(rounds: int = 3) -> AuditLedger:
    ledger = AuditLedger(run_config={"algorithm": "FedAvg"})
    for r in range(1, rounds + 1):
        record = {
            "round": r,
            "total_samples": 300,
            "contributions": {"H1": 100, "H2": 200},
            "metrics": {"f1": 0.5 + r / 100, "roc_auc": 0.7, "confusion_matrix": {"tp": 1}},
        }
        ledger.add_round(record, np.ones(3) * r, 0.1)
    return ledger


def test_chain_links_and_verifies():
    ledger = make_ledger()
    assert len(ledger) == 4
    for prev, block in zip(ledger.chain, ledger.chain[1:]):
        assert block.previous_hash == prev.block_hash
    assert "confusion_matrix" not in ledger.chain[1].metrics
    assert ledger.verify() == (True, "All 4 blocks verified.", None)


def test_tampering_is_detected_at_the_edited_block():
    ledger = make_ledger()
    ledger.tamper(2, "total_samples", 999_999)
    valid, message, bad = ledger.verify()
    assert not valid and bad == 2 and "tampered" in message
    ledger.tamper(1, "f1", 0.99)
    assert ledger.verify()[2] == 1


def test_rehashing_fools_local_check_but_not_consensus():
    original = make_ledger()
    replicas = {f"H{i}": original.copy() for i in range(5)}
    attacked = replicas["H3"]
    attacked.tamper(1, "total_samples", 1)
    attacked.rehash_from(1)
    assert attacked.verify()[0]  # locally consistent again
    result = consensus(replicas)
    assert result["quorum"] and result["votes"] == 4
    assert not result["status"]["H3"]["agrees"]
    assert all(result["status"][n]["agrees"] for n in ("H0", "H1", "H2", "H4"))


def test_save_load_roundtrip(tmp_path):
    ledger = make_ledger()
    path = tmp_path / "chain.json"
    ledger.save(path)
    loaded = AuditLedger.load(path)
    assert loaded.head == ledger.head and loaded.verify()[0]


def test_broken_replica_does_not_vote():
    original = make_ledger()
    replicas = {f"H{i}": original.copy() for i in range(5)}
    replicas["H1"].tamper(2, "f1", 0.99)  # edited without re-hashing: head unchanged but invalid
    result = consensus(replicas)
    assert result["votes"] == 4
    assert not result["status"]["H1"]["valid"] and not result["status"]["H1"]["agrees"]
