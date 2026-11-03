"""
Federated learning simulation: non-IID hospital partitioning, local SGD on
private data, and server-side aggregation (FedAvg, FedProx, optional DP).

Only model parameters and sample counts leave a hospital. Raw records stay in
each HospitalClient.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.config import HOSPITALS, RANDOM_SEED, FLConfig
from src.models import compute_metrics


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))


@dataclass
class HospitalClient:
    hospital_id: str
    X: np.ndarray
    y: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.y)

    def summary(self) -> dict:
        pos = int(self.y.sum())
        return {
            "hospital_id": self.hospital_id,
            "profile": HOSPITALS.get(self.hospital_id, {}).get("profile", ""),
            "total_samples": self.n_samples,
            "positive_samples": pos,
            "negative_samples": self.n_samples - pos,
            "readmission_rate_pct": round(100 * pos / self.n_samples, 2) if self.n_samples else 0.0,
        }

    def local_update(self, w_global, b_global, cfg: FLConfig, pos_weight: float, seed: int):
        """Class-weighted mini-batch SGD for logistic regression, started from the global model."""
        w, b = w_global.copy(), float(b_global)
        n = self.n_samples
        if n == 0:
            return w, b
        rng = np.random.RandomState(seed + n)
        idx = np.arange(n)
        mu = cfg.prox_mu if cfg.algorithm == "fedprox" else 0.0
        for _ in range(cfg.local_epochs):
            rng.shuffle(idx)
            for start in range(0, n, cfg.batch_size):
                batch = idx[start : start + cfg.batch_size]
                Xb, yb = self.X[batch], self.y[batch]
                err = (sigmoid(Xb @ w + b) - yb) * np.where(yb == 1, pos_weight, 1.0)
                grad_w = Xb.T @ err / len(batch) + cfg.l2_reg * w
                grad_b = err.mean()
                if mu:  # FedProx proximal term keeps local models near the global one
                    grad_w += mu * (w - w_global)
                    grad_b += mu * (b - b_global)
                w -= cfg.learning_rate * grad_w
                b -= cfg.learning_rate * grad_b
        return w, b


def partition_non_iid(X, y, seed: int = RANDOM_SEED) -> list[HospitalClient]:
    """Split the training set across hospitals with different sizes and readmission rates."""
    rng = np.random.RandomState(seed)
    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def split(indices, key):
        shares = np.array([h[key] for h in HOSPITALS.values()])
        cuts = (np.cumsum(shares / shares.sum())[:-1] * len(indices)).astype(int)
        return np.split(indices, cuts)

    clients = []
    for hid, p, q in zip(HOSPITALS, split(pos_idx, "pos_share"), split(neg_idx, "neg_share")):
        idx = np.concatenate([p, q])
        rng.shuffle(idx)
        clients.append(HospitalClient(hid, X[idx], y[idx].astype(np.float64)))
    return clients


def predict_proba(X, weights, bias) -> np.ndarray:
    return sigmoid(X @ weights + bias)


def run_federated(
    clients: list[HospitalClient],
    X_test,
    y_test,
    cfg: FLConfig = FLConfig(),
    on_round: Optional[Callable[[dict, np.ndarray, float], None]] = None,
) -> dict:
    """
    Run ``cfg.rounds`` of federated training and evaluate the global model after each round.

    Aggregation is sample-weighted: w = sum_k (n_k / N) * w_k. When ``clip_norm`` > 0 each
    client's update is clipped to that L2 norm; ``noise_multiplier`` > 0 then adds Gaussian
    noise scaled to the aggregate's sensitivity (Gaussian mechanism, no formal epsilon accounting).
    """
    d = clients[0].X.shape[1]
    w, b = np.zeros(d), 0.0
    counts = np.array([c.n_samples for c in clients], dtype=float)
    shares = counts / counts.sum()
    # Clients share only their label counts, from which the server derives the class weight.
    pos_rate = sum(c.y.sum() for c in clients) / counts.sum()
    pos_weight = float((1 - pos_rate) / pos_rate) if pos_rate > 0 else 1.0
    noise_rng = np.random.RandomState(cfg.seed + 7919)
    history = []

    for rnd in range(1, cfg.rounds + 1):
        deltas = []
        for client in clients:
            w_k, b_k = client.local_update(w, b, cfg, pos_weight, seed=cfg.seed + rnd * 10)
            deltas.append(np.append(w_k - w, b_k - b))
        deltas = np.array(deltas)
        norms = np.linalg.norm(deltas, axis=1)

        if cfg.clip_norm > 0:
            deltas *= np.minimum(1.0, cfg.clip_norm / np.maximum(norms, 1e-12))[:, None]
        update = shares @ deltas
        if cfg.noise_multiplier > 0 and cfg.clip_norm > 0:
            sensitivity = cfg.clip_norm * shares.max()
            update += noise_rng.normal(0.0, cfg.noise_multiplier * sensitivity, size=update.shape)

        w, b = w + update[:-1], b + float(update[-1])
        record = {
            "round": rnd,
            "total_samples": int(counts.sum()),
            "contributions": {c.hospital_id: c.n_samples for c in clients},
            "update_norms": {c.hospital_id: round(float(n), 4) for c, n in zip(clients, norms)},
            "metrics": compute_metrics(y_test, predict_proba(X_test, w, b)),
        }
        history.append(record)
        if on_round:
            on_round(record, w, b)

    return {"weights": w, "bias": b, "history": history, "config": cfg.to_dict()}
