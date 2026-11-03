import numpy as np
import pytest

from src.config import HOSPITAL_IDS, FLConfig
from src.federated import HospitalClient, partition_non_iid, run_federated


@pytest.fixture
def toy():
    rng = np.random.RandomState(1)
    X = rng.randn(1000, 6)
    true_w = np.array([1.5, -1.0, 0.5, 0, 0, 0])
    y = (rng.rand(1000) < 1 / (1 + np.exp(-(X @ true_w - 2)))).astype(int)
    return X[:800], y[:800], X[800:], y[800:]


def test_partition_is_disjoint_and_non_iid(toy):
    X, y, *_ = toy
    clients = partition_non_iid(X, y)
    assert [c.hospital_id for c in clients] == HOSPITAL_IDS
    assert sum(c.n_samples for c in clients) == len(y)
    assert sum(c.y.sum() for c in clients) == y.sum()
    rates = [c.summary()["readmission_rate_pct"] for c in clients]
    assert max(rates) - min(rates) > 3  # heterogeneous label distributions


def test_fedavg_learns_and_is_deterministic(toy):
    X, y, Xt, yt = toy
    clients = partition_non_iid(X, y)
    a = run_federated(clients, Xt, yt, FLConfig(rounds=4))
    b = run_federated(clients, Xt, yt, FLConfig(rounds=4))
    assert np.array_equal(a["weights"], b["weights"])
    assert a["history"][-1]["metrics"]["roc_auc"] > 0.75
    assert a["weights"][0] > 0 > a["weights"][1]


def test_single_client_fedavg_equals_local_training(toy):
    X, y, Xt, yt = toy
    client = HospitalClient("H", X, y.astype(float))
    cfg = FLConfig(rounds=1)
    run = run_federated([client], Xt, yt, cfg)
    pos_weight = (1 - y.mean()) / y.mean()
    w, b = client.local_update(np.zeros(X.shape[1]), 0.0, cfg, pos_weight, seed=cfg.seed + 10)
    assert np.allclose(run["weights"], w) and np.isclose(run["bias"], b)


def test_aggregation_is_sample_weighted(toy):
    X, y, Xt, yt = toy
    big, small = HospitalClient("big", X[:600], y[:600].astype(float)), HospitalClient("s", X[600:], y[600:].astype(float))
    cfg = FLConfig(rounds=1)
    run = run_federated([big, small], Xt, yt, cfg)
    pw = (1 - y.mean()) / y.mean()
    wb, _ = big.local_update(np.zeros(6), 0.0, cfg, pw, cfg.seed + 10)
    ws, _ = small.local_update(np.zeros(6), 0.0, cfg, pw, cfg.seed + 10)
    assert np.allclose(run["weights"], 0.75 * wb + 0.25 * ws)


def test_clipping_bounds_update_and_callback_fires(toy):
    X, y, Xt, yt = toy
    seen = []
    run = run_federated(partition_non_iid(X, y), Xt, yt, FLConfig(rounds=2, clip_norm=0.05),
                        on_round=lambda rec, w, b: seen.append(rec["round"]))
    assert seen == [1, 2]
    assert np.linalg.norm(np.append(run["weights"], run["bias"])) <= 0.1 + 1e-9


def test_fedprox_stays_closer_to_global(toy):
    X, y, *_ = toy
    client = HospitalClient("H", X, y.astype(float))
    w0 = np.zeros(6)
    w_avg, _ = client.local_update(w0, 0.0, FLConfig(), 10.0, 0)
    w_prox, _ = client.local_update(w0, 0.0, FLConfig(algorithm="fedprox", prox_mu=1.0), 10.0, 0)
    assert np.linalg.norm(w_prox) < np.linalg.norm(w_avg)
