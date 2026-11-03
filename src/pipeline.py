"""End-to-end experiment: data -> centralised baselines -> federated variants -> audit ledger -> artifacts."""

import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import joblib
import numpy as np

from src.blockchain import AuditLedger
from src.config import (
    ARRAYS_FILE,
    ARTIFACTS_DIR,
    FL_VARIANTS,
    LEDGER_FILE,
    MODEL_BUNDLE_FILE,
    RAW_DATA_FILE,
    RESULTS_FILE,
)
from src.data import apply_cohort_selection, dataset_statistics, load_raw_dataset, prepare_train_test_data
from src.federated import partition_non_iid, predict_proba, run_federated
from src.models import compute_metrics, fit_logistic_regression, fit_random_forest

Progress = Callable[[float, str], None]


def run_pipeline(data_path=RAW_DATA_FILE, save: bool = True, progress: Optional[Progress] = None) -> dict:
    """Run the full experiment. Artifacts are written to ``artifacts/`` only when ``save`` is True."""
    report = progress or (lambda frac, msg: print(f"[{frac:>4.0%}] {msg}"))
    start = time.time()

    report(0.02, "Loading dataset and applying cohort criteria")
    cohort = apply_cohort_selection(load_raw_dataset(data_path))
    stats = dataset_statistics(cohort)

    report(0.10, "Stratified split and preprocessing (fit on training data only)")
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(cohort)

    report(0.18, "Training centralised logistic regression")
    lr = fit_logistic_regression(X_train, y_train)
    report(0.28, "Training centralised random forest")
    rf = fit_random_forest(X_train, y_train)
    probas = {
        "centralized_lr": lr.predict_proba(X_test)[:, 1],
        "centralized_rf": rf.predict_proba(X_test)[:, 1],
    }

    report(0.40, "Partitioning training data across hospitals (non-IID)")
    clients = partition_non_iid(X_train, y_train)

    report(0.45, "Training local-only models (no collaboration)")
    local_only = [
        {"hospital_id": c.hospital_id, "metrics": compute_metrics(
            y_test, fit_logistic_regression(c.X, c.y.astype(int)).predict_proba(X_test)[:, 1])}
        for c in clients
    ]

    fl_runs, ledger, fed_model = {}, None, None
    for i, (name, cfg) in enumerate(FL_VARIANTS.items()):
        report(0.55 + 0.12 * i, f"Federated training: {cfg.label} ({cfg.rounds} rounds)")
        run_ledger = AuditLedger(run_config={"algorithm": cfg.label, **cfg.to_dict()})
        run = run_federated(clients, X_test, y_test, cfg, on_round=lambda r, w, b: run_ledger.add_round(r, w, b))
        probas[name] = predict_proba(X_test, run["weights"], run["bias"])
        fl_runs[name] = {"label": cfg.label, "config": run["config"], "history": run["history"]}
        if name == "fedavg":
            ledger, fed_model = run_ledger, {"weights": run["weights"], "bias": run["bias"]}

    valid, message, _ = ledger.verify()
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": stats,
        "split": {"train": len(y_train), "test": len(y_test), "features": X_train.shape[1]},
        "hospitals": [c.summary() for c in clients],
        "models": {name: compute_metrics(y_test, p) for name, p in probas.items()},
        "fl_runs": fl_runs,
        "local_only": local_only,
        "ledger": {"blocks": len(ledger), "verified": valid, "message": message, "head": ledger.head},
        "runtime_seconds": round(time.time() - start, 2),
    }

    if save:
        report(0.95, "Saving artifacts")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        ledger.save(LEDGER_FILE)
        joblib.dump(
            {
                "preprocessor": preprocessor,
                "feature_names": list(preprocessor.get_feature_names_out()),
                "centralized_lr": lr,
                "federated": fed_model,
            },
            MODEL_BUNDLE_FILE,
        )
        np.savez_compressed(
            ARRAYS_FILE,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            **{f"proba_{k}": v for k, v in probas.items()},
        )
    report(1.0, f"Done in {time.time() - start:.1f}s")
    return results


def artifacts_exist() -> bool:
    return all(p.exists() for p in (RESULTS_FILE, LEDGER_FILE, MODEL_BUNDLE_FILE, ARRAYS_FILE))


def load_results() -> dict:
    return json.loads(RESULTS_FILE.read_text())
