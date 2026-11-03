"""Centralised baselines and evaluation metrics."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from src.config import RANDOM_SEED

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "specificity": "Specificity",
}


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Threshold-dependent and ranking metrics for a binary classifier."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def ratio(a, b):
        return float(a / b) if b else 0.0

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    both_classes = len(np.unique(y_true)) == 2
    metrics = {
        "accuracy": ratio(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if both_classes else 0.5,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if both_classes else 0.0,
        "specificity": ratio(tn, tn + fp),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    metrics["threshold"] = threshold
    return metrics


def fit_logistic_regression(X, y, seed: int = RANDOM_SEED) -> LogisticRegression:
    """Class-balanced logistic regression: the reference model FedAvg is compared against."""
    return LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed).fit(X, y)


def fit_random_forest(X, y, seed: int = RANDOM_SEED) -> RandomForestClassifier:
    # n_jobs=1 on purpose: process-based parallelism stalls for minutes when
    # launched from Streamlit or sandboxed shells, while single-core takes ~1.5 s.
    return RandomForestClassifier(
        n_estimators=60,
        max_depth=8,
        min_samples_split=20,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    ).fit(X, y)
