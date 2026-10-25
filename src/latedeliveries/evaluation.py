"""Business costs and cost-based decision thresholds."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def order_cost(y_true, flagged):
    """Four dollars per voucher plus fifteen dollars per missed late order."""
    y = np.asarray(y_true, dtype=int)
    flags = np.asarray(flagged, dtype=bool)
    if y.shape != flags.shape:
        raise ValueError("Truth and predictions must have equal shape")
    return int(4 * flags.sum() + 15 * ((y == 1) & ~flags).sum())


def choose_threshold(y_true, probabilities):
    """Select cheapest threshold; ties choose higher threshold (fewer vouchers)."""
    p = np.asarray(probabilities)
    if p.ndim != 1 or len(p) != len(y_true):
        raise ValueError("One probability per observation required")
    return float(min(THRESHOLDS, key=lambda t: (order_cost(y_true, p >= t), -t)))


def score_predictions(y_true, probabilities, threshold):
    """Report threshold-dependent metrics and ROC-AUC."""
    flagged = np.asarray(probabilities) >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, flagged)),
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "f1": float(f1_score(y_true, flagged, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }
