"""Training metrics helpers."""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from room_vlm.constants import CANONICAL_LABELS, INVALID_LABEL


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> dict:
    labels = list(labels)
    invalid_rate = sum(1 for p in y_pred if p == INVALID_LABEL) / max(len(y_pred), 1)
    # Map invalid to a sentinel not in labels for sklearn metrics on valid subset? Keep as-is:
    # invalid predictions count as wrong.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = {}
    for i, lab in enumerate(labels):
        per_class[lab] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "invalid_rate": float(invalid_rate),
        "per_class": per_class,
    }
