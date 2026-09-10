"""Evaluation metrics including calibration-related scores."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from room_vlm.constants import CANONICAL_LABELS, INVALID_LABEL
from room_vlm.training.metrics import classification_metrics


def top2_accuracy(y_true: Sequence[str], ranked_labels: Sequence[Sequence[str]]) -> float:
    hits = 0
    for truth, ranked in zip(y_true, ranked_labels):
        if truth in ranked[:2]:
            hits += 1
    return hits / max(len(y_true), 1)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """ECE over equal-width confidence bins. Treats confidences as candidate likelihoods."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1] if i < n_bins - 1 else confidences <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def brier_score_multiclass(
    y_true: Sequence[str],
    score_dicts: Sequence[dict[str, float]],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> float:
    labels = list(labels)
    total = 0.0
    for truth, scores in zip(y_true, score_dicts):
        for lab in labels:
            p = float(scores.get(lab, 0.0))
            y = 1.0 if lab == truth else 0.0
            total += (p - y) ** 2
    return total / max(len(y_true), 1)


def negative_log_likelihood(
    y_true: Sequence[str],
    score_dicts: Sequence[dict[str, float]],
    eps: float = 1e-12,
) -> float:
    nll = 0.0
    for truth, scores in zip(y_true, score_dicts):
        p = max(float(scores.get(truth, 0.0)), eps)
        nll += -np.log(p)
    return float(nll / max(len(y_true), 1))


def full_evaluation_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    score_dicts: Sequence[dict[str, float]] | None = None,
    ranked_labels: Sequence[Sequence[str]] | None = None,
    latencies_ms: Sequence[float] | None = None,
    labels: Sequence[str] = CANONICAL_LABELS,
) -> dict:
    base = classification_metrics(y_true, y_pred, labels=labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    out = {
        **base,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
    }
    if ranked_labels is not None:
        out["top2_accuracy"] = float(top2_accuracy(y_true, ranked_labels))
    if score_dicts is not None:
        confidences = [float(max(s.values()) if s else 0.0) for s in score_dicts]
        correct = [p == t for p, t in zip(y_pred, y_true)]
        out["nll"] = negative_log_likelihood(y_true, score_dicts)
        out["brier"] = brier_score_multiclass(y_true, score_dicts, labels=labels)
        out["ece"] = expected_calibration_error(confidences, correct)
        out["calibration_note"] = (
            "NLL/Brier/ECE treat softmaxed candidate likelihoods as probabilities; "
            "they are not necessarily calibrated real-world probabilities."
        )
    if latencies_ms is not None and len(latencies_ms) > 0:
        arr = np.asarray(latencies_ms, dtype=float)
        out["latency_ms_mean"] = float(arr.mean())
        out["latency_ms_p50"] = float(np.percentile(arr, 50))
        out["examples_per_second"] = float(1000.0 / max(arr.mean(), 1e-6))
    return out
