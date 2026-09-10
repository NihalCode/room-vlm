"""Score aggregation for multi-frame video predictions."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

from room_vlm.constants import CANONICAL_LABELS
from room_vlm.model.scoring import softmax_dict


def aggregate_frame_scores(
    frame_results: Sequence[dict[str, Any]],
    labels: Sequence[str] = CANONICAL_LABELS,
    method: str = "mean_log_probability",
) -> dict[str, Any]:
    """
    Aggregate per-frame candidate scores.

    Methods:
      - mean_log_probability: average log(prob) per class, then softmax
        (default; corresponds to a geometric-mean of probabilities)
      - mean_probability: arithmetic mean of probabilities, then renormalize
      - majority_vote: majority of argmax labels; confidence = vote fraction
    """
    labels = list(labels)
    if not frame_results:
        scores = {lab: 0.0 for lab in labels}
        return {
            "prediction": "invalid",
            "confidence": 0.0,
            "scores": scores,
            "frames_used": 0,
            "temporal_predictions": [],
        }

    temporal = []
    for fr in frame_results:
        temporal.append(
            {
                "label": fr.get("label"),
                "confidence": fr.get("confidence"),
                "scores": fr.get("scores"),
                "frame_index": fr.get("frame_index"),
                "timestamp": fr.get("timestamp"),
            }
        )

    if method == "majority_vote":
        votes = Counter(fr.get("label") for fr in frame_results)
        pred, count = votes.most_common(1)[0]
        scores = {lab: votes.get(lab, 0) / len(frame_results) for lab in labels}
        return {
            "prediction": pred,
            "confidence": float(scores.get(pred, 0.0)),
            "scores": scores,
            "frames_used": len(frame_results),
            "temporal_predictions": temporal,
            "aggregation": method,
        }

    if method == "mean_probability":
        acc = {lab: 0.0 for lab in labels}
        for fr in frame_results:
            scores = fr.get("scores") or {}
            for lab in labels:
                acc[lab] += float(scores.get(lab, 0.0))
        n = float(len(frame_results))
        mean_scores = {lab: acc[lab] / n for lab in labels}
        # Renormalize
        total = sum(mean_scores.values()) or 1.0
        probs = {lab: mean_scores[lab] / total for lab in labels}
    else:
        # mean_log_probability (default)
        log_acc = {lab: 0.0 for lab in labels}
        for fr in frame_results:
            scores = fr.get("scores") or {}
            for lab in labels:
                p = max(float(scores.get(lab, 0.0)), 1e-12)
                log_acc[lab] += math.log(p)
        n = float(len(frame_results))
        mean_log = {lab: log_acc[lab] / n for lab in labels}
        probs = softmax_dict(mean_log)

    pred = max(probs, key=probs.get)
    return {
        "prediction": pred,
        "confidence": float(probs[pred]),
        "scores": probs,
        "frames_used": len(frame_results),
        "temporal_predictions": temporal,
        "aggregation": method,
    }
