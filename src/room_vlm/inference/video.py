"""Video inference via frame sampling and score aggregation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from room_vlm.constants import CANONICAL_LABELS
from room_vlm.inference.aggregation import aggregate_frame_scores
from room_vlm.inference.image import predict_image

logger = logging.getLogger(__name__)


def sample_video_frames(
    video_path: str | Path,
    *,
    interval_seconds: float = 1.0,
    max_frames: int = 32,
) -> list[tuple[int, float, Image.Image]]:
    """Sample frames every ``interval_seconds`` up to ``max_frames``."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless is required for video inference") from exc

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = 30.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    logger.info(
        "Video %s: fps=%.2f frames=%d duration=%.2fs",
        video_path.name,
        fps,
        frame_count,
        duration,
    )

    step = max(int(round(fps * interval_seconds)), 1)
    frames: list[tuple[int, float, Image.Image]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            timestamp = idx / fps
            frames.append((idx, timestamp, image))
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def predict_video(
    model: Any,
    processor: Any,
    video_path: str | Path,
    *,
    interval_seconds: float = 1.0,
    max_frames: int = 32,
    aggregation: str = "mean_log_probability",
    video_strategy: str = "frame_aggregation",
    labels: list[str] | None = None,
    length_normalize: bool = True,
    abstention_enabled: bool = False,
    abstention_threshold: float | None = None,
    include_temporal: bool = True,
    device=None,
) -> dict[str, Any]:
    labels = labels or list(CANONICAL_LABELS)

    if video_strategy == "direct_vlm":
        logger.warning(
            "direct_vlm video strategy is experimental; falling back to frame_aggregation "
            "because fine-tuning was frame-based."
        )

    sampled = sample_video_frames(
        video_path, interval_seconds=interval_seconds, max_frames=max_frames
    )
    frame_results = []
    for frame_index, timestamp, image in sampled:
        pred = predict_image(
            model,
            processor,
            image,
            strategy="candidate_scoring",
            labels=labels,
            length_normalize=length_normalize,
            abstention_enabled=False,
            device=device,
        )
        frame_results.append(
            {
                "label": pred["label"],
                "confidence": pred["confidence"],
                "scores": pred["scores"],
                "frame_index": frame_index,
                "timestamp": timestamp,
            }
        )

    aggregated = aggregate_frame_scores(frame_results, labels=labels, method=aggregation)
    if abstention_enabled and abstention_threshold is not None:
        if aggregated["confidence"] < float(abstention_threshold):
            aggregated["prediction"] = "unknown"
            aggregated["abstained"] = True
        else:
            aggregated["abstained"] = False
    else:
        aggregated["abstained"] = False

    if not include_temporal:
        aggregated.pop("temporal_predictions", None)
    return aggregated
