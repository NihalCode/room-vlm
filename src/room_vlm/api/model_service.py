"""Singleton model service loaded once at API startup."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from room_vlm.constants import CANONICAL_LABELS
from room_vlm.inference.image import predict_image
from room_vlm.inference.video import predict_video
from room_vlm.model.loader import attach_adapter, load_model_and_processor

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = None
        self.model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct")
        self.adapter_path = os.environ.get("ADAPTER_PATH") or None
        self.adapter_name = Path(self.adapter_path).name if self.adapter_path else None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        device = os.environ.get("DEVICE", "auto")
        logger.info("Loading model %s (adapter=%s)", self.model_name, self.adapter_name)
        loaded = load_model_and_processor(
            self.model_name,
            quantization={"enabled": False},
            device=device,
            allow_bf16_lora=True,
        )
        model = loaded.model
        if self.adapter_path:
            model = attach_adapter(model, self.adapter_path)
        model.eval()
        self.model = model
        self.processor = loaded.processor
        self.device = loaded.device
        self.loaded = True

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            # Return adapter name only — never absolute host paths.
            "adapter": self.adapter_name,
            "device": str(self.device) if self.device is not None else "unknown",
            "labels": list(CANONICAL_LABELS),
            "strategy": "candidate_scoring",
        }

    def predict_image_file(self, path: Path) -> dict[str, Any]:
        assert self.loaded
        t0 = time.perf_counter()
        threshold = os.environ.get("ABSTENTION_THRESHOLD")
        result = predict_image(
            self.model,
            self.processor,
            path,
            abstention_enabled=bool(threshold),
            abstention_threshold=float(threshold) if threshold else None,
            device=self.device,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "label": result["label"],
            "confidence": result["confidence"],
            "scores": result["scores"],
            "model": self.model_name,
            "adapter": self.adapter_name,
            "latency_ms": latency,
            "abstained": result.get("abstained", False),
        }

    def predict_video_file(self, path: Path, include_temporal: bool = False) -> dict[str, Any]:
        assert self.loaded
        t0 = time.perf_counter()
        max_frames = int(os.environ.get("MAX_VIDEO_FRAMES", "32"))
        threshold = os.environ.get("ABSTENTION_THRESHOLD")
        result = predict_video(
            self.model,
            self.processor,
            path,
            max_frames=max_frames,
            include_temporal=include_temporal,
            abstention_enabled=bool(threshold),
            abstention_threshold=float(threshold) if threshold else None,
            device=self.device,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "label": result["prediction"],
            "confidence": result["confidence"],
            "scores": result["scores"],
            "model": self.model_name,
            "adapter": self.adapter_name,
            "latency_ms": latency,
            "abstained": result.get("abstained", False),
            "frames_used": result.get("frames_used"),
            "temporal_predictions": result.get("temporal_predictions") if include_temporal else None,
        }


# Module-level singleton
model_service = ModelService()
