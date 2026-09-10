#!/usr/bin/env python3
"""Predict room label for a video via frame aggregation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room_vlm.config import load_config
from room_vlm.inference.video import predict_video
from room_vlm.model.loader import attach_adapter, load_model_and_processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--json", action="store_true", default=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    loaded = load_model_and_processor(
        cfg["model"]["name"],
        quantization={"enabled": False},
        allow_bf16_lora=True,
    )
    model = loaded.model
    if args.checkpoint:
        model = attach_adapter(model, str(args.checkpoint))
    model.eval()

    inf = cfg.get("inference", {})
    result = predict_video(
        model,
        loaded.processor,
        args.video,
        interval_seconds=float(inf.get("video_interval_seconds", 1.0)),
        max_frames=int(inf.get("max_video_frames", 32)),
        aggregation=str(inf.get("aggregation", "mean_log_probability")),
        video_strategy=str(inf.get("video_strategy", "frame_aggregation")),
        length_normalize=bool(inf.get("length_normalize", True)),
        abstention_enabled=bool(inf.get("abstention_enabled", False)),
        abstention_threshold=inf.get("abstention_threshold"),
        include_temporal=True,
        device=loaded.device,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
