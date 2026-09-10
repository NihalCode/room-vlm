#!/usr/bin/env python3
"""Predict room label for a single image."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room_vlm.config import load_config
from room_vlm.inference.image import predict_image
from room_vlm.model.loader import attach_adapter, load_model_and_processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
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
    result = predict_image(
        model,
        loaded.processor,
        args.image,
        strategy=inf.get("strategy", "candidate_scoring"),
        length_normalize=bool(inf.get("length_normalize", True)),
        abstention_enabled=bool(inf.get("abstention_enabled", False)),
        abstention_threshold=inf.get("abstention_threshold"),
        device=loaded.device,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nClass scores:")
    for lab, score in result["scores"].items():
        print(f"  {lab:12s} {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
