#!/usr/bin/env python3
"""Calibrate abstention threshold on validation data only (never on test)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room_vlm.config import load_config
from room_vlm.data.dataset import read_jsonl
from room_vlm.model.loader import attach_adapter, load_model_and_processor
from room_vlm.model.scoring import score_image_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("calibrate_threshold")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/abstention_threshold.json"))
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.9,
        help="Keep predictions covering this fraction of val; rest may abstain.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    val_path = Path(cfg["dataset"]["processed_root"]) / "val.jsonl"
    if not val_path.exists():
        logger.error("Missing validation split: %s", val_path)
        return 1

    loaded = load_model_and_processor(
        cfg["model"]["name"],
        quantization={"enabled": False},
        allow_bf16_lora=True,
    )
    model = loaded.model
    if args.checkpoint:
        model = attach_adapter(model, str(args.checkpoint))
    model.eval()

    records = read_jsonl(val_path)
    confidences: list[float] = []
    correct: list[bool] = []
    for rec in records:
        with Image.open(rec["image"]) as img:
            image = img.convert("RGB")
        scored = score_image_labels(
            model,
            loaded.processor,
            image,
            labels=list(cfg.get("labels", [])),
            device=loaded.device,
        )
        confidences.append(float(scored["confidence"]))
        correct.append(scored["label"] == rec["label"])

    # Choose threshold as the confidence quantile so that target_coverage is retained.
    import numpy as np

    confidences_arr = np.asarray(confidences)
    threshold = float(np.quantile(confidences_arr, 1.0 - args.target_coverage))
    retained = confidences_arr >= threshold
    accuracy_retained = float(np.mean([c for c, keep in zip(correct, retained) if keep])) if retained.any() else None

    payload = {
        "threshold": threshold,
        "target_coverage": args.target_coverage,
        "n_val": len(records),
        "retained_fraction": float(retained.mean()) if len(retained) else 0.0,
        "accuracy_on_retained": accuracy_retained,
        "split_used": "validation",
        "note": "Threshold selected on validation only. Do not tune on test.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
