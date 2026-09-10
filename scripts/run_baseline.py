#!/usr/bin/env python3
"""Zero-shot baseline evaluation of the unmodified base VLM on the test split."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room_vlm.config import load_config
from room_vlm.evaluation.evaluator import evaluate_split
from room_vlm.model.loader import load_model_and_processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_baseline")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "val"])
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    processed = Path(cfg["dataset"]["processed_root"])
    split = "val" if args.split in {"val", "validation"} else args.split
    jsonl = processed / ("val.jsonl" if split == "val" else f"{split}.jsonl")
    if not jsonl.exists():
        logger.error("Missing %s — run prepare_nyu.py first", jsonl)
        return 1

    logger.info("Loading UNMODIFIED base model (no LoRA adapter): %s", cfg["model"]["name"])
    loaded = load_model_and_processor(
        cfg["model"]["name"],
        quantization={"enabled": False},
        allow_bf16_lora=True,
    )
    loaded.model.eval()

    metrics = evaluate_split(
        loaded.model,
        loaded.processor,
        jsonl,
        args.output,
        strategy=cfg.get("inference", {}).get("strategy", "candidate_scoring"),
        labels=list(cfg.get("labels", [])),
        length_normalize=bool(cfg.get("inference", {}).get("length_normalize", True)),
        device=loaded.device,
    )
    print("Baseline metrics:")
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "invalid_rate"):
        print(f"  {key}: {metrics.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
