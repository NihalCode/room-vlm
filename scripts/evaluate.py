#!/usr/bin/env python3
"""Evaluate a LoRA adapter (or base model) on a held-out split."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room_vlm.config import load_config
from room_vlm.evaluation.evaluator import compare_baseline_and_finetuned, evaluate_split
from room_vlm.model.loader import attach_adapter, load_model_and_processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "val"])
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    processed = Path(cfg["dataset"]["processed_root"])
    split = "val" if args.split in {"val", "validation"} else args.split
    jsonl = processed / ("val.jsonl" if split == "val" else f"{split}.jsonl")
    if not jsonl.exists():
        logger.error("Missing %s", jsonl)
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

    metrics = evaluate_split(
        model,
        loaded.processor,
        jsonl,
        args.output,
        strategy=cfg.get("inference", {}).get("strategy", "candidate_scoring"),
        labels=list(cfg.get("labels", [])),
        length_normalize=bool(cfg.get("inference", {}).get("length_normalize", True)),
        device=loaded.device,
    )
    print("Evaluation metrics:")
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "invalid_rate", "ece"):
        if key in metrics:
            print(f"  {key}: {metrics[key]}")

    baseline_metrics = Path("outputs/baseline/metrics.json")
    if baseline_metrics.exists() and (args.output / "metrics.json").exists():
        compare_baseline_and_finetuned(
            baseline_metrics,
            args.output / "metrics.json",
            args.output / "comparison.json",
        )
        print(f"Wrote comparison to {args.output / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
