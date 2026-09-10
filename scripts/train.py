#!/usr/bin/env python3
"""Fine-tune Qwen3-VL with LoRA/QLoRA for room classification."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room_vlm.config import load_config
from room_vlm.training.trainer import RoomVLMTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora.yaml"))
    parser.add_argument(
        "--allow-bf16-lora",
        action="store_true",
        help="If 4-bit QLoRA is unavailable, explicitly fall back to BF16 LoRA (never silent).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    trainer = RoomVLMTrainer(cfg, allow_bf16_lora=args.allow_bf16_lora)
    adapter = trainer.train()
    print(f"Training complete. Adapter saved to: {adapter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
