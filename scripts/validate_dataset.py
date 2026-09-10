#!/usr/bin/env python3
"""Validate processed dataset integrity and scene-level split invariants."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room_vlm.config import load_config
from room_vlm.data.validation import validate_split_jsonl_dir
from room_vlm.nyu.splits import SceneLeakageError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("validate_dataset")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--processed-root", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    processed = args.processed_root or Path(cfg["dataset"]["processed_root"])
    try:
        report = validate_split_jsonl_dir(processed)
    except SceneLeakageError as exc:
        logger.error("SCENE LEAKAGE DETECTED: %s", exc)
        return 2
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        return 1

    print(json.dumps(report, indent=2))
    print("Dataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
