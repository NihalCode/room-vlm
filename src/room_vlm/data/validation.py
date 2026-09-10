"""Dataset integrity validation (scene leakage, missing files, label checks)."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from room_vlm.constants import CANONICAL_LABELS
from room_vlm.data.dataset import read_jsonl
from room_vlm.nyu.splits import SceneLeakageError, assert_no_scene_leakage, scene_ids_by_split

logger = logging.getLogger(__name__)


def validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    check_files: bool = True,
    require_splits: bool = True,
) -> dict[str, Any]:
    """Validate a list of example records. Raises on fatal issues."""
    if not records:
        raise ValueError("No records to validate")

    scene_to_split: dict[str, str] = {}
    issues: list[str] = []
    missing = 0
    for rec in records:
        for key in ("image", "label", "scene_id"):
            if key not in rec:
                raise ValueError(f"Record missing {key}: {rec}")
        if rec["label"] not in CANONICAL_LABELS:
            raise ValueError(f"Invalid label {rec['label']!r} for scene {rec['scene_id']}")
        split = rec.get("split")
        if require_splits and not split:
            raise ValueError(f"Record missing split: {rec['scene_id']}")
        if split:
            prev = scene_to_split.get(rec["scene_id"])
            if prev is None:
                scene_to_split[rec["scene_id"]] = split
            elif prev != split:
                raise SceneLeakageError(
                    f"Scene {rec['scene_id']} appears in both {prev} and {split}"
                )
        if check_files and not Path(rec["image"]).exists():
            missing += 1
            issues.append(f"missing image: {rec['image']}")

    assert_no_scene_leakage(scene_to_split)

    # Frame-level: all frames of a scene must share one split (already checked).
    by_split = scene_ids_by_split(scene_to_split)
    report = {
        "num_records": len(records),
        "num_scenes": len(scene_to_split),
        "scenes_per_split": {k: len(v) for k, v in by_split.items()},
        "missing_images": missing,
        "ok": missing == 0,
    }
    if missing:
        logger.error("Dataset validation found %d missing images", missing)
        raise FileNotFoundError(
            f"{missing} image paths missing. First: {issues[0] if issues else ''}"
        )
    logger.info("Dataset validation OK: %s", report)
    return report


def validate_split_jsonl_dir(processed_root: str | Path) -> dict[str, Any]:
    processed_root = Path(processed_root)
    all_records: list[dict[str, Any]] = []
    for split_name, filename in (
        ("train", "train.jsonl"),
        ("validation", "val.jsonl"),
        ("test", "test.jsonl"),
    ):
        path = processed_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        rows = read_jsonl(path)
        for row in rows:
            row = dict(row)
            row.setdefault("split", split_name if split_name != "validation" else "validation")
            # Normalize val naming
            if filename == "val.jsonl":
                row["split"] = "validation"
            all_records.append(row)

    # Cross-file scene leakage check
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for rec in all_records:
        scene_splits[rec["scene_id"]].add(rec["split"])
    leaked = {sid: splits for sid, splits in scene_splits.items() if len(splits) > 1}
    if leaked:
        raise SceneLeakageError(f"Scene leakage across JSONL splits: {leaked}")

    return validate_records(all_records, check_files=True, require_splits=True)
