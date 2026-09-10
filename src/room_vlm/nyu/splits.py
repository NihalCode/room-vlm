"""Scene-level stratified train/validation/test splits (no frame leakage)."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from room_vlm.constants import CANONICAL_LABELS, DEFAULT_SEED
from room_vlm.nyu.discovery import SceneInfo

logger = logging.getLogger(__name__)


class SceneLeakageError(RuntimeError):
    """Raised when the same scene_id appears in more than one split."""


@dataclass(frozen=True)
class SplitAssignment:
    scene_id: str
    label: str
    split: str


def _stratified_scene_split(
    scenes: Sequence[SceneInfo],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    by_label: dict[str, list[SceneInfo]] = defaultdict(list)
    for scene in scenes:
        by_label[scene.label].append(scene)

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}

    for label in CANONICAL_LABELS:
        group = sorted(by_label.get(label, []), key=lambda s: s.scene_id)
        if not group:
            logger.warning("No scenes found for label %s", label)
            continue
        indices = np.arange(len(group))
        rng.shuffle(indices)
        n = len(group)
        if n == 1:
            n_train, n_val, n_test = 0, 0, 1
        elif n == 2:
            # Prefer train+test so evaluation always has an unseen scene.
            n_train, n_val, n_test = 1, 0, 1
        elif n == 3:
            n_train, n_val, n_test = 1, 1, 1
        else:
            n_test = max(1, int(round(n * test_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            if n_test + n_val >= n:
                n_val = max(0, n - n_test - 1)
            if n_test + n_val >= n:
                n_test = max(1, n - 1)
                n_val = 0
            n_train = n - n_test - n_val
        if n_train < 0:
            raise RuntimeError(f"Invalid split sizes for label={label}: n={n}")

        ordered = [group[i] for i in indices]
        for scene in ordered[:n_train]:
            assignment[scene.scene_id] = "train"
        for scene in ordered[n_train : n_train + n_val]:
            assignment[scene.scene_id] = "validation"
        for scene in ordered[n_train + n_val :]:
            assignment[scene.scene_id] = "test"

        if "test" not in {assignment[s.scene_id] for s in group}:
            # Force last scene into test for this class.
            assignment[group[-1].scene_id] = "test"

    return assignment


def assert_no_scene_leakage(split_map: Mapping[str, str]) -> None:
    """Assert that each scene_id maps to exactly one split (invariant)."""
    # split_map is already scene_id -> split; check inverse sets are disjoint.
    by_split: dict[str, set[str]] = defaultdict(set)
    for scene_id, split in split_map.items():
        by_split[split].add(scene_id)
    train = by_split.get("train", set())
    val = by_split.get("validation", set()) | by_split.get("val", set())
    test = by_split.get("test", set())
    if train & val:
        raise SceneLeakageError(f"Train/val scene overlap: {sorted(train & val)}")
    if train & test:
        raise SceneLeakageError(f"Train/test scene overlap: {sorted(train & test)}")
    if val & test:
        raise SceneLeakageError(f"Val/test scene overlap: {sorted(val & test)}")


def assert_at_least_one_test_per_class(
    scenes: Sequence[SceneInfo],
    split_map: Mapping[str, str],
) -> None:
    for label in CANONICAL_LABELS:
        test_scenes = [
            s.scene_id
            for s in scenes
            if s.label == label and split_map.get(s.scene_id) == "test"
        ]
        if not test_scenes:
            # Only enforce when the class exists in the dataset.
            if any(s.label == label for s in scenes):
                raise SceneLeakageError(f"No test scene for label={label}")


def split_scenes(
    scenes: Sequence[SceneInfo],
    train: float = 0.70,
    validation: float = 0.15,
    test: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """
    Stratified scene-level split.

    Returns mapping scene_id -> split name in {train, validation, test}.
    """
    assignment = _stratified_scene_split(scenes, train, validation, test, seed)
    assert_no_scene_leakage(assignment)
    assert_at_least_one_test_per_class(scenes, assignment)
    logger.info(
        "Split scenes: train=%d val=%d test=%d",
        sum(1 for v in assignment.values() if v == "train"),
        sum(1 for v in assignment.values() if v == "validation"),
        sum(1 for v in assignment.values() if v == "test"),
    )
    return assignment


def scene_ids_by_split(split_map: Mapping[str, str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for scene_id, split in split_map.items():
        key = "validation" if split == "val" else split
        out[key].add(scene_id)
    return dict(out)


def validate_frame_manifest_against_scenes(
    frame_scene_ids: Iterable[str],
    split_map: Mapping[str, str],
) -> None:
    """Refuse training if any frame's scene is missing or leaked."""
    assert_no_scene_leakage(split_map)
    unknown = sorted({sid for sid in frame_scene_ids if sid not in split_map})
    if unknown:
        raise SceneLeakageError(f"Frames reference unknown scene_ids: {unknown[:10]}")
