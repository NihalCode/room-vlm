"""Tests for scene-level split leakage prevention."""

from pathlib import Path

import pytest

from room_vlm.data.validation import validate_records
from room_vlm.nyu.discovery import discover_scenes
from room_vlm.nyu.splits import SceneLeakageError, assert_no_scene_leakage, scene_ids_by_split, split_scenes

FIXTURES = Path(__file__).parent / "fixtures" / "nyu"


def test_scene_split_no_leakage():
    scenes = discover_scenes(FIXTURES)
    split_map = split_scenes(scenes, train=0.5, validation=0.25, test=0.25, seed=42)
    by = scene_ids_by_split(split_map)
    train, val, test = by.get("train", set()), by.get("validation", set()), by.get("test", set())
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert_no_scene_leakage(split_map)


def test_leakage_detection():
    records = [
        {"image": "x.jpg", "label": "bedroom", "scene_id": "bedroom_0001", "split": "train"},
        {"image": "y.jpg", "label": "bedroom", "scene_id": "bedroom_0001", "split": "test"},
    ]
    with pytest.raises(SceneLeakageError):
        validate_records(records, check_files=False)
