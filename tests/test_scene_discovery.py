"""Tests for scene discovery and label mapping."""

from pathlib import Path

from room_vlm.nyu.discovery import discover_scenes, label_from_scene_id, normalize_label, parse_scene_dirname

FIXTURES = Path(__file__).parent / "fixtures" / "nyu"


def test_label_mapping_examples():
    assert normalize_label("bedroom") == "bedroom"
    assert label_from_scene_id("bedroom_0001") == "bedroom"
    assert label_from_scene_id("bathroom_0012") == "bathroom"
    assert label_from_scene_id("kitchen_0005") == "kitchen"
    assert label_from_scene_id("living_room_0012") == "living room"
    assert normalize_label("living_room") == "living room"


def test_parse_scene_dirname():
    assert parse_scene_dirname("bedroom_0001") == ("bedroom", "bedroom_0001")
    assert parse_scene_dirname("living_room_0012") == ("living_room", "living_room_0012")
    assert parse_scene_dirname("office_0001") is None


def test_discover_fixture_scenes():
    scenes = discover_scenes(FIXTURES)
    ids = {s.scene_id for s in scenes}
    assert "bedroom_0001" in ids
    assert "living_room_0001" in ids
    assert len(scenes) == 8
    labels = {s.label for s in scenes}
    assert labels == {"bedroom", "living room", "bathroom", "kitchen"}
