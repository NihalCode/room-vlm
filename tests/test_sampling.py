"""Tests for temporal sampling."""

from pathlib import Path

from room_vlm.nyu.sampling import collect_scene_frames, sample_by_timestamp, sample_every_n_frames

FIXTURES = Path(__file__).parent / "fixtures" / "nyu"


def test_timestamp_sampling_spacing():
    scene = FIXTURES / "bedroom_0001"
    frames = collect_scene_frames(scene)
    assert len(frames) > 10
    sampled = sample_by_timestamp(frames, interval_seconds=1.0)
    assert len(sampled) >= 2
    # Adjacent kept frames should be ~>= 1 second apart (allow tiny float eps)
    for a, b in zip(sampled, sampled[1:]):
        assert b.timestamp + 1e-9 >= a.timestamp + 1.0


def test_every_n_frames():
    scene = FIXTURES / "kitchen_0001"
    frames = collect_scene_frames(scene)
    sampled = sample_every_n_frames(frames, every_n=2)
    assert len(sampled) == (len(frames) + 1) // 2
