"""Tests for NYU timestamp parsing."""

from room_vlm.nyu.timestamps import parse_timestamp_from_filename, timestamp_or_fallback


def test_parse_nyu_rgb_filename():
    result = parse_timestamp_from_filename("r-1294886362.238178-3118787619.ppm")
    assert result.ok
    assert result.timestamp == 1294886362.238178
    assert result.sequence == "3118787619"
    assert not result.fallback_used


def test_unparseable_uses_fallback():
    ts, used = timestamp_or_fallback("not-a-nyu-file.jpg", frame_index=7)
    assert used is True
    assert ts == 7.0
