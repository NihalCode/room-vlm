"""Temporal frame sampling for NYU RGB streams."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from room_vlm.nyu.discovery import list_rgb_files
from room_vlm.nyu.timestamps import timestamp_or_fallback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    timestamp: float
    frame_index: int
    original_filename: str
    used_fallback_timestamp: bool


def sample_by_timestamp(
    frames: list[FrameRecord],
    interval_seconds: float = 1.0,
) -> list[FrameRecord]:
    """
    Greedy temporal sampling: keep the first frame, then the next frame whose
    timestamp is at least ``interval_seconds`` after the last kept timestamp.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")
    if not frames:
        return []
    ordered = sorted(frames, key=lambda f: (f.timestamp, f.frame_index, f.path.name))
    kept = [ordered[0]]
    last_ts = ordered[0].timestamp
    for frame in ordered[1:]:
        if frame.timestamp + 1e-9 >= last_ts + interval_seconds:
            kept.append(frame)
            last_ts = frame.timestamp
    return kept


def sample_every_n_frames(frames: list[FrameRecord], every_n: int = 30) -> list[FrameRecord]:
    if every_n < 1:
        raise ValueError("every_n_frames must be >= 1")
    ordered = sorted(frames, key=lambda f: (f.frame_index, f.path.name))
    return ordered[::every_n]


def collect_scene_frames(scene_dir: Path) -> list[FrameRecord]:
    """Collect all RGB frames in a scene with parsed or fallback timestamps."""
    files = list_rgb_files(scene_dir)
    records: list[FrameRecord] = []
    for idx, path in enumerate(files):
        ts, used_fallback = timestamp_or_fallback(path.name, idx)
        records.append(
            FrameRecord(
                path=path,
                timestamp=ts,
                frame_index=idx,
                original_filename=path.name,
                used_fallback_timestamp=used_fallback,
            )
        )
    return records


def sample_scene_frames(
    scene_dir: Path,
    strategy: str = "timestamp",
    interval_seconds: float = 1.0,
    every_n_frames: int = 30,
) -> tuple[list[FrameRecord], list[FrameRecord]]:
    """
    Return (all_frames, sampled_frames) for a scene directory.
    """
    all_frames = collect_scene_frames(scene_dir)
    if strategy == "timestamp":
        sampled = sample_by_timestamp(all_frames, interval_seconds=interval_seconds)
    elif strategy == "every_n_frames":
        sampled = sample_every_n_frames(all_frames, every_n=every_n_frames)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    logger.info(
        "Scene %s: %d RGB frames -> %d sampled (%s)",
        scene_dir.name,
        len(all_frames),
        len(sampled),
        strategy,
    )
    return all_frames, sampled
