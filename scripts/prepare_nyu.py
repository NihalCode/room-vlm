#!/usr/bin/env python3
"""Prepare NYU Depth V2 RGB frames: sample, split, convert, write manifests."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

# Allow running without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room_vlm.config import load_config
from room_vlm.constants import CANONICAL_LABELS, SOURCE_DATASET_NYU
from room_vlm.data.dataset import ExampleRecord, write_jsonl
from room_vlm.nyu.discovery import discover_scenes
from room_vlm.nyu.sampling import sample_scene_frames
from room_vlm.nyu.splits import split_scenes
from room_vlm.nyu.statistics import (
    build_dataset_stats,
    plot_dataset_analysis,
    print_split_table,
    save_stats,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_nyu")


def convert_frame(src: Path, dest: Path, quality: int = 95) -> tuple[int, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        dest_suffix = dest.suffix.lower()
        if dest_suffix in {".jpg", ".jpeg"}:
            rgb.save(dest, format="JPEG", quality=quality)
        else:
            rgb.save(dest)
        return width, height


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="NYU raw root or fixture root")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root containing extracted/processed/metadata (default: from config)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_root = args.output
    if out_root is None:
        extracted_root = Path(cfg["dataset"]["extracted_root"])
        processed_root = Path(cfg["dataset"]["processed_root"])
        metadata_root = Path(cfg["dataset"]["metadata_root"])
    else:
        extracted_root = out_root / "extracted"
        processed_root = out_root / "processed"
        metadata_root = out_root / "metadata"

    extracted_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    scenes = discover_scenes(args.source)
    if not scenes:
        logger.error("No scenes discovered under %s", args.source)
        return 1

    split_cfg = cfg["split"]
    split_map = split_scenes(
        scenes,
        train=float(split_cfg.get("train", 0.7)),
        validation=float(split_cfg.get("validation", 0.15)),
        test=float(split_cfg.get("test", 0.15)),
        seed=int(split_cfg.get("seed", cfg.get("project", {}).get("seed", 42))),
    )

    sampling_cfg = cfg.get("sampling", {})
    strategy = sampling_cfg.get("strategy", "timestamp")
    interval = float(sampling_cfg.get("interval_seconds", 1.0))
    every_n = int(sampling_cfg.get("every_n_frames", 30))
    jpeg_quality = int(cfg["dataset"].get("jpeg_quality", 95))
    convert = bool(cfg["dataset"].get("convert_to_jpeg", True))

    all_frame_rows: list[dict] = []
    scene_rows: list[dict] = []
    split_records: dict[str, list[ExampleRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    scenes_per_class: dict[str, int] = defaultdict(int)
    frames_per_class: dict[str, int] = defaultdict(int)
    split_scenes_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    split_frames_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    resolution_counts: dict[str, int] = defaultdict(int)
    total_raw = 0
    total_sampled = 0
    corrupt = 0
    unreadable = 0
    skipped = 0
    fallback_ts = 0

    for scene in scenes:
        split = split_map[scene.scene_id]
        scenes_per_class[scene.label] += 1
        split_scenes_counts[split][scene.label] += 1

        all_frames, sampled = sample_scene_frames(
            scene.path,
            strategy=strategy,
            interval_seconds=interval,
            every_n_frames=every_n,
        )
        total_raw += len(all_frames)
        fallback_ts += sum(1 for f in all_frames if f.used_fallback_timestamp)

        scene_out = extracted_root / scene.scene_id
        scene_out.mkdir(parents=True, exist_ok=True)

        sampled_ok = 0
        for frame in sampled:
            try:
                if convert:
                    dest = scene_out / f"frame_{frame.frame_index:05d}.jpg"
                    width, height = convert_frame(frame.path, dest, quality=jpeg_quality)
                    image_path = dest
                else:
                    image_path = frame.path
                    with Image.open(image_path) as img:
                        width, height = img.size
            except Exception as exc:
                logger.warning("Unreadable/corrupt frame %s: %s", frame.path, exc)
                unreadable += 1
                corrupt += 1
                skipped += 1
                continue

            resolution_counts[f"{width}x{height}"] += 1
            sampled_ok += 1
            total_sampled += 1
            frames_per_class[scene.label] += 1
            split_frames_counts[split][scene.label] += 1

            row = {
                "image_path": str(image_path.as_posix()),
                "label": scene.label,
                "scene_id": scene.scene_id,
                "timestamp": frame.timestamp,
                "original_filename": frame.original_filename,
                "split": split,
                "source_dataset": SOURCE_DATASET_NYU,
                "frame_index": frame.frame_index,
                "width": width,
                "height": height,
            }
            all_frame_rows.append(row)
            split_key = "validation" if split == "val" else split
            split_records[split_key].append(
                ExampleRecord(
                    image=str(image_path),
                    label=scene.label,
                    scene_id=scene.scene_id,
                    source_dataset=SOURCE_DATASET_NYU,
                    split=split_key,
                    timestamp=frame.timestamp,
                    original_filename=frame.original_filename,
                    frame_index=frame.frame_index,
                    width=width,
                    height=height,
                )
            )

        scene_rows.append(
            {
                "scene_id": scene.scene_id,
                "label": scene.label,
                "split": split,
                "total_rgb_frames": len(all_frames),
                "sampled_frames": sampled_ok,
            }
        )

    # Write CSV manifests
    frames_csv = metadata_root / "all_frames.csv"
    with frames_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "label",
                "scene_id",
                "timestamp",
                "original_filename",
                "split",
                "source_dataset",
                "frame_index",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(all_frame_rows)

    scenes_csv = metadata_root / "scenes.csv"
    with scenes_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_id", "label", "split", "total_rgb_frames", "sampled_frames"],
        )
        writer.writeheader()
        writer.writerows(scene_rows)

    write_jsonl(split_records["train"], processed_root / "train.jsonl")
    write_jsonl(split_records["validation"], processed_root / "val.jsonl")
    write_jsonl(split_records["test"], processed_root / "test.jsonl")

    stats = build_dataset_stats(
        scenes_per_class=scenes_per_class,
        frames_per_class=frames_per_class,
        split_scenes=split_scenes_counts,
        split_frames=split_frames_counts,
        total_raw_rgb=total_raw,
        total_sampled=total_sampled,
        sampling={"strategy": strategy, "interval_seconds": interval, "every_n_frames": every_n},
        resolution_counts=resolution_counts,
        corrupt_images=corrupt,
        unreadable_images=unreadable,
        skipped_frames=skipped,
        fallback_timestamps=fallback_ts,
    )
    save_stats(stats, metadata_root / "dataset_stats.json")
    print_split_table(split_scenes_counts, split_frames_counts)

    analysis_dir = Path("outputs/dataset_analysis")
    if args.output is not None:
        analysis_dir = Path(args.output) / "dataset_analysis"
    plot_dataset_analysis(scenes_per_class, frames_per_class, split_frames_counts, analysis_dir)

    logger.info("Wrote manifests under %s and splits under %s", metadata_root, processed_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
