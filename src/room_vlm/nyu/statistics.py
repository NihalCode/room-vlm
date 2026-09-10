"""Dataset statistics and reporting."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from room_vlm.constants import CANONICAL_LABELS

logger = logging.getLogger(__name__)


def compute_imbalance_ratio(counts: Mapping[str, int]) -> float:
    values = [c for c in counts.values() if c > 0]
    if not values:
        return 0.0
    return float(max(values) / max(min(values), 1))


def build_dataset_stats(
    *,
    scenes_per_class: Mapping[str, int],
    frames_per_class: Mapping[str, int],
    split_scenes: Mapping[str, Mapping[str, int]],
    split_frames: Mapping[str, Mapping[str, int]],
    total_raw_rgb: int,
    total_sampled: int,
    sampling: Mapping[str, Any],
    resolution_counts: Mapping[str, int],
    corrupt_images: int = 0,
    unreadable_images: int = 0,
    skipped_frames: int = 0,
    fallback_timestamps: int = 0,
) -> dict[str, Any]:
    return {
        "total_discovered_scenes": int(sum(scenes_per_class.values())),
        "total_raw_rgb_frames": int(total_raw_rgb),
        "total_sampled_frames": int(total_sampled),
        "scenes_per_class": dict(scenes_per_class),
        "sampled_frames_per_class": dict(frames_per_class),
        "split_scenes": {k: dict(v) for k, v in split_scenes.items()},
        "split_frames": {k: dict(v) for k, v in split_frames.items()},
        "image_resolution_distribution": dict(resolution_counts),
        "sampling": dict(sampling),
        "corrupt_images": int(corrupt_images),
        "unreadable_images": int(unreadable_images),
        "skipped_frames": int(skipped_frames),
        "fallback_timestamps": int(fallback_timestamps),
        "class_imbalance_ratio_frames": compute_imbalance_ratio(frames_per_class),
        "class_imbalance_ratio_scenes": compute_imbalance_ratio(scenes_per_class),
    }


def save_stats(stats: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def print_split_table(
    split_scenes: Mapping[str, Mapping[str, int]],
    split_frames: Mapping[str, Mapping[str, int]],
) -> None:
    splits = ["train", "validation", "test"]
    print("\nScenes by class/split:")
    header = f"{'':16} " + " ".join(f"{s:>12}" for s in splits)
    print(header)
    for label in CANONICAL_LABELS:
        row = f"{label:16} " + " ".join(
            f"{split_scenes.get(s, {}).get(label, 0):12d}" for s in splits
        )
        print(row)
    total_row = f"{'total':16} " + " ".join(
        f"{sum(split_scenes.get(s, {}).values()):12d}" for s in splits
    )
    print(total_row)

    print("\nSampled frames by class/split:")
    print(header)
    for label in CANONICAL_LABELS:
        row = f"{label:16} " + " ".join(
            f"{split_frames.get(s, {}).get(label, 0):12d}" for s in splits
        )
        print(row)
    total_row = f"{'total':16} " + " ".join(
        f"{sum(split_frames.get(s, {}).values()):12d}" for s in splits
    )
    print(total_row)


def plot_dataset_analysis(
    scenes_per_class: Mapping[str, int],
    frames_per_class: Mapping[str, int],
    split_frames: Mapping[str, Mapping[str, int]],
    output_dir: str | Path,
) -> None:
    """Save analysis plots. Uses matplotlib defaults (no manual colors)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping dataset plots")
        return

    labels = list(CANONICAL_LABELS)

    fig, ax = plt.subplots()
    ax.bar(labels, [frames_per_class.get(l, 0) for l in labels])
    ax.set_title("Sampled frames per class")
    ax.set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(output_dir / "frames_per_class.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(labels, [scenes_per_class.get(l, 0) for l in labels])
    ax.set_title("Scenes per class")
    ax.set_ylabel("scenes")
    fig.tight_layout()
    fig.savefig(output_dir / "scenes_per_class.png")
    plt.close(fig)

    splits = ["train", "validation", "test"]
    fig, ax = plt.subplots()
    x = range(len(labels))
    width = 0.25
    for i, split in enumerate(splits):
        vals = [split_frames.get(split, {}).get(l, 0) for l in labels]
        ax.bar([xi + i * width for xi in x], vals, width=width, label=split)
    ax.set_xticks([xi + width for xi in x], labels)
    ax.set_title("Split distribution (sampled frames)")
    ax.set_ylabel("frames")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "split_distribution.png")
    plt.close(fig)
