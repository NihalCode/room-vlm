"""Dataset records, JSONL IO, and PyTorch dataset wrapper."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image

from room_vlm.constants import CANONICAL_LABELS

logger = logging.getLogger(__name__)


@dataclass
class ExampleRecord:
    """Generic multimodal training/evaluation record (dataset-agnostic)."""

    image: str
    label: str
    scene_id: str
    source_dataset: str = "nyu_depth_v2"
    split: str | None = None
    timestamp: float | None = None
    original_filename: str | None = None
    frame_index: int | None = None
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def write_jsonl(records: Sequence[ExampleRecord | dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            payload = rec.to_dict() if isinstance(rec, ExampleRecord) else rec
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    for row in read_jsonl(path):
        yield row


class RoomImageDataset:
    """Simple map-style dataset yielding image path + label + scene_id."""

    def __init__(self, records: Sequence[dict[str, Any]], labels: Sequence[str] | None = None):
        self.records = list(records)
        self.labels = list(labels or CANONICAL_LABELS)
        for rec in self.records:
            if "image" not in rec or "label" not in rec or "scene_id" not in rec:
                raise ValueError(f"Record missing required keys: {rec}")
            if rec["label"] not in self.labels:
                raise ValueError(f"Unknown label {rec['label']!r}")

    @classmethod
    def from_jsonl(cls, path: str | Path, labels: Sequence[str] | None = None) -> "RoomImageDataset":
        return cls(read_jsonl(path), labels=labels)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        image_path = Path(rec["image"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return {
            "image": image,
            "image_path": str(image_path),
            "label": rec["label"],
            "scene_id": rec["scene_id"],
            "source_dataset": rec.get("source_dataset", "unknown"),
        }
