"""NYU Depth V2 scene discovery and label mapping."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from room_vlm.constants import CATEGORY_TO_LABEL, RGB_EXTENSIONS, SOURCE_DATASET_NYU

logger = logging.getLogger(__name__)

SCENE_DIR_RE = re.compile(
    r"^(?P<category>bedroom|bathroom|kitchen|living_room)_(?P<id>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class SceneInfo:
    scene_id: str
    label: str
    category: str
    path: Path
    source_dataset: str = SOURCE_DATASET_NYU


def normalize_label(raw: str) -> str:
    """Map folder category or free-text variants to a canonical label."""
    text = raw.strip().lower().replace("-", " ").replace("__", "_")
    text = re.sub(r"\s+", " ", text)
    if text in CATEGORY_TO_LABEL:
        return CATEGORY_TO_LABEL[text]
    if text.replace(" ", "_") in CATEGORY_TO_LABEL:
        return CATEGORY_TO_LABEL[text.replace(" ", "_")]
    # Already canonical?
    if text in {"bedroom", "living room", "bathroom", "kitchen"}:
        return text
    raise ValueError(f"Unknown room category/label: {raw!r}")


def parse_scene_dirname(name: str) -> tuple[str, str] | None:
    """Return (category, scene_id) if dirname matches a supported NYU scene."""
    match = SCENE_DIR_RE.match(name.strip())
    if not match:
        return None
    category = match.group("category").lower()
    scene_id = f"{category}_{match.group('id')}"
    return category, scene_id


def label_from_scene_id(scene_id: str) -> str:
    category, _ = parse_scene_dirname(scene_id) or (None, None)
    if category is None:
        # Try prefix before last underscore group
        for cat in CATEGORY_TO_LABEL:
            if scene_id.startswith(cat + "_"):
                return CATEGORY_TO_LABEL[cat]
        raise ValueError(f"Cannot derive label from scene_id={scene_id!r}")
    return CATEGORY_TO_LABEL[category]


def is_rgb_file(path: Path) -> bool:
    name = path.name.lower()
    if not path.is_file():
        return False
    if name.startswith("r-") and path.suffix.lower() in {".ppm", ".jpg", ".jpeg", ".png"}:
        return True
    # Fixtures / converted images without NYU prefix
    if path.suffix.lower() in RGB_EXTENSIONS and not name.startswith(("d-", "a-")):
        # Prefer files that look like frames
        return True
    return False


def list_rgb_files(scene_dir: Path) -> list[Path]:
    """List RGB frame files in a scene directory (deterministic order)."""
    files = [p for p in scene_dir.iterdir() if is_rgb_file(p)]
    # Prefer NYU r-* naming when mixed; still include fixtures.
    files.sort(key=lambda p: p.name)
    return files


def discover_scenes(root: str | Path) -> list[SceneInfo]:
    """
    Discover relevant NYU scene folders under root.

    Supports either:
      root/bedroom_0001/...
    or nested layouts containing those folder names.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    scenes: dict[str, SceneInfo] = {}
    # Walk deterministically
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        parsed = parse_scene_dirname(path.name)
        if parsed is None:
            continue
        category, scene_id = parsed
        label = CATEGORY_TO_LABEL[category]
        if scene_id in scenes:
            logger.warning("Duplicate scene_id %s at %s (keeping first)", scene_id, path)
            continue
        scenes[scene_id] = SceneInfo(
            scene_id=scene_id,
            label=label,
            category=category,
            path=path.resolve(),
        )

    result = sorted(scenes.values(), key=lambda s: s.scene_id)
    logger.info("Discovered %d scenes under %s", len(result), root)
    return result
