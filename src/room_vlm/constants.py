"""Shared constants for room classification."""

from __future__ import annotations

from typing import Final

CANONICAL_LABELS: Final[tuple[str, ...]] = (
    "bedroom",
    "living room",
    "bathroom",
    "kitchen",
)

# NYU scene folder prefixes -> canonical label
CATEGORY_TO_LABEL: Final[dict[str, str]] = {
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "kitchen": "kitchen",
    "living_room": "living room",
}

LABEL_TO_CATEGORY: Final[dict[str, str]] = {
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "kitchen": "kitchen",
    "living room": "living_room",
}

INVALID_LABEL: Final[str] = "invalid"
UNKNOWN_LABEL: Final[str] = "unknown"
SOURCE_DATASET_NYU: Final[str] = "nyu_depth_v2"

DEFAULT_SEED: Final[int] = 42
IGNORE_INDEX: Final[int] = -100

# Official NYU Depth V2 raw archive base (never use unofficial mirrors by default).
NYU_OFFICIAL_BASE_URL: Final[str] = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2"
NYU_DATASET_PAGE: Final[str] = "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html"

NYU_TARGET_ARCHIVES: Final[tuple[str, ...]] = (
    "bathrooms_part1.zip",
    "bathrooms_part2.zip",
    "bathrooms_part3.zip",
    "bathrooms_part4.zip",
    "bedrooms_part1.zip",
    "bedrooms_part2.zip",
    "bedrooms_part3.zip",
    "bedrooms_part4.zip",
    "bedrooms_part5.zip",
    "bedrooms_part6.zip",
    "bedrooms_part7.zip",
    "kitchens_part1.zip",
    "kitchens_part2.zip",
    "kitchens_part3.zip",
    "living_rooms_part1.zip",
    "living_rooms_part2.zip",
    "living_rooms_part3.zip",
    "living_rooms_part4.zip",
)

RGB_EXTENSIONS: Final[tuple[str, ...]] = (".ppm", ".jpg", ".jpeg", ".png", ".bmp", ".webp")
