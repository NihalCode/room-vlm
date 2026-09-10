"""Single-image inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from room_vlm.constants import CANONICAL_LABELS, UNKNOWN_LABEL
from room_vlm.model.generation import generate_label
from room_vlm.model.scoring import score_image_labels


def predict_image(
    model: Any,
    processor: Any,
    image: str | Path | Image.Image,
    *,
    strategy: str = "candidate_scoring",
    labels: list[str] | None = None,
    length_normalize: bool = True,
    abstention_enabled: bool = False,
    abstention_threshold: float | None = None,
    device=None,
) -> dict[str, Any]:
    labels = labels or list(CANONICAL_LABELS)
    if not isinstance(image, Image.Image):
        with Image.open(image) as img:
            pil = img.convert("RGB")
    else:
        pil = image.convert("RGB")

    if strategy == "constrained_generation":
        gen = generate_label(model, processor, pil, labels=labels, device=device)
        scores = {lab: (1.0 if lab == gen["label"] else 0.0) for lab in labels}
        result = {
            "label": gen["label"],
            "confidence": float(scores.get(gen["label"], 0.0)),
            "scores": scores,
            "raw_text": gen.get("raw_text"),
            "strategy": strategy,
        }
    else:
        scored = score_image_labels(
            model,
            processor,
            pil,
            labels=labels,
            length_normalize=length_normalize,
            device=device,
        )
        result = {
            "label": scored["label"],
            "confidence": scored["confidence"],
            "scores": scored["scores"],
            "strategy": strategy,
        }

    if abstention_enabled and abstention_threshold is not None:
        if result["confidence"] < float(abstention_threshold):
            result["label"] = UNKNOWN_LABEL
            result["abstained"] = True
        else:
            result["abstained"] = False
    else:
        result["abstained"] = False
    return result
