"""Constrained generation helpers."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from room_vlm.constants import CANONICAL_LABELS
from room_vlm.data.collator import normalize_predicted_label
from room_vlm.prompts import build_conversation


@torch.inference_mode()
def generate_label(
    model: Any,
    processor: Any,
    image: Any,
    labels: Sequence[str] = CANONICAL_LABELS,
    *,
    max_new_tokens: int = 8,
    do_sample: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Generate a short answer and normalize to a canonical label or invalid."""
    device = device or next(model.parameters()).device
    messages = build_conversation(image, label=None)
    messages[0]["content"][0] = {"type": "image", "image": image}
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    trimmed = output_ids[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    label = normalize_predicted_label(text, labels, strict=True)
    return {"raw_text": text, "label": label, "invalid": label == "invalid"}
