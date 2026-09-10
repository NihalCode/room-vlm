"""Centralized prompts for room classification."""

from __future__ import annotations

from room_vlm.constants import CANONICAL_LABELS

USER_PROMPT = (
    "Classify the indoor room shown in the image.\n\n"
    "Choose exactly one label from:\n"
    "bedroom\n"
    "living room\n"
    "bathroom\n"
    "kitchen\n\n"
    "Respond with only the label."
)


def build_user_message(image_ref: str | None = None) -> dict:
    """Build a multimodal chat user message for the processor."""
    content: list[dict] = []
    if image_ref is not None:
        content.append({"type": "image", "image": image_ref})
    else:
        content.append({"type": "image"})
    content.append({"type": "text", "text": USER_PROMPT})
    return {"role": "user", "content": content}


def build_assistant_message(label: str) -> dict:
    """Build the assistant training target message."""
    return {"role": "assistant", "content": [{"type": "text", "text": label}]}


def build_conversation(image_ref: str, label: str | None = None) -> list[dict]:
    """Build a full chat conversation optionally including the assistant answer."""
    messages = [build_user_message(image_ref)]
    if label is not None:
        messages.append(build_assistant_message(label))
    return messages


def label_choices() -> tuple[str, ...]:
    return CANONICAL_LABELS
