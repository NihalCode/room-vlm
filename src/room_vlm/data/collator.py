"""Multimodal collator with assistant-only loss masking."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from room_vlm.constants import IGNORE_INDEX
from room_vlm.prompts import USER_PROMPT, build_conversation

logger = logging.getLogger(__name__)


def normalize_predicted_label(text: str, labels: Sequence[str], strict: bool = True) -> str:
    """Normalize model output to a canonical label or 'invalid'."""
    cleaned = (text or "").strip().lower()
    cleaned = cleaned.strip("\"'` \n\t.")
    cleaned = cleaned.replace("_", " ")
    # Take first line only
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    if cleaned in labels:
        return cleaned
    if not strict:
        for label in labels:
            if label in cleaned:
                return label
    return "invalid"


def build_masked_labels(
    prompt_len: int,
    full_input_ids: Sequence[int],
    pad_token_id: int | None = None,
) -> list[int]:
    """
    Create causal LM labels where prompt tokens (and padding) are IGNORE_INDEX.

    Only positions >= prompt_len corresponding to the assistant answer are trained.
    """
    labels = [IGNORE_INDEX] * len(full_input_ids)
    for i in range(prompt_len, len(full_input_ids)):
        token_id = int(full_input_ids[i])
        if pad_token_id is not None and token_id == pad_token_id:
            labels[i] = IGNORE_INDEX
        else:
            labels[i] = token_id
    return labels


class RoomVLMCollator:
    """
    Collate image+label examples into processor tensors with loss masking.

    Loss is computed only on assistant answer tokens.
    """

    def __init__(self, processor: Any, labels: Sequence[str]):
        self.processor = processor
        self.labels = list(labels)

    def _tokenize_messages(self, messages: list[dict], add_generation_prompt: bool) -> dict[str, Any]:
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Process examples one-by-one then pad; keeps masking logic clear.
        import torch

        all_input_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_attention: list[torch.Tensor] = []
        extra_keys: dict[str, list] = {}

        pad_token_id = getattr(self.processor, "tokenizer", self.processor)
        pad_id = getattr(pad_token_id, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.processor, "pad_token_id", 0)

        for item in batch:
            image = item["image"]
            label = item["label"]
            # Prompt-only (for masking length)
            prompt_messages = build_conversation(image, label=None)
            # Some processors need PIL in content; replace image ref with PIL.
            prompt_messages[0]["content"][0] = {"type": "image", "image": image}
            full_messages = build_conversation(image, label=label)
            full_messages[0]["content"][0] = {"type": "image", "image": image}

            prompt_inputs = self._tokenize_messages(prompt_messages, add_generation_prompt=True)
            full_inputs = self._tokenize_messages(full_messages, add_generation_prompt=False)

            prompt_ids = prompt_inputs["input_ids"][0]
            full_ids = full_inputs["input_ids"][0]
            prompt_len = int(prompt_ids.shape[0])
            # Safety: if full is not longer, still mask all but last few tokens of answer length.
            labels = build_masked_labels(prompt_len, full_ids.tolist(), pad_token_id=pad_id)

            all_input_ids.append(full_ids)
            all_labels.append(torch.tensor(labels, dtype=torch.long))
            attn = full_inputs.get("attention_mask")
            if attn is None:
                attn = torch.ones_like(full_ids)
            else:
                attn = attn[0]
            all_attention.append(attn)

            for key, value in full_inputs.items():
                if key in {"input_ids", "attention_mask", "labels"}:
                    continue
                extra_keys.setdefault(key, []).append(value[0] if hasattr(value, "__getitem__") else value)

        # Pad sequences
        max_len = max(x.shape[0] for x in all_input_ids)
        def pad_stack(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
            out = []
            for t in tensors:
                if t.shape[0] < max_len:
                    pad = torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype)
                    t = torch.cat([t, pad], dim=0)
                out.append(t)
            return torch.stack(out, dim=0)

        batch_out: dict[str, Any] = {
            "input_ids": pad_stack(all_input_ids, int(pad_id)),
            "attention_mask": pad_stack(all_attention, 0),
            "labels": pad_stack(all_labels, IGNORE_INDEX),
        }
        # Pass through pixel values etc. when batch size is 1-friendly; for >1, stack if possible.
        for key, values in extra_keys.items():
            try:
                batch_out[key] = torch.stack(values, dim=0)
            except Exception:
                # Leave unbatched extras out rather than failing training setup in tests.
                logger.debug("Could not stack collator key %s", key)
        # Keep reference prompt text for debugging (not used by model).
        batch_out["prompt_text"] = USER_PROMPT
        return batch_out
