"""Candidate label likelihood scoring for constrained multi-class VLM evaluation."""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from room_vlm.constants import CANONICAL_LABELS
from room_vlm.prompts import USER_PROMPT, build_conversation

logger = logging.getLogger(__name__)


def softmax_dict(scores: dict[str, float]) -> dict[str, float]:
    """Numerically stable softmax over a label->score mapping."""
    if not scores:
        return {}
    keys = list(scores.keys())
    vals = torch.tensor([scores[k] for k in keys], dtype=torch.float64)
    probs = torch.softmax(vals, dim=0)
    return {k: float(p) for k, p in zip(keys, probs.tolist())}


def score_labels_from_token_logprobs(
    label_token_logprobs: dict[str, list[float]],
    *,
    length_normalize: bool = True,
) -> dict[str, Any]:
    """
    Aggregate per-token log-probs into class scores.

    score(label) = mean(token log-probs) if length_normalize else sum(token log-probs)

    Softmax over scores yields relative candidate likelihoods — not calibrated
    real-world probabilities.
    """
    raw_scores: dict[str, float] = {}
    for label, logprobs in label_token_logprobs.items():
        if not logprobs:
            raw_scores[label] = float("-inf")
            continue
        total = float(sum(logprobs))
        raw_scores[label] = total / len(logprobs) if length_normalize else total
    probs = softmax_dict(raw_scores)
    best = max(probs, key=probs.get) if probs else "invalid"
    return {
        "label": best,
        "confidence": float(probs.get(best, 0.0)),
        "scores": probs,
        "raw_scores": raw_scores,
    }


@torch.inference_mode()
def score_image_labels(
    model: Any,
    processor: Any,
    image: Any,
    labels: Sequence[str] = CANONICAL_LABELS,
    *,
    length_normalize: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Evaluate conditional log-likelihood of each candidate label given image+prompt.

    For multi-token labels such as ``living room``, token log-probs are summed
    (or length-normalized) consistently across classes before softmax.
    """
    device = device or next(model.parameters()).device
    label_token_logprobs: dict[str, list[float]] = {}

    for label in labels:
        messages = build_conversation(image, label=label)
        messages[0]["content"][0] = {"type": "image", "image": image}
        # Prompt-only length for slicing answer tokens
        prompt_messages = build_conversation(image, label=None)
        prompt_messages[0]["content"][0] = {"type": "image", "image": image}

        prompt_inputs = processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        full_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in full_inputs.items()}
        outputs = model(**full_inputs)
        logits = outputs.logits  # [1, seq, vocab]
        # Teacher-forced next-token logprobs for answer tokens
        answer_ids = full_inputs["input_ids"][0, prompt_len:]
        if answer_ids.numel() == 0:
            label_token_logprobs[label] = []
            continue
        # Logits at position i predict token i+1; for answer starting at prompt_len,
        # use logits[:, prompt_len-1 : -1] aligned with answer_ids.
        shift_logits = logits[0, prompt_len - 1 : -1, :]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = []
        for i, token_id in enumerate(answer_ids.tolist()):
            if i >= log_probs.shape[0]:
                break
            token_lp.append(float(log_probs[i, token_id].item()))
        label_token_logprobs[label] = token_lp

    result = score_labels_from_token_logprobs(
        label_token_logprobs, length_normalize=length_normalize
    )
    result["prompt"] = USER_PROMPT
    return result
