"""PEFT LoRA configuration with runtime module inspection."""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Candidate projection names commonly used in decoder-only LMs.
CANDIDATE_LINEAR_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

VISION_HINTS = ("visual", "vision", "vit", "vision_tower", "vision_model")
PROJECTOR_HINTS = ("merger", "projector", "multi_modal", "mm_projector", "visual_projector")


def _module_is_vision(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in VISION_HINTS)


def _module_is_projector(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in PROJECTOR_HINTS)


def discover_lora_target_modules(
    model: Any,
    *,
    train_vision: bool = False,
    train_projector: bool = False,
) -> list[str]:
    """
    Inspect ``model.named_modules()`` and return LoRA target module names.

    Defaults to language-model attention/MLP projections only.
    Does not guess: only names that actually exist are returned.
    """
    leaf_names: set[str] = set()
    full_names: list[str] = []
    for name, module in model.named_modules():
        short = name.split(".")[-1]
        if short in CANDIDATE_LINEAR_NAMES:
            if _module_is_vision(name) and not train_vision:
                continue
            if _module_is_projector(name) and not train_projector:
                # projector modules rarely use q_proj names; keep filter anyway
                pass
            leaf_names.add(short)
            full_names.append(name)

    if train_projector:
        for name, module in model.named_modules():
            if _module_is_projector(name) and hasattr(module, "weight"):
                # Prefer exact leaf names for PEFT when they are Linear-like.
                short = name.split(".")[-1]
                leaf_names.add(short)
                full_names.append(name)

    targets = sorted(leaf_names)
    if not targets:
        raise RuntimeError(
            "No LoRA target modules discovered. Inspect model.named_modules() "
            "and update discover_lora_target_modules."
        )
    logger.info("LoRA target modules (%d leaf names): %s", len(targets), targets)
    logger.info("Example matched module paths (up to 20): %s", full_names[:20])
    return targets


def freeze_vision_and_base(model: Any, *, train_vision: bool = False, train_projector: bool = False) -> None:
    """Ensure base weights stay frozen; LoRA adapters remain trainable after PEFT wrap."""
    for name, param in model.named_parameters():
        if _module_is_vision(name) and not train_vision:
            param.requires_grad = False
        if _module_is_projector(name) and not train_projector:
            param.requires_grad = False


def apply_lora(
    model: Any,
    lora_cfg: dict[str, Any],
    *,
    train_vision: bool = False,
    train_projector: bool = False,
) -> Any:
    """Attach PEFT LoRA adapters using discovered target modules."""
    from peft import LoraConfig, get_peft_model

    if not lora_cfg.get("enabled", True):
        logger.warning("LoRA disabled in config — this will train full weights if unfrozen.")
        return model

    targets = discover_lora_target_modules(
        model, train_vision=train_vision, train_projector=train_projector
    )
    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=targets,
    )
    peft_model = get_peft_model(model, config)
    freeze_vision_and_base(peft_model, train_vision=train_vision, train_projector=train_projector)
    try:
        trainable, total = peft_model.get_nb_trainable_parameters()
        logger.info(
            "Trainable params: %s / %s (%.4f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / max(total, 1),
        )
    except Exception:
        logger.info("Trainable parameter counts unavailable via get_nb_trainable_parameters()")
    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass
    return peft_model
