"""Model loading for Qwen3-VL (Transformers current API)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    device: torch.device
    quantization_enabled: bool
    model_name: str


def resolve_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def load_processor(model_name: str) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    return processor


def _build_quantization_config(cfg: dict[str, Any]) -> Any | None:
    if not cfg.get("enabled", False):
        return None
    if not torch.cuda.is_available():
        raise RuntimeError(
            "quantization.enabled=true requires CUDA. "
            "Set quantization.enabled=false and pass --allow-bf16-lora for BF16 LoRA, "
            "or run on a CUDA GPU with bitsandbytes."
        )
    if not bitsandbytes_available():
        raise RuntimeError(
            "bitsandbytes is not available. Install bitsandbytes for QLoRA, "
            "or disable quantization and use BF16 LoRA explicitly (--allow-bf16-lora)."
        )
    from transformers import BitsAndBytesConfig

    compute_dtype_name = str(cfg.get("compute_dtype", "bfloat16")).lower()
    compute_dtype = torch.bfloat16 if compute_dtype_name in {"bfloat16", "bf16"} else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=int(cfg.get("bits", 4)) == 4,
        bnb_4bit_quant_type=str(cfg.get("type", "nf4")),
        bnb_4bit_use_double_quant=bool(cfg.get("double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_processor(
    model_name: str,
    *,
    quantization: dict[str, Any] | None = None,
    device: str | None = "auto",
    torch_dtype: str | None = "bfloat16",
    attn_implementation: str | None = None,
    allow_bf16_lora: bool = False,
) -> LoadedModel:
    """
    Load Qwen3-VL using the current Transformers API.

    Prefer ``Qwen3VLForConditionalGeneration``; fall back to
    ``AutoModelForImageTextToText`` if needed.
    """
    quantization = quantization or {"enabled": False}
    quant_cfg = None
    try:
        quant_cfg = _build_quantization_config(quantization)
    except RuntimeError as exc:
        if quantization.get("enabled") and allow_bf16_lora:
            logger.warning("%s Falling back to BF16 LoRA because --allow-bf16-lora was set.", exc)
            quant_cfg = None
        elif quantization.get("enabled"):
            raise

    processor = load_processor(model_name)

    dtype = None
    if quant_cfg is None:
        if torch_dtype in {"bfloat16", "bf16"} and (torch.cuda.is_available() or True):
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype in {"float16", "fp16"}:
            dtype = torch.float16
        else:
            dtype = torch.float32

    model_kwargs: dict[str, Any] = {
        "device_map": "auto" if device == "auto" and torch.cuda.is_available() else None,
    }
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
    if dtype is not None and quant_cfg is None:
        model_kwargs["torch_dtype"] = dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = None
    try:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        logger.info("Loaded model via Qwen3VLForConditionalGeneration")
    except Exception as exc:
        logger.warning(
            "Qwen3VLForConditionalGeneration failed (%s); trying AutoModelForImageTextToText",
            exc,
        )
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)

    resolved = resolve_device(device)
    if model_kwargs.get("device_map") is None:
        model.to(resolved)

    if resolved.type == "cpu":
        logger.warning("CUDA unavailable: CPU inference/training will be very slow.")

    return LoadedModel(
        model=model,
        processor=processor,
        device=resolved,
        quantization_enabled=quant_cfg is not None,
        model_name=model_name,
    )


def attach_adapter(model: Any, adapter_path: str) -> Any:
    from peft import PeftModel

    logger.info("Loading LoRA adapter from %s", adapter_path)
    return PeftModel.from_pretrained(model, adapter_path)


def print_oom_remediation() -> None:
    print(
        "\nCUDA out-of-memory remediation suggestions:\n"
        "  - Reduce per_device_train_batch_size to 1\n"
        "  - Increase gradient_accumulation_steps\n"
        "  - Enable gradient_checkpointing\n"
        "  - Keep quantization.enabled=true (4-bit QLoRA)\n"
        "  - Lower image/token limits if your processor supports them\n"
        "  - Use a GPU with more VRAM (24GB+ recommended for comfort)\n"
        "These are estimated remedies; exact VRAM needs were not measured in this repository.\n"
    )
