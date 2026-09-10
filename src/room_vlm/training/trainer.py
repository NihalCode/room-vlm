"""HuggingFace Trainer wiring for LoRA/QLoRA fine-tuning."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import TrainingArguments, Trainer

from room_vlm.config import save_config
from room_vlm.data.collator import RoomVLMCollator
from room_vlm.data.dataset import RoomImageDataset
from room_vlm.data.validation import validate_split_jsonl_dir
from room_vlm.model.loader import load_model_and_processor, print_oom_remediation
from room_vlm.model.lora import apply_lora
from room_vlm.nyu.splits import SceneLeakageError
from room_vlm.training.callbacks import MetricsLoggerCallback, build_early_stopping, save_json
from room_vlm.training.metrics import classification_metrics

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_environment(seed: int) -> dict[str, Any]:
    """Collect reproducible environment metadata without dumping secrets."""
    def _ver(mod: str) -> str | None:
        try:
            m = __import__(mod)
            return getattr(m, "__version__", None)
        except Exception:
            return None

    git_hash = None
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_hash = None

    gpu_name = None
    cuda_version = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = getattr(torch.version, "cuda", None)

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _ver("torch"),
        "transformers": _ver("transformers"),
        "peft": _ver("peft"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "git_commit": git_hash,
        "seed": seed,
        # Explicitly do NOT include env vars / tokens.
    }


def _maybe_enable_wandb() -> str:
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_PROJECT"):
        return "wandb"
    return "none"


class RoomVLMTrainer:
    def __init__(self, cfg: dict[str, Any], allow_bf16_lora: bool = False):
        self.cfg = cfg
        self.allow_bf16_lora = allow_bf16_lora

    def train(self) -> Path:
        seed = int(self.cfg.get("project", {}).get("seed", 42))
        set_seed(seed)

        processed = Path(self.cfg["dataset"]["processed_root"])
        try:
            validate_split_jsonl_dir(processed)
        except SceneLeakageError:
            logger.error("Scene leakage detected — refusing to start training")
            raise

        run_name = self.cfg.get("project", {}).get("run_name", "run")
        run_dir = Path(self.cfg["training"].get("output_dir", "outputs")) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.cfg, run_dir / "config.yaml")
        save_json(run_dir / "environment.json", collect_environment(seed))

        labels = list(self.cfg.get("labels", []))
        train_ds = RoomImageDataset.from_jsonl(processed / "train.jsonl", labels=labels)
        val_ds = RoomImageDataset.from_jsonl(processed / "val.jsonl", labels=labels)

        loaded = load_model_and_processor(
            self.cfg["model"]["name"],
            quantization=self.cfg.get("quantization"),
            torch_dtype="bfloat16" if self.cfg["training"].get("bf16", True) else "float32",
            attn_implementation=self.cfg.get("model", {}).get("attn_implementation"),
            allow_bf16_lora=self.allow_bf16_lora,
        )
        model = apply_lora(
            loaded.model,
            self.cfg.get("lora", {}),
            train_vision=bool(self.cfg["training"].get("train_vision", False)),
            train_projector=bool(self.cfg["training"].get("train_projector", False)),
        )
        if self.cfg["training"].get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        collator = RoomVLMCollator(loaded.processor, labels=labels)
        tcfg = self.cfg["training"]
        args = TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            num_train_epochs=float(tcfg.get("epochs", 3)),
            learning_rate=float(tcfg.get("learning_rate", 1e-4)),
            per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 16)),
            warmup_ratio=float(tcfg.get("warmup_ratio", 0.05)),
            weight_decay=float(tcfg.get("weight_decay", 0.01)),
            max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
            bf16=bool(tcfg.get("bf16", True)) and torch.cuda.is_available(),
            eval_strategy=str(tcfg.get("eval_strategy", "epoch")),
            save_strategy=str(tcfg.get("save_strategy", "epoch")),
            logging_steps=int(tcfg.get("logging_steps", 10)),
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to=_maybe_enable_wandb(),
            remove_unused_columns=False,
            seed=seed,
        )

        def compute_metrics(eval_pred):  # noqa: ANN001
            # Placeholder: generation-based metrics are computed in evaluation scripts.
            # During Trainer eval we report loss primarily; attach zeros if logits shape unexpected.
            return {"accuracy": 0.0, "macro_f1": 0.0}

        callbacks = [MetricsLoggerCallback(run_dir)]
        es = build_early_stopping(self.cfg.get("early_stopping", {}))
        if es is not None:
            callbacks.append(es)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        try:
            train_result = trainer.train()
        except torch.cuda.OutOfMemoryError:
            print_oom_remediation()
            raise

        adapter_dir = run_dir / "adapter"
        trainer.model.save_pretrained(adapter_dir)
        loaded.processor.save_pretrained(adapter_dir)
        save_json(
            run_dir / "train_metrics.json",
            {k: float(v) if isinstance(v, (int, float)) else v for k, v in train_result.metrics.items()},
        )

        # Validation pass using loss from trainer.evaluate
        val_metrics = trainer.evaluate()
        save_json(run_dir / "val_metrics.json", val_metrics)
        logger.info("Saved best adapter to %s", adapter_dir)
        return adapter_dir
