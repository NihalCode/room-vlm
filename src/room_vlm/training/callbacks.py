"""Training callbacks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from transformers import EarlyStoppingCallback, TrainerCallback

logger = logging.getLogger(__name__)


class MetricsLoggerCallback(TrainerCallback):
    """Append epoch metrics to a CSV and JSON sidecars under the run directory."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "training_log.csv"
        if not self.log_path.exists():
            self.log_path.write_text(
                "step,epoch,loss,eval_loss,eval_accuracy,eval_macro_f1\n",
                encoding="utf-8",
            )

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        row = {
            "step": state.global_step,
            "epoch": state.epoch,
            "loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "eval_accuracy": logs.get("eval_accuracy"),
            "eval_macro_f1": logs.get("eval_macro_f1"),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{row['step']},{row['epoch']},{row['loss']},{row['eval_loss']},"
                f"{row['eval_accuracy']},{row['eval_macro_f1']}\n"
            )


def build_early_stopping(cfg: dict[str, Any]) -> EarlyStoppingCallback | None:
    if not cfg.get("enabled", True):
        return None
    patience = int(cfg.get("patience", 2))
    return EarlyStoppingCallback(early_stopping_patience=patience)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
