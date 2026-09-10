"""Held-out evaluation runner."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from PIL import Image

from room_vlm.constants import CANONICAL_LABELS
from room_vlm.data.dataset import read_jsonl
from room_vlm.evaluation.confusion import save_confusion_matrix
from room_vlm.evaluation.error_analysis import save_errors_csv, write_error_html_report
from room_vlm.evaluation.metrics import full_evaluation_metrics
from room_vlm.model.generation import generate_label
from room_vlm.model.scoring import score_image_labels

logger = logging.getLogger(__name__)


def evaluate_split(
    model: Any,
    processor: Any,
    jsonl_path: str | Path,
    output_dir: str | Path,
    *,
    strategy: str = "candidate_scoring",
    labels: list[str] | None = None,
    length_normalize: bool = True,
    device=None,
) -> dict[str, Any]:
    labels = labels or list(CANONICAL_LABELS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(jsonl_path)

    y_true: list[str] = []
    y_pred: list[str] = []
    score_dicts: list[dict[str, float]] = []
    ranked: list[list[str]] = []
    latencies: list[float] = []
    predictions: list[dict[str, Any]] = []

    for rec in records:
        with Image.open(rec["image"]) as img:
            image = img.convert("RGB")
        t0 = time.perf_counter()
        if strategy == "constrained_generation":
            gen = generate_label(model, processor, image, labels=labels, device=device)
            pred = gen["label"]
            scores = {lab: (1.0 if lab == pred else 0.0) for lab in labels}
            conf = scores.get(pred, 0.0)
        else:
            scored = score_image_labels(
                model,
                processor,
                image,
                labels=labels,
                length_normalize=length_normalize,
                device=device,
            )
            pred = scored["label"]
            scores = scored["scores"]
            conf = scored["confidence"]
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)
        y_true.append(rec["label"])
        y_pred.append(pred)
        score_dicts.append(scores)
        ranked.append([k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)])
        predictions.append(
            {
                "image_path": rec["image"],
                "scene_id": rec["scene_id"],
                "true_label": rec["label"],
                "predicted_label": pred,
                "confidence": conf,
                "scores": scores,
                "latency_ms": latency,
            }
        )

    metrics = full_evaluation_metrics(
        y_true,
        y_pred,
        score_dicts=score_dicts,
        ranked_labels=ranked,
        latencies_ms=latencies,
        labels=labels,
    )
    # Save artifacts
    import pandas as pd

    pd.DataFrame(predictions).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(metrics["confusion_matrix"], output_dir, labels=labels)
    save_errors_csv(predictions, output_dir / "errors.csv")
    write_error_html_report(predictions, output_dir / "errors.html")
    _write_report_template(output_dir / "report.md", metrics)
    return metrics


def _write_report_template(path: Path, metrics: dict[str, Any]) -> None:
    """Populate report sections with real metrics only (no fabricated numbers)."""
    path.write_text(
        "\n".join(
            [
                "# Room VLM Evaluation Report",
                "",
                "## Dataset",
                "_Fill with dataset stats from data/metadata/dataset_stats.json after prepare._",
                "",
                "## Model",
                "_Base: Qwen/Qwen3-VL-4B-Instruct (+ optional LoRA adapter)._",
                "",
                "## Training Configuration",
                "_See the run directory config.yaml._",
                "",
                "## Zero-Shot Baseline",
                "_See outputs/baseline/metrics.json when available._",
                "",
                "## Fine-Tuned Results",
                f"- accuracy: {metrics.get('accuracy')}",
                f"- balanced_accuracy: {metrics.get('balanced_accuracy')}",
                f"- macro_f1: {metrics.get('macro_f1')}",
                f"- invalid_rate: {metrics.get('invalid_rate')}",
                "",
                "## Per-Class Results",
                "```json",
                json.dumps(metrics.get("per_class", {}), indent=2),
                "```",
                "",
                "## Confusion Matrix",
                "See confusion_matrix.csv / confusion_matrix.png",
                "",
                "## Calibration",
                f"- nll: {metrics.get('nll')}",
                f"- brier: {metrics.get('brier')}",
                f"- ece: {metrics.get('ece')}",
                "",
                metrics.get("calibration_note", ""),
                "",
                "## Video Evaluation",
                "_Not populated unless a video eval was run._",
                "",
                "## Error Analysis",
                "See errors.csv and errors.html (hypotheses only).",
                "",
                "## Limitations",
                "See README limitations section.",
                "",
                "## Reproducibility",
                "seed=42 where configured; GPU bit-exact reproducibility is not guaranteed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def compare_baseline_and_finetuned(
    baseline_metrics_path: str | Path,
    finetuned_metrics_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write comparison.json only from real metric files."""
    baseline = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))
    finetuned = json.loads(Path(finetuned_metrics_path).read_text(encoding="utf-8"))

    def per_f1(m: dict, label: str) -> float | None:
        return (m.get("per_class") or {}).get(label, {}).get("f1")

    keys = [
        ("accuracy", "accuracy"),
        ("balanced_accuracy", "balanced_accuracy"),
        ("macro_f1", "macro_f1"),
        ("invalid_rate", "invalid_rate"),
        ("latency", "latency_ms_mean"),
    ]
    comparison: dict[str, Any] = {"baseline": {}, "fine_tuned": {}, "delta": {}}
    for name, key in keys:
        b = baseline.get(key)
        f = finetuned.get(key)
        comparison["baseline"][name] = b
        comparison["fine_tuned"][name] = f
        if isinstance(b, (int, float)) and isinstance(f, (int, float)):
            comparison["delta"][name] = f - b
    for lab in CANONICAL_LABELS:
        key = lab.replace(" ", "_") + "_f1"
        b = per_f1(baseline, lab)
        f = per_f1(finetuned, lab)
        comparison["baseline"][key] = b
        comparison["fine_tuned"][key] = f
        if isinstance(b, (int, float)) and isinstance(f, (int, float)):
            comparison["delta"][key] = f - b

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison
