"""Error analysis exports and HTML contact sheet."""

from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from room_vlm.constants import CANONICAL_LABELS

# Hypotheses only — do not claim causal certainty.
ERROR_HYPOTHESES = {
    ("bedroom", "living room"): "Hypothesis: open-plan layout or limited bed visibility.",
    ("living room", "bedroom"): "Hypothesis: soft seating/bedding cues or unusual layout.",
    ("bathroom", "kitchen"): "Hypothesis: tile/fixture confusion or poor lighting.",
    ("kitchen", "living room"): "Hypothesis: partial room visibility or open-plan kitchen.",
}


def build_error_rows(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for p in predictions:
        if p.get("true_label") == p.get("predicted_label"):
            continue
        scores = p.get("scores") or {}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        second = ranked[1] if len(ranked) > 1 else ("", 0.0)
        rows.append(
            {
                "image_path": p.get("image_path"),
                "scene_id": p.get("scene_id"),
                "true_label": p.get("true_label"),
                "predicted_label": p.get("predicted_label"),
                "confidence": p.get("confidence"),
                "second_choice": second[0],
                "second_choice_confidence": second[1],
            }
        )
    return rows


def save_errors_csv(predictions: Sequence[dict[str, Any]], path: str | Path) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(build_error_rows(predictions))
    df.to_csv(path, index=False)
    return df


def write_error_html_report(
    predictions: Sequence[dict[str, Any]],
    path: str | Path,
    max_examples: int = 64,
) -> None:
    """Create a simple HTML contact sheet of misclassified images."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = build_error_rows(predictions)[:max_examples]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in errors:
        grouped[(row["true_label"], row["predicted_label"])].append(row)

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Room VLM Error Analysis</title>",
        "<style>body{font-family:sans-serif} .card{display:inline-block;margin:8px;vertical-align:top}",
        "img{max-width:200px;max-height:150px} .hyp{color:#444;font-size:0.9em}</style></head><body>",
        "<h1>Misclassification report</h1>",
        "<p>Hypotheses below are speculative and not verified causes.</p>",
    ]
    for (true_l, pred_l), rows in sorted(grouped.items()):
        hyp = ERROR_HYPOTHESES.get((true_l, pred_l), "Hypothesis: partial visibility, blur, or unusual layout.")
        parts.append(f"<h2>{html.escape(true_l)} → {html.escape(pred_l)} ({len(rows)})</h2>")
        parts.append(f"<p class='hyp'>{html.escape(hyp)}</p>")
        for row in rows:
            img = html.escape(str(row.get("image_path") or ""))
            parts.append("<div class='card'>")
            if img:
                parts.append(f"<img src='{img}' alt='misclassified'/>")
            parts.append(
                f"<div>scene={html.escape(str(row.get('scene_id')))}<br/>"
                f"pred={html.escape(str(row.get('predicted_label')))} "
                f"({row.get('confidence')})<br/>"
                f"2nd={html.escape(str(row.get('second_choice')))}</div></div>"
            )
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
