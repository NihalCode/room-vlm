"""Confusion matrix export."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from room_vlm.constants import CANONICAL_LABELS


def save_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    output_dir: str | Path,
    labels: Sequence[str] = CANONICAL_LABELS,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(labels)
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(output_dir / "confusion_matrix.csv")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.asarray(matrix), interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)
