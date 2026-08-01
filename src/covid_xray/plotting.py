"""Matplotlib figures for evaluation artefacts and training curves.

The Agg backend is selected explicitly so figures render identically on a
headless training box and on a laptop.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from covid_xray.history import TrainingHistory

LOGGER: Final = logging.getLogger(__name__)

MisclassifiedSample = tuple[Path, int, int]


def _save(figure: plt.Figure, path: Path) -> None:
    """Write a figure to ``path`` and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    LOGGER.info("Wrote %s", path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    classes: Sequence[str],
    path: Path,
    *,
    normalize: bool = False,
) -> None:
    """Render a confusion matrix with per-cell counts.

    Args:
        matrix: A ``(n_classes, n_classes)`` array of counts, true labels on
            rows and predictions on columns.
        classes: Class names in label-index order.
        path: Destination PNG path.
        normalize: When ``True``, colour cells by per-row proportion so that
            rare classes remain visible next to the majority class.
    """
    counts = matrix.astype(float)
    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        shaded = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)
    else:
        shaded = counts

    figure, axes = plt.subplots(figsize=(6, 6))
    image = axes.imshow(shaded, cmap="Blues", vmin=0.0)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    axes.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=list(classes),
        yticklabels=list(classes),
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix" + (" (row-normalised)" if normalize else ""),
    )
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right")

    threshold = shaded.max() / 2.0 if shaded.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                color="white" if shaded[row, column] > threshold else "black",
            )
    _save(figure, path)


def plot_misclassified(
    samples: Sequence[MisclassifiedSample],
    classes: Sequence[str],
    path: Path,
    *,
    max_examples: int = 9,
) -> bool:
    """Render a grid of misclassified images.

    Args:
        samples: ``(image path, true label, predicted label)`` triples.
        classes: Class names in label-index order.
        path: Destination PNG path.
        max_examples: Upper bound on the number of images shown.

    Returns:
        ``True`` if a figure was written, ``False`` if there was nothing to plot.
    """
    selected = list(samples[:max_examples])
    if not selected:
        LOGGER.info("No misclassified samples; skipping %s", path.name)
        return False

    columns = min(3, len(selected))
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.2 * rows), squeeze=False)
    flat = axes.flatten()

    for axis, (image_path, true_label, predicted_label) in zip(flat, selected, strict=False):
        with Image.open(image_path) as handle:
            axis.imshow(handle.convert("RGB"))
        axis.set_title(f"true: {classes[true_label]}\npred: {classes[predicted_label]}", fontsize=9)
        axis.axis("off")

    # Hide the trailing axes of a partially filled final row.
    for axis in flat[len(selected) :]:
        axis.axis("off")

    figure.suptitle(f"Misclassified examples ({len(selected)} of {len(samples)})")
    _save(figure, path)
    return True


def plot_loss_curves(histories: Sequence[TrainingHistory], path: Path) -> None:
    """Plot train and validation loss for one or more runs."""
    figure, axes = plt.subplots(figsize=(8, 4.5))
    for history in histories:
        train_epochs, train_values = history.series("train_loss")
        val_epochs, val_values = history.series("val_loss")
        axes.plot(train_epochs, train_values, label=f"{history.run_name} train")
        axes.plot(val_epochs, val_values, linestyle="--", label=f"{history.run_name} val")
    axes.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training and validation loss")
    axes.grid(visible=True, alpha=0.3)
    axes.legend()
    _save(figure, path)


def plot_f1_curves(histories: Sequence[TrainingHistory], path: Path) -> None:
    """Plot validation macro-F1 for one or more runs."""
    figure, axes = plt.subplots(figsize=(8, 4.5))
    for history in histories:
        epochs, values = history.series("val_f1")
        axes.plot(epochs, values, label=f"{history.run_name} val F1")
    axes.set(xlabel="Epoch", ylabel="Macro F1", title="Validation macro-F1")
    axes.grid(visible=True, alpha=0.3)
    axes.legend()
    _save(figure, path)
