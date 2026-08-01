"""Classification metric helpers.

Metrics are macro-averaged by default: the dataset is imbalanced, and a
micro-average would let the majority ``Normal`` class hide poor recall on
``COVID``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEFAULT_AVERAGE: Final = "macro"


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Accuracy plus macro-averaged precision, recall and F1."""

    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a JSON-serialisable mapping."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def format_summary(self) -> str:
        """Return a single-line, aligned summary suitable for logging."""
        return (
            f"accuracy={self.accuracy:.4f} precision={self.precision:.4f} "
            f"recall={self.recall:.4f} f1={self.f1:.4f}"
        )


def compute_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    average: str = DEFAULT_AVERAGE,
) -> ClassificationMetrics:
    """Compute classification metrics.

    Args:
        labels: Ground-truth class indices.
        predictions: Predicted class indices, aligned with ``labels``.
        average: Averaging strategy passed through to scikit-learn.

    Raises:
        ValueError: If the two sequences have different lengths or are empty.
    """
    if len(labels) != len(predictions):
        raise ValueError(
            f"labels and predictions must have equal length, "
            f"got {len(labels)} and {len(predictions)}."
        )
    if not labels:
        raise ValueError("Cannot compute metrics over an empty sequence.")

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )
