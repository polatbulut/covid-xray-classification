"""Test-set evaluation: metrics, per-class report and diagnostic figures."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from covid_xray.checkpoints import Checkpoint, load_checkpoint, restore_model
from covid_xray.config import ExperimentConfig
from covid_xray.data import Batch, XRayDataset, build_dataloader, build_eval_transform
from covid_xray.metrics import ClassificationMetrics, compute_metrics
from covid_xray.models import create_model
from covid_xray.plotting import MisclassifiedSample, plot_confusion_matrix, plot_misclassified
from covid_xray.runtime import amp_is_supported, autocast_context

LOGGER: Final = logging.getLogger(__name__)

CONFUSION_MATRIX_FILENAME: Final = "confusion_matrix.png"
MISCLASSIFIED_FILENAME: Final = "misclassified_grid.png"
METRICS_FILENAME: Final = "metrics.json"
REPORT_FILENAME: Final = "classification_report.txt"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Everything produced by a single pass over the test split."""

    classes: tuple[str, ...]
    metrics: ClassificationMetrics
    confusion: np.ndarray
    report: str
    labels: tuple[int, ...]
    predictions: tuple[int, ...]
    misclassified: tuple[MisclassifiedSample, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable summary of the run."""
        return {
            "classes": list(self.classes),
            "num_samples": len(self.labels),
            "num_misclassified": len(self.misclassified),
            "metrics": self.metrics.as_dict(),
            "confusion_matrix": self.confusion.tolist(),
        }


def predict(
    model: nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
    *,
    show_progress: bool = True,
) -> tuple[list[int], list[int]]:
    """Run the model over ``loader`` and return ``(labels, predictions)``."""
    model.eval()
    use_amp = amp_is_supported(device)
    labels: list[int] = []
    predictions: list[int] = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="test", leave=False, disable=not show_progress):
            images = images.to(device, non_blocking=True)
            with autocast_context(device, enabled=use_amp):
                outputs: torch.Tensor = model(images)
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())

    return labels, predictions


def evaluate(
    config: ExperimentConfig,
    checkpoint_path: Path,
    device: torch.device,
    *,
    show_progress: bool = True,
) -> EvaluationResult:
    """Load a checkpoint and evaluate it on the configured test split.

    The class ordering recorded in the checkpoint is imposed on the test
    dataset, so predictions and labels always refer to the same classes the
    model was trained on.
    """
    checkpoint: Checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    if checkpoint.model_name and checkpoint.model_name != config.model.name:
        LOGGER.warning(
            "Checkpoint was trained with model '%s' but the config asks for '%s'.",
            checkpoint.model_name,
            config.model.name,
        )

    dataset = XRayDataset(
        config.data.test_dir,
        build_eval_transform(config.data),
        classes=checkpoint.classes or None,
    )
    loader = build_dataloader(dataset, config.training, device, shuffle=False)

    model = create_model(config.model.name, len(dataset.classes), pretrained=False)
    model.to(device)
    restore_model(checkpoint, model)
    LOGGER.info(
        "Evaluating '%s' (epoch %d) on %d test images across %d classes.",
        config.model.name,
        checkpoint.epoch,
        len(dataset),
        len(dataset.classes),
    )

    labels, predictions = predict(model, loader, device, show_progress=show_progress)
    metrics = compute_metrics(labels, predictions)
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(dataset.classes))))
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(dataset.classes))),
        target_names=list(dataset.classes),
        digits=4,
        zero_division=0,
    )

    misclassified = tuple(
        (path, true_label, predicted_label)
        for (path, _), true_label, predicted_label in zip(
            dataset.samples, labels, predictions, strict=True
        )
        if true_label != predicted_label
    )

    return EvaluationResult(
        classes=dataset.classes,
        metrics=metrics,
        confusion=np.asarray(matrix),
        report=str(report),
        labels=tuple(labels),
        predictions=tuple(predictions),
        misclassified=misclassified,
    )


def write_artifacts(
    result: EvaluationResult,
    output_dir: Path,
    *,
    max_misclassified: int = 9,
) -> None:
    """Write metrics, the per-class report and diagnostic figures to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / METRICS_FILENAME
    metrics_path.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", metrics_path)

    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(result.report + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)

    plot_confusion_matrix(result.confusion, result.classes, output_dir / CONFUSION_MATRIX_FILENAME)
    plot_misclassified(
        result.misclassified,
        result.classes,
        output_dir / MISCLASSIFIED_FILENAME,
        max_examples=max_misclassified,
    )
