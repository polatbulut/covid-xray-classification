from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from covid_xray.checkpoints import (
    Checkpoint,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from covid_xray.evaluation import (
    CONFUSION_MATRIX_FILENAME,
    METRICS_FILENAME,
    REPORT_FILENAME,
    evaluate,
    write_artifacts,
)
from covid_xray.training import fit

from .conftest import CLASS_NAMES, make_experiment

DEVICE = torch.device("cpu")


def test_evaluate_returns_aligned_predictions(tmp_path: Path) -> None:
    config = make_experiment(tmp_path, epochs=1)
    fit(config, DEVICE, show_progress=False)

    result = evaluate(config, config.training.checkpoint_path, DEVICE, show_progress=False)

    assert result.classes == tuple(sorted(CLASS_NAMES))
    assert len(result.labels) == len(result.predictions) == 4 * len(CLASS_NAMES)
    assert result.confusion.shape == (len(CLASS_NAMES), len(CLASS_NAMES))
    assert result.confusion.sum() == len(result.labels)
    assert 0.0 <= result.metrics.accuracy <= 1.0
    assert len(result.misclassified) == sum(
        1 for true, pred in zip(result.labels, result.predictions, strict=True) if true != pred
    )


def test_evaluate_uses_the_checkpoint_class_order(tmp_path: Path) -> None:
    """A checkpoint trained on a reversed class order must not be silently remapped."""
    config = make_experiment(tmp_path, epochs=1)
    fit(config, DEVICE, show_progress=False)

    checkpoint = load_checkpoint(config.training.checkpoint_path)
    reversed_classes = tuple(reversed(checkpoint.classes))
    save_checkpoint(
        Checkpoint(
            epoch=checkpoint.epoch,
            model_name=checkpoint.model_name,
            classes=reversed_classes,
            model_state=checkpoint.model_state,
        ),
        config.training.checkpoint_path,
    )

    result = evaluate(config, config.training.checkpoint_path, DEVICE, show_progress=False)
    assert result.classes == reversed_classes


def test_write_artifacts_creates_every_file(tmp_path: Path) -> None:
    config = make_experiment(tmp_path, epochs=1)
    fit(config, DEVICE, show_progress=False)
    result = evaluate(config, config.training.checkpoint_path, DEVICE, show_progress=False)

    output_dir = tmp_path / "results" / "run"
    write_artifacts(result, output_dir)

    assert (output_dir / METRICS_FILENAME).is_file()
    assert (output_dir / REPORT_FILENAME).is_file()
    assert (output_dir / CONFUSION_MATRIX_FILENAME).is_file()

    payload = json.loads((output_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert payload["classes"] == list(result.classes)
    assert payload["num_samples"] == len(result.labels)
    assert set(payload["metrics"]) == {"accuracy", "precision", "recall", "f1"}


def test_write_artifacts_creates_missing_directories(tmp_path: Path) -> None:
    """The old evaluate script assumed `results/` already existed and crashed if not."""
    config = make_experiment(tmp_path, epochs=1)
    fit(config, DEVICE, show_progress=False)
    result = evaluate(config, config.training.checkpoint_path, DEVICE, show_progress=False)

    deep = tmp_path / "does" / "not" / "exist" / "yet"
    write_artifacts(result, deep)
    assert (deep / METRICS_FILENAME).is_file()


def test_evaluate_rejects_a_missing_checkpoint(tmp_path: Path) -> None:
    config = make_experiment(tmp_path, epochs=1)
    with pytest.raises(CheckpointError, match="not found"):
        evaluate(config, tmp_path / "absent.pth", DEVICE, show_progress=False)
