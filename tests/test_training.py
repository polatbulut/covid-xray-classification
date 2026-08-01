from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from covid_xray.checkpoints import load_checkpoint
from covid_xray.training import compute_class_weights, fit

from .conftest import CLASS_NAMES, make_experiment


def test_no_weighting_returns_none() -> None:
    assert compute_class_weights([10, 20], "none") is None


def test_inverse_weighting_favours_rare_classes() -> None:
    weights = compute_class_weights([10, 90], "inverse")
    assert weights is not None
    assert weights[0] > weights[1]
    assert weights == pytest.approx([10.0, 100 / 90])


def test_balanced_weighting_matches_sklearn_formula() -> None:
    weights = compute_class_weights([10, 90], "balanced")
    assert weights is not None
    assert weights == pytest.approx([100 / (2 * 10), 100 / (2 * 90)])


def test_inverse_is_a_scaled_balanced() -> None:
    """`inverse` is `balanced` times the class count -- documented, not accidental."""
    counts = [10, 30, 60]
    inverse = compute_class_weights(counts, "inverse")
    balanced = compute_class_weights(counts, "balanced")
    assert inverse is not None and balanced is not None
    assert inverse == pytest.approx([value * len(counts) for value in balanced])


def test_empty_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_class_weights([], "inverse")


def test_zero_count_class_does_not_divide_by_zero() -> None:
    weights = compute_class_weights([10, 0], "inverse")
    assert weights is not None
    assert weights[1] == 0.0


def test_fit_produces_a_checkpoint_and_history(tmp_path: Path) -> None:
    config = make_experiment(tmp_path)
    history = fit(config, torch.device("cpu"), show_progress=False)

    assert len(history) == 2
    assert config.training.checkpoint_path.is_file()

    history_path = config.training.resolved_history_path
    assert history_path.is_file()
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 2
    assert payload["run_name"] == "run"


def test_checkpoint_records_the_training_class_order(tmp_path: Path) -> None:
    """This is what lets evaluation avoid re-deriving labels from the test folder."""
    config = make_experiment(tmp_path, epochs=1)
    fit(config, torch.device("cpu"), show_progress=False)
    checkpoint = load_checkpoint(config.training.checkpoint_path)
    assert checkpoint.classes == tuple(sorted(CLASS_NAMES))
    assert checkpoint.model_name == "simple"
    assert "val_f1" in checkpoint.metrics


def test_history_records_every_metric(tmp_path: Path) -> None:
    config = make_experiment(tmp_path, epochs=1)
    history = fit(config, torch.device("cpu"), show_progress=False)
    record = history.records[0]
    assert record.epoch == 1
    assert record.val_accuracy is not None
    assert record.val_precision is not None
    assert record.val_recall is not None
    assert record.learning_rate is not None


def test_training_is_reproducible_for_a_fixed_seed(tmp_path: Path) -> None:
    device = torch.device("cpu")
    first = fit(make_experiment(tmp_path / "a", epochs=1), device, show_progress=False)
    second = fit(make_experiment(tmp_path / "b", epochs=1), device, show_progress=False)
    assert first.records[0].train_loss == pytest.approx(second.records[0].train_loss)


def test_early_stopping_can_end_a_run_early(tmp_path: Path) -> None:
    """With patience=1 on a degenerate dataset the run must stop before `epochs`."""
    config = make_experiment(tmp_path, epochs=20, patience=1)
    history = fit(config, torch.device("cpu"), show_progress=False)
    assert len(history) < 20
