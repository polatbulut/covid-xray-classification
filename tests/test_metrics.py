from __future__ import annotations

import pytest

from covid_xray.metrics import ClassificationMetrics, compute_metrics


def test_perfect_predictions() -> None:
    labels = [0, 1, 2, 3, 0, 1]
    metrics = compute_metrics(labels, labels)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)


def test_all_wrong_predictions() -> None:
    metrics = compute_metrics([0, 0, 1, 1], [1, 1, 0, 0])
    assert metrics.accuracy == pytest.approx(0.0)
    assert metrics.f1 == pytest.approx(0.0)


def test_known_confusion() -> None:
    # Three of four correct: one true-1 predicted as 0.
    metrics = compute_metrics([0, 0, 1, 1], [0, 0, 0, 1])
    assert metrics.accuracy == pytest.approx(0.75)
    # Class 0: precision 2/3, recall 1.0. Class 1: precision 1.0, recall 0.5.
    assert metrics.precision == pytest.approx((2 / 3 + 1.0) / 2)
    assert metrics.recall == pytest.approx((1.0 + 0.5) / 2)


def test_argument_order_is_labels_then_predictions() -> None:
    """Precision and recall swap if the arguments are transposed, so pin the order."""
    labels = [0, 0, 0, 1]
    predictions = [0, 0, 1, 1]
    forward = compute_metrics(labels, predictions)
    # macro precision = (1.0 + 0.5) / 2, macro recall = (2/3 + 1.0) / 2
    assert forward.precision == pytest.approx(0.75)
    assert forward.recall == pytest.approx((2 / 3 + 1.0) / 2)
    backward = compute_metrics(predictions, labels)
    assert forward.precision == pytest.approx(backward.recall)
    assert forward.recall == pytest.approx(backward.precision)


def test_macro_average_weights_classes_equally() -> None:
    """A rare class predicted entirely wrong must drag the macro F1 down."""
    labels = [0] * 99 + [1]
    predictions = [0] * 100
    metrics = compute_metrics(labels, predictions)
    assert metrics.accuracy == pytest.approx(0.99)
    assert metrics.f1 < 0.6


def test_zero_division_is_handled() -> None:
    metrics = compute_metrics([0, 0, 0], [0, 0, 0], average="macro")
    assert metrics.f1 == pytest.approx(1.0)


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="equal length"):
        compute_metrics([0, 1], [0])


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([], [])


def test_as_dict_and_summary() -> None:
    metrics = ClassificationMetrics(accuracy=0.5, precision=0.25, recall=0.75, f1=0.375)
    assert metrics.as_dict() == {
        "accuracy": 0.5,
        "precision": 0.25,
        "recall": 0.75,
        "f1": 0.375,
    }
    assert "accuracy=0.5000" in metrics.format_summary()


def test_as_dict_accepts_a_prefix() -> None:
    """Checkpoints store val_-prefixed keys so they match `training.monitor`."""
    metrics = ClassificationMetrics(accuracy=0.5, precision=0.25, recall=0.75, f1=0.375)
    assert set(metrics.as_dict(prefix="val_")) == {
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
    }
