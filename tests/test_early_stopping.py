from __future__ import annotations

import pytest

from covid_xray.early_stopping import EarlyStopping


def test_first_value_always_improves() -> None:
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.update(0.0) is True
    assert stopper.best == 0.0


def test_max_mode_tracks_the_highest_value() -> None:
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.update(0.50) is True
    assert stopper.update(0.60) is True
    assert stopper.update(0.55) is False
    assert stopper.best == pytest.approx(0.60)


def test_min_mode_tracks_the_lowest_value() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper.update(1.0) is True
    assert stopper.update(0.5) is True
    assert stopper.update(0.7) is False
    assert stopper.best == pytest.approx(0.5)


def test_stops_after_patience_is_exhausted() -> None:
    stopper = EarlyStopping(patience=2, mode="max")
    stopper.update(0.9)
    assert stopper.should_stop is False
    stopper.update(0.1)
    assert stopper.should_stop is False
    stopper.update(0.1)
    assert stopper.should_stop is True


def test_improvement_resets_the_counter() -> None:
    stopper = EarlyStopping(patience=2, mode="max")
    stopper.update(0.5)
    stopper.update(0.4)
    assert stopper.epochs_without_improvement == 1
    stopper.update(0.6)
    assert stopper.epochs_without_improvement == 0
    assert stopper.should_stop is False


def test_zero_patience_never_stops() -> None:
    """patience=0 keeps tracking the best value but disables stopping."""
    stopper = EarlyStopping(patience=0, mode="max")
    stopper.update(0.9)
    for _ in range(10):
        stopper.update(0.1)
    assert stopper.should_stop is False
    assert stopper.best == pytest.approx(0.9)


def test_min_delta_ignores_noise() -> None:
    stopper = EarlyStopping(patience=1, mode="max", min_delta=0.01)
    assert stopper.update(0.900) is True
    assert stopper.update(0.905) is False
    assert stopper.update(0.950) is True


@pytest.mark.parametrize(("patience", "min_delta"), [(-1, 0.0), (1, -0.5)])
def test_invalid_arguments_are_rejected(patience: int, min_delta: float) -> None:
    with pytest.raises(ValueError, match=">= 0"):
        EarlyStopping(patience=patience, min_delta=min_delta)
