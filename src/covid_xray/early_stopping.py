"""Early stopping on a monitored validation metric."""

from __future__ import annotations

from dataclasses import dataclass, field

from covid_xray.config import MonitorMode


@dataclass(slots=True)
class EarlyStopping:
    """Track a monitored metric and report when it has stopped improving.

    Args:
        patience: Number of consecutive non-improving epochs to tolerate before
            :attr:`should_stop` becomes ``True``. ``0`` disables early stopping
            entirely while still tracking the best value.
        mode: ``"max"`` when a higher metric value is better (accuracy, F1),
            ``"min"`` when lower is better (loss).
        min_delta: Minimum change that counts as an improvement. Guards against
            declaring victory on floating-point noise.
    """

    patience: int
    mode: MonitorMode = "max"
    min_delta: float = 0.0
    best: float | None = field(default=None, init=False)
    epochs_without_improvement: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate the configured patience and delta."""
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}.")
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}.")

    def _is_improvement(self, value: float) -> bool:
        """Return ``True`` when ``value`` beats the current best by ``min_delta``."""
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, value: float) -> bool:
        """Record a monitored value and return whether it improved on the best so far."""
        if self._is_improvement(value):
            self.best = value
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Return ``True`` once patience has been exhausted."""
        return self.patience > 0 and self.epochs_without_improvement >= self.patience
