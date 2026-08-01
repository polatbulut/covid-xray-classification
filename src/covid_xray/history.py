"""Per-epoch training history.

The history is written to JSON after every epoch so that a run which is
interrupted still leaves a usable record, and so that training curves can be
re-plotted later from data instead of from numbers pasted into a script.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final

from covid_xray.config import MonitorMode

_REQUIRED_FIELDS: Final[tuple[str, ...]] = ("epoch", "train_loss", "val_loss", "val_f1")


class HistoryError(ValueError):
    """Raised when a history file cannot be parsed."""


def _optional_float(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` when it was not recorded."""
    return None if value is None else float(value)


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Metrics for a single training epoch.

    Only the four fields needed to plot loss and F1 curves are required. The
    rest are optional so that histories recovered from older runs, which did not
    record every metric, still load.
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_f1: float
    val_accuracy: float | None = None
    val_precision: float | None = None
    val_recall: float | None = None
    learning_rate: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-serialisable mapping."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_f1": self.val_f1,
            "val_accuracy": self.val_accuracy,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpochRecord:
        """Build a record from a parsed JSON object, ignoring unknown keys."""
        missing = [name for name in _REQUIRED_FIELDS if data.get(name) is None]
        if missing:
            raise HistoryError(f"Epoch record is missing required field(s): {missing}.")
        return cls(
            epoch=int(data["epoch"]),
            train_loss=float(data["train_loss"]),
            val_loss=float(data["val_loss"]),
            val_f1=float(data["val_f1"]),
            val_accuracy=_optional_float(data.get("val_accuracy")),
            val_precision=_optional_float(data.get("val_precision")),
            val_recall=_optional_float(data.get("val_recall")),
            learning_rate=_optional_float(data.get("learning_rate")),
        )

    def value_of(self, metric: str) -> float:
        """Return the named metric, or raise if it was never recorded."""
        value = self.as_dict().get(metric)
        if not isinstance(value, int | float):
            raise HistoryError(f"Epoch {self.epoch} has no recorded value for {metric!r}.")
        return float(value)


@dataclass(slots=True)
class TrainingHistory:
    """An ordered collection of :class:`EpochRecord` values for one run."""

    run_name: str
    records: list[EpochRecord] = field(default_factory=list)

    CSV_COLUMNS: ClassVar[tuple[str, ...]] = (
        "epoch",
        "train_loss",
        "val_loss",
        "val_f1",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "learning_rate",
    )

    def append(self, record: EpochRecord) -> None:
        """Add a record for the next epoch."""
        self.records.append(record)

    def __len__(self) -> int:
        """Return the number of recorded epochs."""
        return len(self.records)

    def __iter__(self) -> Iterator[EpochRecord]:
        """Iterate over the recorded epochs in order."""
        return iter(self.records)

    def best(self, metric: str, mode: MonitorMode) -> EpochRecord | None:
        """Return the epoch with the best value of ``metric``, or ``None`` if empty."""
        if not self.records:
            return None
        if mode == "max":
            return max(self.records, key=lambda record: record.value_of(metric))
        return min(self.records, key=lambda record: record.value_of(metric))

    def series(self, metric: str) -> tuple[list[int], list[float]]:
        """Return ``(epochs, values)`` for ``metric``, skipping epochs that lack it."""
        epochs: list[int] = []
        values: list[float] = []
        for record in self.records:
            value = record.as_dict().get(metric)
            if isinstance(value, int | float):
                epochs.append(record.epoch)
                values.append(float(value))
        return epochs, values

    def to_json(self, path: Path) -> None:
        """Write the history to ``path`` as JSON, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": self.run_name,
            "records": [record.as_dict() for record in self.records],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def to_csv(self, path: Path) -> None:
        """Write the history to ``path`` as CSV, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.CSV_COLUMNS))
            writer.writeheader()
            for record in self.records:
                writer.writerow(record.as_dict())

    @classmethod
    def from_json(cls, path: Path) -> TrainingHistory:
        """Load a history written by :meth:`to_json`.

        Raises:
            HistoryError: If the file is unreadable or does not match the schema.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise HistoryError(f"Could not read history file {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise HistoryError(f"Could not parse JSON in {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise HistoryError(f"History file {path} must contain a JSON object.")

        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise HistoryError(f"History file {path} is missing a 'records' list.")

        run_name = payload.get("run_name")
        return cls(
            run_name=run_name if isinstance(run_name, str) else path.stem,
            records=[EpochRecord.from_dict(item) for item in raw_records],
        )
