from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from covid_xray.history import EpochRecord, HistoryError, TrainingHistory

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_history() -> TrainingHistory:
    history = TrainingHistory(run_name="demo")
    history.append(
        EpochRecord(
            epoch=1,
            train_loss=0.50,
            val_loss=0.40,
            val_f1=0.80,
            val_accuracy=0.81,
            val_precision=0.82,
            val_recall=0.83,
            learning_rate=1e-4,
        )
    )
    history.append(
        EpochRecord(
            epoch=2,
            train_loss=0.30,
            val_loss=0.45,
            val_f1=0.90,
            val_accuracy=0.91,
            val_precision=0.92,
            val_recall=0.93,
            learning_rate=5e-5,
        )
    )
    return history


def test_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run_history.json"
    make_history().to_json(path)
    loaded = TrainingHistory.from_json(path)
    assert loaded.run_name == "demo"
    assert len(loaded) == 2
    assert loaded.records == make_history().records


def test_to_json_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "run_history.json"
    make_history().to_json(path)
    assert path.is_file()


def test_csv_export(tmp_path: Path) -> None:
    path = tmp_path / "run_history.csv"
    make_history().to_csv(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    assert list(rows[0]) == list(TrainingHistory.CSV_COLUMNS)


def test_best_respects_mode() -> None:
    history = make_history()
    best_f1 = history.best("val_f1", "max")
    best_loss = history.best("val_loss", "min")
    assert best_f1 is not None and best_f1.epoch == 2
    assert best_loss is not None and best_loss.epoch == 1


def test_best_of_empty_history_is_none() -> None:
    assert TrainingHistory(run_name="empty").best("val_f1", "max") is None


def test_series_skips_unrecorded_metrics() -> None:
    history = TrainingHistory(run_name="partial")
    history.append(EpochRecord(epoch=1, train_loss=0.5, val_loss=0.4, val_f1=0.7))
    assert history.series("val_f1") == ([1], [0.7])
    assert history.series("val_accuracy") == ([], [])


def test_value_of_unrecorded_metric_raises() -> None:
    record = EpochRecord(epoch=1, train_loss=0.5, val_loss=0.4, val_f1=0.7)
    with pytest.raises(HistoryError, match="val_accuracy"):
        record.value_of("val_accuracy")


def test_optional_fields_default_to_none() -> None:
    record = EpochRecord(epoch=1, train_loss=0.5, val_loss=0.4, val_f1=0.7)
    assert record.val_accuracy is None
    assert record.as_dict()["learning_rate"] is None


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(HistoryError, match="val_loss"):
        EpochRecord.from_dict({"epoch": 1, "train_loss": 0.1, "val_f1": 0.5})


def test_unknown_fields_are_ignored() -> None:
    record = EpochRecord.from_dict(
        {"epoch": 1, "train_loss": 0.1, "val_loss": 0.2, "val_f1": 0.3, "gpu_temp": 71}
    )
    assert record.epoch == 1


def test_from_json_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(HistoryError, match="JSON object"):
        TrainingHistory.from_json(path)


def test_from_json_rejects_missing_records(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"run_name": "x"}), encoding="utf-8")
    with pytest.raises(HistoryError, match="records"):
        TrainingHistory.from_json(path)


def test_from_json_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(HistoryError, match="parse JSON"):
        TrainingHistory.from_json(path)


def test_from_json_missing_file(tmp_path: Path) -> None:
    with pytest.raises(HistoryError, match="Could not read"):
        TrainingHistory.from_json(tmp_path / "absent.json")


@pytest.mark.parametrize(
    "history_path",
    sorted((REPO_ROOT / "results" / "history").glob("*.json")),
    ids=lambda p: p.name,
)
def test_recorded_histories_load(history_path: Path) -> None:
    """The committed run histories must stay loadable as the schema evolves."""
    history = TrainingHistory.from_json(history_path)
    assert len(history) > 0
    best = history.best("val_f1", "max")
    assert best is not None
    assert 0.0 <= best.val_f1 <= 1.0
