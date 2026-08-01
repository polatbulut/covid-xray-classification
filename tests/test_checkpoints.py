from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from covid_xray.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    Checkpoint,
    CheckpointError,
    load_checkpoint,
    restore_model,
    save_checkpoint,
)
from covid_xray.models import SimpleCNN

CLASSES = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")


def make_checkpoint(model: SimpleCNN) -> Checkpoint:
    return Checkpoint(
        epoch=7,
        model_name="simple",
        classes=CLASSES,
        model_state=model.state_dict(),
        metrics={"val_f1": 0.9},
    )


def test_round_trip_preserves_metadata(tmp_path: Path) -> None:
    model = SimpleCNN(num_classes=4)
    path = tmp_path / "nested" / "run.pth"
    save_checkpoint(make_checkpoint(model), path)

    loaded = load_checkpoint(path)
    assert loaded.epoch == 7
    assert loaded.model_name == "simple"
    assert loaded.classes == CLASSES
    assert loaded.metrics["val_f1"] == pytest.approx(0.9)
    assert loaded.format_version == CHECKPOINT_FORMAT_VERSION


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c" / "run.pth"
    save_checkpoint(make_checkpoint(SimpleCNN(num_classes=4)), path)
    assert path.is_file()


def test_restore_model_reproduces_predictions(tmp_path: Path) -> None:
    torch.manual_seed(0)
    original = SimpleCNN(num_classes=4)
    original.eval()
    batch = torch.randn(2, 3, 32, 32)
    expected = original(batch)

    path = tmp_path / "run.pth"
    save_checkpoint(make_checkpoint(original), path)

    restored = SimpleCNN(num_classes=4)
    restore_model(load_checkpoint(path), restored)
    restored.eval()
    assert torch.allclose(restored(batch), expected)


def test_optimizer_state_round_trip(tmp_path: Path) -> None:
    model = SimpleCNN(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.randn(1, 3, 32, 32)).sum().backward()
    optimizer.step()

    path = tmp_path / "run.pth"
    checkpoint = make_checkpoint(model)
    checkpoint.optimizer_state = optimizer.state_dict()
    save_checkpoint(checkpoint, path)

    restored_model = SimpleCNN(num_classes=4)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    restore_model(load_checkpoint(path), restored_model, restored_optimizer)
    assert restored_optimizer.state_dict()["param_groups"][0]["lr"] == pytest.approx(1e-3)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "absent.pth")


def test_corrupt_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pth"
    path.write_bytes(b"definitely not a torch archive")
    with pytest.raises(CheckpointError, match="Could not load"):
        load_checkpoint(path)


def test_payload_without_model_state_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "run.pth"
    torch.save({"epoch": 1}, path)
    with pytest.raises(CheckpointError, match="model_state"):
        load_checkpoint(path)


def test_legacy_checkpoint_loads_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Pre-refactor checkpoints used `optim_state` and recorded no class order."""
    model = SimpleCNN(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "legacy.pth"
    torch.save(
        {
            "epoch": 3,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )

    with caplog.at_level(logging.WARNING, logger="covid_xray.checkpoints"):
        loaded = load_checkpoint(path)

    assert loaded.epoch == 3
    assert loaded.classes == ()
    assert loaded.optimizer_state is not None
    assert loaded.format_version == 0
    assert "class ordering" in caplog.text


def test_restore_model_rejects_mismatched_architecture(tmp_path: Path) -> None:
    path = tmp_path / "run.pth"
    save_checkpoint(make_checkpoint(SimpleCNN(num_classes=4)), path)
    with pytest.raises(CheckpointError, match="do not match"):
        restore_model(load_checkpoint(path), SimpleCNN(num_classes=7))
