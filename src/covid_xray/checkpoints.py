"""Checkpoint serialisation.

A checkpoint records the class ordering used during training alongside the
weights. Without it, evaluation has to re-derive labels from whatever
directories happen to exist on disk, which silently produces wrong metrics if
the two orderings ever disagree.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn
from torch.optim import Optimizer

LOGGER: Final = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION: Final = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint file is missing or does not match the schema."""


@dataclass(slots=True)
class Checkpoint:
    """The saved state of a training run at its best epoch."""

    epoch: int
    model_name: str
    classes: tuple[str, ...]
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    format_version: int = CHECKPOINT_FORMAT_VERSION

    def to_payload(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for :func:`torch.save`."""
        return {
            "format_version": self.format_version,
            "epoch": self.epoch,
            "model_name": self.model_name,
            "classes": list(self.classes),
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "metrics": self.metrics,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, source: Path | None = None) -> Checkpoint:
        """Build a checkpoint from a loaded payload, tolerating the pre-1.0 layout.

        Checkpoints written before this schema existed used ``optim_state`` and
        recorded neither the class ordering nor the model name.
        """
        origin = f" in {source}" if source is not None else ""
        model_state = payload.get("model_state")
        if not isinstance(model_state, dict):
            raise CheckpointError(f"Checkpoint{origin} does not contain a 'model_state' mapping.")

        classes = payload.get("classes")
        if not isinstance(classes, list | tuple) or not all(
            isinstance(name, str) for name in classes
        ):
            LOGGER.warning(
                "Checkpoint%s does not record its class ordering; falling back to the class "
                "order discovered on disk. Re-train to embed the ordering.",
                origin,
            )
            classes = []

        optimizer_state = payload.get("optimizer_state", payload.get("optim_state"))
        metrics = payload.get("metrics")

        return cls(
            epoch=int(payload.get("epoch", 0)),
            model_name=str(payload.get("model_name", "")),
            classes=tuple(str(name) for name in classes),
            model_state=model_state,
            optimizer_state=optimizer_state if isinstance(optimizer_state, dict) else None,
            metrics=dict(metrics) if isinstance(metrics, dict) else {},
            format_version=int(payload.get("format_version", 0)),
        )


def save_checkpoint(checkpoint: Checkpoint, path: Path) -> None:
    """Write ``checkpoint`` to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.to_payload(), path)
    LOGGER.debug("Wrote checkpoint for epoch %d to %s", checkpoint.epoch, path)


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> Checkpoint:
    """Load a checkpoint from ``path``.

    The payload holds only tensors and plain Python values, so it is loaded with
    ``weights_only=True``; arbitrary pickled objects are rejected.

    Raises:
        CheckpointError: If the file is missing or malformed.
    """
    if not path.is_file():
        raise CheckpointError(f"Checkpoint file not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"Could not load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"Checkpoint {path} must contain a dictionary payload.")
    return Checkpoint.from_payload(payload, source=path)


def restore_model(
    checkpoint: Checkpoint,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> None:
    """Load weights (and optionally optimiser state) from ``checkpoint`` in place.

    Raises:
        CheckpointError: If the weights do not match the given model.
    """
    try:
        model.load_state_dict(checkpoint.model_state)
    except (RuntimeError, KeyError) as exc:
        raise CheckpointError(
            f"Checkpoint weights do not match the model architecture: {exc}"
        ) from exc
    if optimizer is not None and checkpoint.optimizer_state is not None:
        optimizer.load_state_dict(checkpoint.optimizer_state)
