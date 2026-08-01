"""Device selection, seeding and mixed-precision helpers.

These utilities exist so that the training and evaluation entry points make the
same choices about hardware and reproducibility without duplicating the logic.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
from contextlib import AbstractContextManager
from typing import Final, cast

import numpy as np
import torch

LOGGER: Final = logging.getLogger(__name__)


def resolve_device(preference: str | None = None) -> torch.device:
    """Return the compute device to use.

    Args:
        preference: An explicit device string such as ``"cuda"``, ``"mps"`` or
            ``"cpu"``. When ``None``, the best available device is selected:
            CUDA first, then Apple MPS, then CPU.

    Raises:
        RuntimeError: If an explicitly requested accelerator is unavailable.
    """
    if preference is not None:
        device = torch.device(preference)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but no MPS device is available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Return a human-readable description of a device for logging."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return device.type


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed the Python, NumPy and PyTorch random number generators.

    Args:
        seed: The seed applied to every generator.
        deterministic: When ``True``, disable cuDNN autotuning and force
            deterministic kernels. This makes runs bit-reproducible on the same
            hardware at the cost of throughput.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a dataloader worker so that augmentations are reproducible."""
    del worker_id  # The per-worker seed is derived from torch's base seed.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def amp_is_supported(device: torch.device) -> bool:
    """Return ``True`` when automatic mixed precision should be used on ``device``.

    Only CUDA is enabled here. Autocast on CPU and MPS is either slower or only
    partially supported for the operations used by these models.
    """
    return device.type == "cuda"


def autocast_context(device: torch.device, *, enabled: bool) -> AbstractContextManager[None]:
    """Return an autocast context manager, or a no-op when AMP is disabled."""
    if not enabled:
        return contextlib.nullcontext()
    return cast(AbstractContextManager[None], torch.autocast(device_type=device.type))
