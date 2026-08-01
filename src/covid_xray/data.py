"""Dataset, transform and dataloader construction.

Images are expected in the standard ``<root>/<class name>/<image>`` layout that
:mod:`covid_xray.datasplit` produces. Class discovery and file ordering are both
sorted, so the label mapping and the sample order are stable across machines --
a prerequisite for reproducible runs and for comparing evaluation artefacts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from covid_xray.config import DataConfig, TrainingConfig
from covid_xray.runtime import seed_worker

LOGGER: Final = logging.getLogger(__name__)

IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".png", ".jpg", ".jpeg"})

Sample = tuple[Path, int]
Batch = tuple[torch.Tensor, int]


class DatasetError(RuntimeError):
    """Raised when an image directory does not have the expected layout."""


class XRayDataset(Dataset[Batch]):
    """Chest X-ray images stored as ``<root>/<class name>/<image>``.

    Args:
        root: Directory containing one sub-directory per class.
        transform: Callable applied to each PIL image. Defaults to a plain
            tensor conversion.
        classes: Explicit class ordering. Pass the class tuple recorded in a
            checkpoint so that validation and test splits use exactly the same
            label indices as training did. When ``None``, classes are
            discovered from the directory listing in sorted order.

    Raises:
        DatasetError: If the directory is missing, holds no class
            sub-directories, holds no images, or lacks a requested class.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        classes: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise DatasetError(f"Dataset directory does not exist: {self.root}")

        self.transform: Callable[[Image.Image], torch.Tensor] = (
            transform if transform is not None else transforms.ToTensor()
        )

        discovered = tuple(sorted(entry.name for entry in self.root.iterdir() if entry.is_dir()))
        if not discovered:
            raise DatasetError(f"No class sub-directories found under {self.root}.")

        if classes is None:
            self.classes: tuple[str, ...] = discovered
        else:
            missing = sorted(set(classes) - set(discovered))
            if missing:
                raise DatasetError(
                    f"Classes {missing} are missing from {self.root}. Found: {list(discovered)}."
                )
            self.classes = tuple(classes)

        self.class_to_idx: dict[str, int] = {name: i for i, name in enumerate(self.classes)}
        self.samples: tuple[Sample, ...] = self._collect_samples()
        if not self.samples:
            extensions = sorted(IMAGE_EXTENSIONS)
            raise DatasetError(f"No images with extensions {extensions} found under {self.root}.")

    def _collect_samples(self) -> tuple[Sample, ...]:
        """Return every ``(path, label)`` pair in a deterministic order."""
        samples: list[Sample] = []
        for name in self.classes:
            class_dir = self.root / name
            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not paths:
                LOGGER.warning("Class directory %s contains no images.", class_dir)
            label = self.class_to_idx[name]
            samples.extend((path, label) for path in paths)
        return tuple(samples)

    def __len__(self) -> int:
        """Return the number of images in the split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Batch:
        """Return the transformed image and integer label at ``index``."""
        path, label = self.samples[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        return self.transform(image), label

    @property
    def targets(self) -> tuple[int, ...]:
        """Return the label of every sample, in dataset order."""
        return tuple(label for _, label in self.samples)

    def class_counts(self) -> tuple[int, ...]:
        """Return the number of images per class, indexed like :attr:`classes`."""
        counts = [0] * len(self.classes)
        for _, label in self.samples:
            counts[label] += 1
        return tuple(counts)


def build_train_transform(data: DataConfig) -> transforms.Compose:
    """Build the training pipeline: resize, optional augmentation, normalise."""
    steps: list[Callable[..., object]] = [transforms.Resize(data.img_size)]
    augment = data.augment
    if augment.random_horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    if augment.random_rotation > 0.0:
        steps.append(transforms.RandomRotation(augment.random_rotation))
    if augment.brightness_jitter > 0.0 or augment.contrast_jitter > 0.0:
        steps.append(
            transforms.ColorJitter(
                brightness=augment.brightness_jitter,
                contrast=augment.contrast_jitter,
            )
        )
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(data.mean, data.std))
    return transforms.Compose(steps)


def build_eval_transform(data: DataConfig) -> transforms.Compose:
    """Build the deterministic validation and test pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(data.img_size),
            transforms.ToTensor(),
            transforms.Normalize(data.mean, data.std),
        ]
    )


def build_dataloader(
    dataset: XRayDataset,
    training: TrainingConfig,
    device: torch.device,
    *,
    shuffle: bool,
) -> DataLoader[Batch]:
    """Build a dataloader with seeded workers and device-appropriate pinning."""
    generator = torch.Generator()
    generator.manual_seed(training.seed)
    use_workers = training.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=training.batch_size,
        shuffle=shuffle,
        num_workers=training.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=use_workers,
        worker_init_fn=seed_worker if use_workers else None,
        generator=generator,
    )
