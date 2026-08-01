"""Deterministic train/validation/test split of the raw Kaggle dataset.

The COVID-19 Radiography Database ships as one directory per class, each with an
``images`` sub-directory. The exact name of the extracted top-level directory has
changed between releases of the dataset, so it is resolved at runtime rather than
hard-coded.
"""

from __future__ import annotations

import logging
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

LOGGER: Final = logging.getLogger(__name__)

DEFAULT_CLASSES: Final[tuple[str, ...]] = (
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia",
)

#: Directory names the dataset has shipped under, newest first.
RAW_DIR_CANDIDATES: Final[tuple[str, ...]] = (
    "COVID-19_Radiography_Dataset",
    "COVID-19 Radiography Database",
    "covid19_radiography_dataset",
)

IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (".png", ".jpg", ".jpeg")

SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "val", "test")


class SplitError(RuntimeError):
    """Raised when the raw dataset cannot be located or is laid out unexpectedly."""


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Fractions of each class assigned to the train, validation and test splits."""

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        """Validate that the ratios are positive and sum to one."""
        for name, value in (("train", self.train), ("val", self.val), ("test", self.test)):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} ratio must be between 0 and 1, got {value}.")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}.")


def _class_image_dir(class_dir: Path) -> Path:
    """Return the directory holding a class's images.

    Newer releases nest images under ``<class>/images``; older ones place them
    directly in ``<class>``.
    """
    nested = class_dir / "images"
    return nested if nested.is_dir() else class_dir


def _list_images(directory: Path) -> list[Path]:
    """Return every image in ``directory``, sorted for reproducibility."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_raw_dir(raw_dir: Path, classes: Sequence[str] = DEFAULT_CLASSES) -> Path:
    """Locate the directory that actually holds the class folders.

    ``raw_dir`` may be the dataset directory itself or the parent it was
    extracted into.

    Raises:
        SplitError: If no directory containing the expected classes is found.
    """
    if not raw_dir.is_dir():
        raise SplitError(f"Raw data directory does not exist: {raw_dir}")

    def holds_classes(candidate: Path) -> bool:
        """Return ``True`` when ``candidate`` has a sub-directory for every class."""
        return all((candidate / name).is_dir() for name in classes)

    if holds_classes(raw_dir):
        return raw_dir

    for name in RAW_DIR_CANDIDATES:
        candidate = raw_dir / name
        if candidate.is_dir() and holds_classes(candidate):
            return candidate

    for candidate in sorted(entry for entry in raw_dir.iterdir() if entry.is_dir()):
        if holds_classes(candidate):
            LOGGER.info("Resolved dataset directory to %s", candidate)
            return candidate

    raise SplitError(
        f"Could not find a directory under {raw_dir} containing the classes {list(classes)}. "
        f"Run scripts/download_data.sh first, or pass --raw-dir explicitly."
    )


def split_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    ratios: SplitRatios | None = None,
    classes: Sequence[str] = DEFAULT_CLASSES,
    seed: int = 42,
    link: bool = False,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """Copy (or symlink) each class's images into train/val/test directories.

    Args:
        raw_dir: The extracted dataset, or the directory it was extracted into.
        output_dir: Destination root; ``<output_dir>/<split>/<class>`` is created.
        ratios: Split fractions. Defaults to 70/15/15.
        classes: Class directories to include.
        seed: Seed for the shuffle, so the split is reproducible.
        link: Create symlinks instead of copies. Saves duplicating several GB,
            but the split then depends on the raw data staying in place.
        overwrite: Replace an existing, non-empty ``output_dir``.

    Returns:
        A mapping of class name to a per-split image count.

    Raises:
        SplitError: If the raw layout is unexpected or the output already exists.
    """
    ratios = ratios or SplitRatios()
    dataset_dir = resolve_raw_dir(raw_dir, classes)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise SplitError(
                f"Output directory {output_dir} already exists and is not empty. "
                f"Pass --overwrite to replace it."
            )
        LOGGER.warning("Removing existing split at %s", output_dir)
        shutil.rmtree(output_dir)

    rng = random.Random(seed)
    summary: dict[str, dict[str, int]] = {}

    for class_name in classes:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise SplitError(f"Expected class directory {class_dir} does not exist.")

        image_dir = _class_image_dir(class_dir)
        images = _list_images(image_dir)
        if not images:
            raise SplitError(f"No images found for class {class_name!r} under {image_dir}.")

        rng.shuffle(images)
        total = len(images)
        n_train = int(ratios.train * total)
        n_val = int(ratios.val * total)
        assignments = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split_name, split_images in assignments.items():
            destination = output_dir / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for source in split_images:
                target = destination / source.name
                if link:
                    target.symlink_to(source.resolve())
                else:
                    shutil.copy2(source, target)

        summary[class_name] = {name: len(items) for name, items in assignments.items()}
        LOGGER.info(
            "%-16s total=%5d train=%5d val=%5d test=%5d",
            class_name,
            total,
            *(summary[class_name][name] for name in SPLIT_NAMES),
        )

    return summary
