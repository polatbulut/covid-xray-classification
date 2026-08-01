"""Shared fixtures.

Every test builds its own synthetic image tree, so the suite runs without the
real 20 GB Kaggle dataset and without network access.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from covid_xray.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig

CLASS_NAMES: tuple[str, ...] = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")


def write_image_tree(
    root: Path,
    classes: Sequence[str] = CLASS_NAMES,
    images_per_class: int = 4,
    *,
    size: tuple[int, int] = (32, 32),
    nested_images_dir: bool = False,
    suffix: str = ".png",
) -> Path:
    """Create a ``<root>/<class>/[images/]<file>`` tree of solid-colour images.

    Args:
        root: Directory to create.
        classes: Class sub-directories to populate.
        images_per_class: Number of images written per class.
        size: Image dimensions in pixels.
        nested_images_dir: Mimic the raw Kaggle layout, which nests images one
            level deeper under ``images/``.
        suffix: File extension for the written images.

    Returns:
        The created ``root``.
    """
    for index, class_name in enumerate(classes):
        directory = root / class_name / "images" if nested_images_dir else root / class_name
        directory.mkdir(parents=True, exist_ok=True)
        shade = (index * 40) % 256
        for number in range(images_per_class):
            image = Image.new("RGB", size, color=(shade, shade, shade))
            image.save(directory / f"{class_name}-{number:03d}{suffix}")
    return root


@pytest.fixture
def image_root(tmp_path: Path) -> Path:
    """A small four-class image directory in the split layout."""
    return write_image_tree(tmp_path / "split")


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """A parent directory holding the raw dataset in its released layout."""
    raw = tmp_path / "raw"
    write_image_tree(
        raw / "COVID-19_Radiography_Dataset",
        images_per_class=20,
        nested_images_dir=True,
    )
    return raw


def make_experiment(
    root: Path,
    *,
    epochs: int = 2,
    patience: int = 0,
    images_per_class: int = 4,
) -> ExperimentConfig:
    """Build a runnable tiny experiment: three synthetic splits and the baseline CNN.

    Used by the training and evaluation integration tests, which exercise the
    full loop end to end on 16x16 images in a couple of seconds.
    """
    for split in ("train", "val", "test"):
        write_image_tree(root / "data" / split, images_per_class=images_per_class, size=(16, 16))
    return ExperimentConfig(
        data=DataConfig(
            train_dir=root / "data" / "train",
            val_dir=root / "data" / "val",
            test_dir=root / "data" / "test",
            img_size=(16, 16),
        ),
        model=ModelConfig(name="simple", pretrained=False),
        training=TrainingConfig(
            epochs=epochs,
            checkpoint_path=root / "models" / "run.pth",
            batch_size=4,
            num_workers=0,
            lr=1e-3,
            patience=patience,
            seed=0,
        ),
    )


@pytest.fixture
def config_mapping(tmp_path: Path) -> dict[str, object]:
    """A minimal valid experiment configuration as a plain mapping."""
    return {
        "data": {
            "train_dir": str(tmp_path / "train"),
            "val_dir": str(tmp_path / "val"),
            "test_dir": str(tmp_path / "test"),
        },
        "model": {"name": "simple"},
        "training": {"epochs": 2, "checkpoint_path": str(tmp_path / "models" / "run.pth")},
    }
