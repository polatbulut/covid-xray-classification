from __future__ import annotations

from pathlib import Path

import pytest
import torch

from covid_xray.config import AugmentationConfig, DataConfig
from covid_xray.data import (
    DatasetError,
    XRayDataset,
    build_eval_transform,
    build_train_transform,
)

from .conftest import CLASS_NAMES, write_image_tree


def make_data_config(root: Path, augment: AugmentationConfig | None = None) -> DataConfig:
    return DataConfig(
        train_dir=root,
        val_dir=root,
        test_dir=root,
        img_size=(16, 16),
        augment=augment or AugmentationConfig(),
    )


def test_class_discovery_is_sorted(image_root: Path) -> None:
    dataset = XRayDataset(image_root)
    assert dataset.classes == tuple(sorted(CLASS_NAMES))
    assert dataset.class_to_idx == {name: i for i, name in enumerate(dataset.classes)}


def test_sample_count_and_class_counts(image_root: Path) -> None:
    dataset = XRayDataset(image_root)
    assert len(dataset) == 4 * len(CLASS_NAMES)
    assert dataset.class_counts() == (4, 4, 4, 4)
    assert len(dataset.targets) == len(dataset)


def test_sample_order_is_deterministic(image_root: Path) -> None:
    """Sample order feeds the misclassified grid, so it must not depend on the OS."""
    first = XRayDataset(image_root).samples
    second = XRayDataset(image_root).samples
    assert first == second
    assert list(first) == sorted(first)


def test_explicit_class_order_is_honoured(image_root: Path) -> None:
    """Evaluation pins the checkpoint's class order onto the test split."""
    reversed_classes = tuple(sorted(CLASS_NAMES, reverse=True))
    dataset = XRayDataset(image_root, classes=reversed_classes)
    assert dataset.classes == reversed_classes
    assert dataset.class_to_idx[reversed_classes[0]] == 0


def test_unknown_requested_class_is_rejected(image_root: Path) -> None:
    with pytest.raises(DatasetError, match="Pneumothorax"):
        XRayDataset(image_root, classes=["COVID", "Pneumothorax"])


def test_missing_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="does not exist"):
        XRayDataset(tmp_path / "absent")


def test_directory_without_classes_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetError, match="No class sub-directories"):
        XRayDataset(empty)


def test_directory_without_images_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "COVID").mkdir(parents=True)
    (root / "COVID" / "notes.txt").write_text("not an image", encoding="utf-8")
    with pytest.raises(DatasetError, match="No images"):
        XRayDataset(root)


def test_non_image_files_are_ignored(image_root: Path) -> None:
    (image_root / "COVID" / "README.md").write_text("ignore me", encoding="utf-8")
    (image_root / "COVID" / "thumbs.db").write_bytes(b"\x00")
    assert len(XRayDataset(image_root)) == 4 * len(CLASS_NAMES)


def test_uppercase_extensions_are_accepted(tmp_path: Path) -> None:
    root = write_image_tree(tmp_path / "upper", classes=("COVID", "Normal"), suffix=".PNG")
    assert len(XRayDataset(root)) == 8


def test_getitem_returns_a_normalised_tensor(image_root: Path) -> None:
    config = make_data_config(image_root)
    dataset = XRayDataset(image_root, build_eval_transform(config))
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 16, 16)
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    assert label == 0


def test_default_transform_yields_a_tensor(image_root: Path) -> None:
    image, _ = XRayDataset(image_root)[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3


def test_eval_transform_has_no_random_steps(image_root: Path) -> None:
    config = make_data_config(image_root)
    dataset = XRayDataset(image_root, build_eval_transform(config))
    torch.manual_seed(0)
    first, _ = dataset[0]
    torch.manual_seed(1)
    second, _ = dataset[0]
    assert torch.equal(first, second)


def test_train_transform_without_augmentation_matches_eval(image_root: Path) -> None:
    config = make_data_config(image_root)
    assert len(build_train_transform(config).transforms) == len(
        build_eval_transform(config).transforms
    )


def test_train_transform_adds_configured_augmentations(image_root: Path) -> None:
    config = make_data_config(
        image_root,
        AugmentationConfig(
            random_horizontal_flip=True,
            random_rotation=15,
            brightness_jitter=0.2,
            contrast_jitter=0.2,
        ),
    )
    names = [type(step).__name__ for step in build_train_transform(config).transforms]
    assert names == [
        "Resize",
        "RandomHorizontalFlip",
        "RandomRotation",
        "ColorJitter",
        "ToTensor",
        "Normalize",
    ]


def test_augmentation_flags_are_independent(image_root: Path) -> None:
    config = make_data_config(image_root, AugmentationConfig(random_rotation=10))
    names = [type(step).__name__ for step in build_train_transform(config).transforms]
    assert "RandomRotation" in names
    assert "RandomHorizontalFlip" not in names
    assert "ColorJitter" not in names
