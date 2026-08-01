from __future__ import annotations

from pathlib import Path

import pytest

from covid_xray.datasplit import (
    DEFAULT_CLASSES,
    SplitError,
    SplitRatios,
    resolve_raw_dir,
    split_dataset,
)

from .conftest import CLASS_NAMES, write_image_tree


def test_default_ratios_sum_to_one() -> None:
    ratios = SplitRatios()
    assert ratios.train + ratios.val + ratios.test == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("train", "val", "test"),
    [(0.8, 0.15, 0.15), (0.5, 0.2, 0.2), (0.0, 0.5, 0.5), (1.0, 0.0, 0.0)],
)
def test_invalid_ratios_are_rejected(train: float, val: float, test: float) -> None:
    with pytest.raises(ValueError):
        SplitRatios(train=train, val=val, test=test)


def test_resolve_raw_dir_finds_the_nested_dataset(raw_root: Path) -> None:
    """The Kaggle zip extracts into a directory whose name has changed over time."""
    assert resolve_raw_dir(raw_root) == raw_root / "COVID-19_Radiography_Dataset"


def test_resolve_raw_dir_accepts_the_dataset_itself(raw_root: Path) -> None:
    dataset = raw_root / "COVID-19_Radiography_Dataset"
    assert resolve_raw_dir(dataset) == dataset


def test_resolve_raw_dir_falls_back_to_scanning(tmp_path: Path) -> None:
    write_image_tree(tmp_path / "raw" / "some-unexpected-name", nested_images_dir=True)
    assert resolve_raw_dir(tmp_path / "raw").name == "some-unexpected-name"


def test_resolve_raw_dir_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SplitError, match="does not exist"):
        resolve_raw_dir(tmp_path / "absent")


def test_resolve_raw_dir_without_classes(tmp_path: Path) -> None:
    (tmp_path / "raw" / "unrelated").mkdir(parents=True)
    with pytest.raises(SplitError, match="Could not find"):
        resolve_raw_dir(tmp_path / "raw")


def test_split_counts(raw_root: Path, tmp_path: Path) -> None:
    summary = split_dataset(raw_root, tmp_path / "processed")
    for class_name in DEFAULT_CLASSES:
        assert summary[class_name] == {"train": 14, "val": 3, "test": 3}


def test_split_has_no_leakage(raw_root: Path, tmp_path: Path) -> None:
    """Every source image must land in exactly one split."""
    output = tmp_path / "processed"
    split_dataset(raw_root, output)
    owner: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            for image in (output / split / class_name).iterdir():
                assert image.name not in owner, f"{image.name} is in two splits"
                owner[image.name] = split
    assert len(owner) == 4 * 20


def test_split_is_deterministic_for_a_given_seed(raw_root: Path, tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    split_dataset(raw_root, first, seed=42)
    split_dataset(raw_root, second, seed=42)
    for class_name in CLASS_NAMES:
        assert sorted(p.name for p in (first / "train" / class_name).iterdir()) == sorted(
            p.name for p in (second / "train" / class_name).iterdir()
        )


def test_different_seeds_produce_different_splits(raw_root: Path, tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    split_dataset(raw_root, first, seed=42)
    split_dataset(raw_root, second, seed=7)
    assert any(
        sorted(p.name for p in (first / "train" / name).iterdir())
        != sorted(p.name for p in (second / "train" / name).iterdir())
        for name in CLASS_NAMES
    )


def test_split_refuses_to_clobber_existing_output(raw_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "processed"
    split_dataset(raw_root, output)
    with pytest.raises(SplitError, match="overwrite"):
        split_dataset(raw_root, output)


def test_overwrite_replaces_existing_output(raw_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "processed"
    split_dataset(raw_root, output)
    stale = output / "train" / "COVID" / "stale.png"
    stale.write_bytes(b"stale")
    split_dataset(raw_root, output, overwrite=True)
    assert not stale.exists()


def test_split_accepts_the_flat_layout(tmp_path: Path) -> None:
    """Older releases stored images directly under <class>/ rather than <class>/images/."""
    raw = tmp_path / "raw" / "covid19_radiography_dataset"
    write_image_tree(raw, images_per_class=10, nested_images_dir=False)
    summary = split_dataset(tmp_path / "raw", tmp_path / "processed", seed=1)
    assert summary["COVID"] == {"train": 7, "val": 1, "test": 2}


def test_link_mode_creates_symlinks(raw_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "processed"
    split_dataset(raw_root, output, link=True)
    links = list((output / "train" / "COVID").iterdir())
    assert links
    assert all(path.is_symlink() for path in links)
    assert all(path.resolve().is_file() for path in links)


def test_missing_class_directory_is_reported(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "COVID-19_Radiography_Dataset"
    write_image_tree(raw, classes=CLASS_NAMES[:3], nested_images_dir=True)
    with pytest.raises(SplitError, match="Could not find"):
        split_dataset(tmp_path / "raw", tmp_path / "processed")


def test_custom_ratios(raw_root: Path, tmp_path: Path) -> None:
    ratios = SplitRatios(train=0.5, val=0.25, test=0.25)
    summary = split_dataset(raw_root, tmp_path / "processed", ratios=ratios)
    assert summary["Normal"] == {"train": 10, "val": 5, "test": 5}
