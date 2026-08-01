from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from covid_xray.config import (
    AugmentationConfig,
    ConfigError,
    DataConfig,
    ExperimentConfig,
    TrainingConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_class_key_constants_are_not_dataclass_fields() -> None:
    """`_KEYS` must be a ClassVar; as a field it would join __init__ and __slots__."""
    for cls in (AugmentationConfig, DataConfig, TrainingConfig):
        assert "_KEYS" not in {field.name for field in dataclasses.fields(cls)}
    assert TrainingConfig._KEYS[0] == "epochs"


def test_defaults_are_applied(config_mapping: dict[str, Any]) -> None:
    config = ExperimentConfig.from_mapping(config_mapping)
    assert config.data.img_size == (224, 224)
    assert config.data.mean == (0.485, 0.456, 0.406)
    assert config.data.augment.enabled is False
    assert config.training.batch_size == 32
    assert config.training.monitor == "val_f1"
    assert config.training.monitor_mode == "max"
    assert config.training.class_weighting == "inverse"


def test_run_name_and_history_path_derive_from_checkpoint(config_mapping: dict[str, Any]) -> None:
    config = ExperimentConfig.from_mapping(config_mapping)
    assert config.run_name == "run"
    assert config.training.resolved_history_path.name == "run_history.json"


def test_explicit_history_path_wins(config_mapping: dict[str, Any]) -> None:
    training = dict(config_mapping["training"])
    training["history_path"] = "somewhere/else.json"
    config_mapping["training"] = training
    config = ExperimentConfig.from_mapping(config_mapping)
    assert config.training.resolved_history_path == Path("somewhere/else.json")


def test_monitor_mode_for_loss(config_mapping: dict[str, Any]) -> None:
    training = dict(config_mapping["training"])
    training["monitor"] = "val_loss"
    config_mapping["training"] = training
    assert ExperimentConfig.from_mapping(config_mapping).training.monitor_mode == "min"


def test_augment_section_is_parsed(config_mapping: dict[str, Any]) -> None:
    data = dict(config_mapping["data"])
    data["augment"] = {"random_horizontal_flip": True, "random_rotation": 15}
    config_mapping["data"] = data
    augment = ExperimentConfig.from_mapping(config_mapping).data.augment
    assert augment.enabled is True
    assert augment.random_horizontal_flip is True
    assert augment.random_rotation == pytest.approx(15.0)
    assert augment.brightness_jitter == pytest.approx(0.0)


def test_missing_required_key_is_reported(config_mapping: dict[str, Any]) -> None:
    config_mapping["training"] = {"epochs": 1}
    with pytest.raises(ConfigError, match="checkpoint_path"):
        ExperimentConfig.from_mapping(config_mapping)


def test_legacy_model_type_key_is_rejected(config_mapping: dict[str, Any]) -> None:
    """The pre-refactor schema used `model.type`; it must not silently pass."""
    config_mapping["model"] = {"type": "resnet50"}
    with pytest.raises(ConfigError, match="type"):
        ExperimentConfig.from_mapping(config_mapping)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("epochs", 0, ">= 1"),
        ("batch_size", 0, ">= 1"),
        ("num_workers", -1, ">= 0"),
        ("lr", -0.1, ">= 0"),
        ("patience", -1, ">= 0"),
        ("pct_start", 1.5, "<= 1"),
        ("monitor", "val_auc", "monitor"),
        ("class_weighting", "sqrt", "class_weighting"),
        ("anneal_strategy", "exp", "anneal_strategy"),
        ("epochs", "many", "integer"),
    ],
)
def test_invalid_training_values_are_rejected(
    config_mapping: dict[str, Any], key: str, value: object, message: str
) -> None:
    training = dict(config_mapping["training"])
    training[key] = value
    config_mapping["training"] = training
    with pytest.raises(ConfigError, match=message):
        ExperimentConfig.from_mapping(config_mapping)


@pytest.mark.parametrize("value", [[224], [224, 224, 3], "224", [0, 224]])
def test_invalid_image_sizes_are_rejected(config_mapping: dict[str, Any], value: object) -> None:
    data = dict(config_mapping["data"])
    data["img_size"] = value
    config_mapping["data"] = data
    with pytest.raises(ConfigError, match="img_size"):
        ExperimentConfig.from_mapping(config_mapping)


def test_scalar_image_size_is_expanded(config_mapping: dict[str, Any]) -> None:
    data = dict(config_mapping["data"])
    data["img_size"] = 256
    config_mapping["data"] = data
    assert ExperimentConfig.from_mapping(config_mapping).data.img_size == (256, 256)


def test_booleans_are_not_accepted_as_numbers(config_mapping: dict[str, Any]) -> None:
    training = dict(config_mapping["training"])
    training["epochs"] = True
    config_mapping["training"] = training
    with pytest.raises(ConfigError, match="integer"):
        ExperimentConfig.from_mapping(config_mapping)


def test_from_yaml_round_trip(tmp_path: Path, config_mapping: dict[str, Any]) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config_mapping), encoding="utf-8")
    assert ExperimentConfig.from_yaml(path) == ExperimentConfig.from_mapping(config_mapping)


def test_from_yaml_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read"):
        ExperimentConfig.from_yaml(tmp_path / "nope.yaml")


def test_from_yaml_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        ExperimentConfig.from_yaml(path)


def test_from_yaml_malformed(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="parse YAML"):
        ExperimentConfig.from_yaml(path)


@pytest.mark.parametrize(
    "config_path", sorted((REPO_ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name
)
def test_shipped_configs_are_valid(config_path: Path) -> None:
    """Every config in `configs/` must load, so none can rot unnoticed."""
    config = ExperimentConfig.from_yaml(config_path)
    assert config.training.epochs > 0
    assert config.model.name
    assert config.training.checkpoint_path.suffix == ".pth"


def test_config_is_immutable(config_mapping: dict[str, Any]) -> None:
    config = ExperimentConfig.from_mapping(config_mapping)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.training.epochs = 99  # type: ignore[misc]


def test_dataclasses_replace_supports_overrides(config_mapping: dict[str, Any]) -> None:
    """The CLI's --epochs/--seed overrides rely on `dataclasses.replace`."""
    config = ExperimentConfig.from_mapping(config_mapping)
    updated = dataclasses.replace(config, training=dataclasses.replace(config.training, epochs=7))
    assert updated.training.epochs == 7
    assert config.training.epochs == 2
