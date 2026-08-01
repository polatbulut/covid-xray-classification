"""Typed configuration schema for training and evaluation runs.

Experiments are described by YAML files (see ``configs/``) which are parsed into
frozen dataclasses. Validation happens once, up front, so a typo in a key name
fails immediately with an actionable message instead of surfacing as a
``KeyError`` several epochs into a run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, cast

import yaml

IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

AnnealStrategy = Literal["cos", "linear"]
ClassWeighting = Literal["none", "inverse", "balanced"]
MonitorMetric = Literal["val_loss", "val_accuracy", "val_f1"]
MonitorMode = Literal["min", "max"]

ANNEAL_STRATEGIES: Final[tuple[str, ...]] = ("cos", "linear")
CLASS_WEIGHTINGS: Final[tuple[str, ...]] = ("none", "inverse", "balanced")
MONITOR_METRICS: Final[tuple[str, ...]] = ("val_loss", "val_accuracy", "val_f1")

_MONITOR_MODES: Final[Mapping[str, MonitorMode]] = {
    "val_loss": "min",
    "val_accuracy": "max",
    "val_f1": "max",
}


class ConfigError(ValueError):
    """Raised when a configuration file is missing keys or holds invalid values."""


def _reject_unknown(data: Mapping[str, Any], allowed: Iterable[str], section: str) -> None:
    """Fail when a section contains keys the schema does not define."""
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        expected = ", ".join(sorted(allowed))
        raise ConfigError(
            f"Unknown key(s) {unknown} in section '{section}'. Expected any of: {expected}."
        )


def _require(data: Mapping[str, Any], key: str, section: str) -> Any:
    """Return ``data[key]`` or fail with a message naming the section."""
    if key not in data:
        raise ConfigError(f"Missing required key '{key}' in section '{section}'.")
    return data[key]


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Coerce a YAML node to a mapping."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping, got {type(value).__name__}.")
    return cast(Mapping[str, Any], value)


def _as_bool(value: object, name: str) -> bool:
    """Coerce a YAML node to a boolean."""
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be true or false, got {value!r}.")
    return value


def _as_str(value: object, name: str) -> str:
    """Coerce a YAML node to a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty string, got {value!r}.")
    return value


def _as_int(value: object, name: str, *, minimum: int | None = None) -> int:
    """Coerce a YAML node to an integer, optionally enforcing a lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{name}' must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ConfigError(f"'{name}' must be >= {minimum}, got {value}.")
    return value


def _as_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce a YAML node to a float, optionally enforcing bounds."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"'{name}' must be a number, got {value!r}.")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ConfigError(f"'{name}' must be >= {minimum}, got {number}.")
    if maximum is not None and number > maximum:
        raise ConfigError(f"'{name}' must be <= {maximum}, got {number}.")
    return number


def _as_path(value: object, name: str) -> Path:
    """Coerce a YAML node to a filesystem path."""
    return Path(_as_str(value, name))


def _as_choice(value: object, options: tuple[str, ...], name: str) -> str:
    """Coerce a YAML node to one of a fixed set of strings."""
    if not isinstance(value, str) or value not in options:
        raise ConfigError(f"'{name}' must be one of {list(options)}, got {value!r}.")
    return value


def _as_image_size(value: object, name: str) -> tuple[int, int]:
    """Coerce a YAML node to a ``(height, width)`` pair."""
    if isinstance(value, int) and not isinstance(value, bool):
        side = _as_int(value, name, minimum=1)
        return (side, side)
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ConfigError(f"'{name}' must be a single int or [height, width], got {value!r}.")
    height = _as_int(value[0], f"{name}[0]", minimum=1)
    width = _as_int(value[1], f"{name}[1]", minimum=1)
    return (height, width)


def _as_channel_stats(value: object, name: str) -> tuple[float, float, float]:
    """Coerce a YAML node to a per-channel mean or standard deviation triplet."""
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ConfigError(f"'{name}' must be a list of three numbers, got {value!r}.")
    numbers = [_as_float(item, f"{name}[{index}]") for index, item in enumerate(value)]
    return (numbers[0], numbers[1], numbers[2])


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Train-time image augmentations.

    Every field defaults to "off", so omitting the ``data.augment`` section
    yields a deterministic resize-and-normalise pipeline.
    """

    random_horizontal_flip: bool = False
    random_rotation: float = 0.0
    brightness_jitter: float = 0.0
    contrast_jitter: float = 0.0

    _KEYS: ClassVar[tuple[str, ...]] = (
        "random_horizontal_flip",
        "random_rotation",
        "brightness_jitter",
        "contrast_jitter",
    )

    @property
    def enabled(self) -> bool:
        """Return ``True`` when at least one augmentation is active."""
        return bool(
            self.random_horizontal_flip
            or self.random_rotation > 0.0
            or self.brightness_jitter > 0.0
            or self.contrast_jitter > 0.0
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AugmentationConfig:
        """Build an augmentation config from a raw ``data.augment`` mapping."""
        section = "data.augment"
        _reject_unknown(data, cls._KEYS, section)
        return cls(
            random_horizontal_flip=_as_bool(
                data.get("random_horizontal_flip", False), f"{section}.random_horizontal_flip"
            ),
            random_rotation=_as_float(
                data.get("random_rotation", 0.0),
                f"{section}.random_rotation",
                minimum=0.0,
                maximum=180.0,
            ),
            brightness_jitter=_as_float(
                data.get("brightness_jitter", 0.0), f"{section}.brightness_jitter", minimum=0.0
            ),
            contrast_jitter=_as_float(
                data.get("contrast_jitter", 0.0), f"{section}.contrast_jitter", minimum=0.0
            ),
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the split image folders live and how images are preprocessed."""

    train_dir: Path
    val_dir: Path
    test_dir: Path
    img_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    augment: AugmentationConfig = field(default_factory=AugmentationConfig)

    _KEYS: ClassVar[tuple[str, ...]] = (
        "train_dir",
        "val_dir",
        "test_dir",
        "img_size",
        "mean",
        "std",
        "augment",
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DataConfig:
        """Build a data config from a raw ``data`` mapping."""
        section = "data"
        _reject_unknown(data, cls._KEYS, section)
        augment = data.get("augment")
        return cls(
            train_dir=_as_path(_require(data, "train_dir", section), f"{section}.train_dir"),
            val_dir=_as_path(_require(data, "val_dir", section), f"{section}.val_dir"),
            test_dir=_as_path(_require(data, "test_dir", section), f"{section}.test_dir"),
            img_size=_as_image_size(data.get("img_size", [224, 224]), f"{section}.img_size"),
            mean=_as_channel_stats(data.get("mean", list(IMAGENET_MEAN)), f"{section}.mean"),
            std=_as_channel_stats(data.get("std", list(IMAGENET_STD)), f"{section}.std"),
            augment=AugmentationConfig.from_mapping(
                _as_mapping(augment, f"{section}.augment") if augment is not None else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Which architecture to train.

    ``name`` is either ``"simple"`` for the baseline CNN, one of the explicitly
    supported torchvision backbones, or any ``timm`` model identifier.
    """

    name: str
    pretrained: bool = True

    _KEYS: ClassVar[tuple[str, ...]] = ("name", "pretrained")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ModelConfig:
        """Build a model config from a raw ``model`` mapping."""
        section = "model"
        _reject_unknown(data, cls._KEYS, section)
        return cls(
            name=_as_str(_require(data, "name", section), f"{section}.name"),
            pretrained=_as_bool(data.get("pretrained", True), f"{section}.pretrained"),
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimisation, scheduling, early stopping and checkpoint settings."""

    epochs: int
    checkpoint_path: Path
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 0
    monitor: MonitorMetric = "val_f1"
    class_weighting: ClassWeighting = "inverse"
    pct_start: float = 0.1
    anneal_strategy: AnnealStrategy = "cos"
    grad_clip: float | None = None
    seed: int = 42
    history_path: Path | None = None

    _KEYS: ClassVar[tuple[str, ...]] = (
        "epochs",
        "checkpoint_path",
        "batch_size",
        "num_workers",
        "lr",
        "weight_decay",
        "patience",
        "monitor",
        "class_weighting",
        "pct_start",
        "anneal_strategy",
        "grad_clip",
        "seed",
        "history_path",
    )

    @property
    def monitor_mode(self) -> MonitorMode:
        """Return whether the monitored metric should be minimised or maximised."""
        return _MONITOR_MODES[self.monitor]

    @property
    def resolved_history_path(self) -> Path:
        """Return where the per-epoch history is written, derived from the checkpoint path."""
        if self.history_path is not None:
            return self.history_path
        return self.checkpoint_path.with_name(f"{self.checkpoint_path.stem}_history.json")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrainingConfig:
        """Build a training config from a raw ``training`` mapping."""
        section = "training"
        _reject_unknown(data, cls._KEYS, section)
        grad_clip = data.get("grad_clip")
        history_path = data.get("history_path")
        return cls(
            epochs=_as_int(_require(data, "epochs", section), f"{section}.epochs", minimum=1),
            checkpoint_path=_as_path(
                _require(data, "checkpoint_path", section), f"{section}.checkpoint_path"
            ),
            batch_size=_as_int(data.get("batch_size", 32), f"{section}.batch_size", minimum=1),
            num_workers=_as_int(data.get("num_workers", 4), f"{section}.num_workers", minimum=0),
            lr=_as_float(data.get("lr", 1e-4), f"{section}.lr", minimum=0.0),
            weight_decay=_as_float(
                data.get("weight_decay", 1e-4), f"{section}.weight_decay", minimum=0.0
            ),
            patience=_as_int(data.get("patience", 0), f"{section}.patience", minimum=0),
            monitor=cast(
                MonitorMetric,
                _as_choice(data.get("monitor", "val_f1"), MONITOR_METRICS, f"{section}.monitor"),
            ),
            class_weighting=cast(
                ClassWeighting,
                _as_choice(
                    data.get("class_weighting", "inverse"),
                    CLASS_WEIGHTINGS,
                    f"{section}.class_weighting",
                ),
            ),
            pct_start=_as_float(
                data.get("pct_start", 0.1), f"{section}.pct_start", minimum=0.0, maximum=1.0
            ),
            anneal_strategy=cast(
                AnnealStrategy,
                _as_choice(
                    data.get("anneal_strategy", "cos"),
                    ANNEAL_STRATEGIES,
                    f"{section}.anneal_strategy",
                ),
            ),
            grad_clip=(
                None
                if grad_clip is None
                else _as_float(grad_clip, f"{section}.grad_clip", minimum=0.0)
            ),
            seed=_as_int(data.get("seed", 42), f"{section}.seed", minimum=0),
            history_path=(
                None
                if history_path is None
                else _as_path(history_path, f"{section}.history_path")
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """A complete experiment description: data, model and training settings."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    _KEYS: ClassVar[tuple[str, ...]] = ("data", "model", "training")

    @property
    def run_name(self) -> str:
        """Return a short identifier for the run, derived from the checkpoint filename."""
        return self.training.checkpoint_path.stem

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        """Build an experiment config from an already-parsed YAML document."""
        _reject_unknown(data, cls._KEYS, "<root>")
        return cls(
            data=DataConfig.from_mapping(_as_mapping(_require(data, "data", "<root>"), "data")),
            model=ModelConfig.from_mapping(_as_mapping(_require(data, "model", "<root>"), "model")),
            training=TrainingConfig.from_mapping(
                _as_mapping(_require(data, "training", "<root>"), "training")
            ),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load and validate an experiment config from a YAML file."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"Could not read configuration file {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc
        if raw is None:
            raise ConfigError(f"Configuration file is empty: {path}")
        return cls.from_mapping(_as_mapping(raw, str(path)))
