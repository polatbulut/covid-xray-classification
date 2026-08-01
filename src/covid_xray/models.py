"""Model definitions and the factory shared by training and evaluation.

``create_model`` accepts the baseline CNN name, two explicitly supported
torchvision backbones, or any ``timm`` model identifier. Both entry points build
their model through this single function so a checkpoint can always be paired
with the architecture that produced it.
"""

from __future__ import annotations

import logging
from typing import Any, Final, cast

import timm
import torch
from torch import nn
from torchvision import models

LOGGER: Final = logging.getLogger(__name__)

SIMPLE_MODEL_NAME: Final = "simple"
TORCHVISION_MODELS: Final[tuple[str, ...]] = ("resnet50", "densenet121")


class SimpleCNN(nn.Module):
    """A small three-block convolutional baseline.

    Useful as a sanity check and as a lower bound on the transfer-learning
    backbones: it trains in minutes and needs no pretrained weights.

    Args:
        num_classes: Number of output logits.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of images."""
        features: torch.Tensor = self.features(inputs)
        flattened = torch.flatten(features, start_dim=1)
        logits: torch.Tensor = self.classifier(flattened)
        return logits


def _build_torchvision_model(name: str, num_classes: int, *, pretrained: bool) -> nn.Module:
    """Build a torchvision backbone with its classification head replaced."""
    weights: Any
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model: Any = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return cast(nn.Module, model)

    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return cast(nn.Module, model)


def create_model(name: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Build a model by name.

    Args:
        name: ``"simple"``, a supported torchvision backbone
            (``resnet50``, ``densenet121``), or any ``timm`` model identifier
            such as ``efficientnet_b3``.
        num_classes: Number of output logits.
        pretrained: Whether to load ImageNet weights. Ignored for ``"simple"``,
            which has no pretrained checkpoint.

    Raises:
        ValueError: If ``name`` is not recognised by torchvision or ``timm``.
    """
    key = name.strip().lower()

    if key == SIMPLE_MODEL_NAME:
        if pretrained:
            LOGGER.debug("The '%s' baseline has no pretrained weights; training from scratch.", key)
        return SimpleCNN(num_classes)

    if key in TORCHVISION_MODELS:
        return _build_torchvision_model(key, num_classes, pretrained=pretrained)

    try:
        model: Any = timm.create_model(key, pretrained=pretrained, num_classes=num_classes)
    except (RuntimeError, ValueError) as exc:
        supported = ", ".join((SIMPLE_MODEL_NAME, *TORCHVISION_MODELS))
        raise ValueError(
            f"Unsupported model {name!r}. Expected one of ({supported}) or a valid timm "
            f"model name; timm reported: {exc}"
        ) from exc
    return cast(nn.Module, model)


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Return the number of parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
