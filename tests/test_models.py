from __future__ import annotations

import pytest
import torch

from covid_xray.models import SimpleCNN, count_parameters, create_model


def test_simple_cnn_output_shape() -> None:
    model = SimpleCNN(num_classes=4)
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 4)


def test_simple_cnn_is_resolution_agnostic() -> None:
    """The adaptive pooling layer means input size does not have to be fixed."""
    model = SimpleCNN(num_classes=3)
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 3)
    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 3)


def test_create_model_returns_the_baseline() -> None:
    assert isinstance(create_model("simple", num_classes=4), SimpleCNN)


@pytest.mark.parametrize("name", ["Simple", " SIMPLE "])
def test_model_names_are_normalised(name: str) -> None:
    assert isinstance(create_model(name, num_classes=2), SimpleCNN)


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        create_model("not_a_real_architecture", num_classes=4, pretrained=False)


def test_count_parameters() -> None:
    model = SimpleCNN(num_classes=4)
    total = count_parameters(model)
    assert total > 0
    for parameter in model.parameters():
        parameter.requires_grad = False
    assert count_parameters(model, trainable_only=True) == 0
    assert count_parameters(model, trainable_only=False) == total


@pytest.mark.slow
@pytest.mark.parametrize("name", ["resnet50", "densenet121"])
def test_torchvision_heads_are_resized(name: str) -> None:
    """The classification head must match the dataset, not ImageNet's 1000 classes."""
    model = create_model(name, num_classes=4, pretrained=False)
    logits = model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 4)


@pytest.mark.slow
def test_timm_backbone_is_constructed() -> None:
    model = create_model("efficientnet_b0", num_classes=4, pretrained=False)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 4)
