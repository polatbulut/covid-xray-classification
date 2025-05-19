import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """
    A simple CNN baseline.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_transfer_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Load a pre-trained model and adjust the final layer.

    Supported names: 'resnet50', 'densenet121'
    """
    name = name.lower()
    if name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name '{name}'. Choose 'resnet50' or 'densenet121'.")
    return model
