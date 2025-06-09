import torch.nn as nn
import torchvision.models as models
import timm


class SimpleCNN(nn.Module):
    """
    Small baseline CNN
    """
    def __init__(self, num_classes):
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
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_transfer_model(name, num_classes, pretrained=True):
    """
    Supported:
      - 'resnet50'
      - 'densenet121'
      - any timm model name like 'efficientnet_b3', 'resnet34', etc.
    """
    name = name.lower()
    if name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        try:
            model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        except Exception as e:
            raise ValueError(f"Unsupported model '{name}'. Must be resnet50, densenet121, or a timm name.") from e

    return model
