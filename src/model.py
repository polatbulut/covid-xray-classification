import torch.nn as nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_transfer_model(name, num_classes, pretrained=True):
    name = name.lower()
    if name == 'resnet50':
        m = models.resnet50(pretrained=pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == 'densenet121':
        m = models.densenet121(pretrained=pretrained)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return m

