from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .modules import CBAM, GeM


def build_resnet18_backbone() -> nn.Module:
    try:
        return models.resnet18(weights=None)
    except TypeError:
        return models.resnet18(pretrained=False)


class ResNet18GeMCBAM(nn.Module):
    """ResNet18 + CBAM + GeM + 2-class classification head for inference.

    SupCon projection head is omitted during inference.
    """

    def __init__(self, base_model: nn.Module, num_classes: int = 2):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        in_features = base_model.fc.in_features
        self.feature_dim = in_features
        self.cbam = CBAM(channels=in_features)
        self.avgpool = GeM()
        self.fc = nn.Linear(in_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.fc(features)
        return logits


def create_model(num_classes: int = 2) -> ResNet18GeMCBAM:
    base_model = build_resnet18_backbone()
    return ResNet18GeMCBAM(
        base_model=base_model,
        num_classes=num_classes,
    )
