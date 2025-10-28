from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, pretrained: bool = False) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model name: {name}")
