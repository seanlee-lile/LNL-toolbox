from __future__ import annotations

"""Official-shape SevenCNN used by the CIFAR CA2C workflow."""

import torch
from torch import nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, momentum=0.1),
        nn.ReLU(inplace=True),
    )


class CA2CSevenCNN(nn.Module):
    def __init__(self, num_classes: int = 100, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(input_channels, 64), _block(64, 64), nn.MaxPool2d(2),
            _block(64, 128), _block(128, 128), nn.MaxPool2d(2),
            _block(128, 196), _block(196, 16), nn.MaxPool2d(2),
        )
        self.projector = nn.Sequential(
            nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Linear(256, 128),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Linear(512, int(num_classes)),
        )

    def forward_with_features(self, inputs: torch.Tensor):
        flattened = self.features(inputs).flatten(1)
        return self.classifier(flattened), self.projector(flattened)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs)[0]


__all__ = ["CA2CSevenCNN"]
