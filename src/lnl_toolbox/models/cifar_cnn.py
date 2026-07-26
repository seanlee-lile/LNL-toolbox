from __future__ import annotations

"""The compact CIFAR CNN used by the Active-Passive Losses reference code."""

import torch
from torch import nn


class CifarCnn8(nn.Module):
    """Six-convolution CIFAR classifier matching the APL reference model."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        channels = (64, 64, 128, 128, 196, 196)
        layers: list[nn.Module] = []
        incoming = 3
        for index, outgoing in enumerate(channels):
            layers.extend((
                nn.Conv2d(incoming, outgoing, 3, padding=1),
                nn.BatchNorm2d(outgoing),
                nn.ReLU(inplace=True),
            ))
            if index in (1, 3, 5):
                layers.append(nn.MaxPool2d(2))
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 196, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))
