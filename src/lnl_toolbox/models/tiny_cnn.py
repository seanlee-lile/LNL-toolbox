from __future__ import annotations

"""A small CNN used for fast integration tests and CIFAR smoke runs."""

from torch import nn

from .feature_output import FeatureOutput


class TinyCNN(nn.Module):
    """A compact three-block CNN for end-to-end CIFAR validation."""

    def __init__(self, num_classes: int = 10, width: int = 64) -> None:
        super().__init__()
        channels = (width, width * 2, width * 4)
        layers: list[nn.Module] = []
        incoming = 3
        for outgoing in channels:
            layers.extend((
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.ReLU(inplace=True),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, inputs):
        return self.classifier(self._representation(inputs))

    def _representation(self, inputs):
        return self.pool(self.features(inputs)).flatten(1)

    def forward_with_features(self, inputs) -> FeatureOutput:
        features = self._representation(inputs)
        return FeatureOutput(self.classifier(features), features)
