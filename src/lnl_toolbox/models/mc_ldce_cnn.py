from __future__ import annotations

"""CIFAR classifier used by the MC-LDCE experiments."""

import torch
from torch import nn
from torch.nn import functional as F


class MCLDCECifarCNN(nn.Module):
    """Paper six-convolution network with an explicit feature interface."""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        *,
        classifier_bias: bool = True,
    ) -> None:
        super().__init__()
        self._feature_extractor_frozen = False
        channels = (128, 128, 128, 512, 256, 128)
        layers: list[nn.Module] = []
        incoming = int(input_channels)
        for outgoing in channels:
            layers.extend((
                nn.Conv2d(incoming, outgoing, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            ))
            incoming = outgoing
        self.convolutions = nn.ModuleList(layers)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(
            128,
            int(num_classes),
            bias=bool(classifier_bias),
        )

    def freeze_feature_extractor(self) -> None:
        """Keep the representation fixed for the centroid-risk stage."""

        self._feature_extractor_frozen = True
        for parameter in self.convolutions.parameters():
            parameter.requires_grad_(False)
        self.dropout.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._feature_extractor_frozen:
            self.dropout.eval()
        return self

    def forward_with_features(self, inputs: torch.Tensor):
        value = inputs
        for index in range(0, 6, 2):
            value = self.convolutions[index](value)
            value = self.convolutions[index + 1](value)
        value = self.dropout(F.max_pool2d(value, 2, 2))
        for index in range(6, 12, 2):
            value = self.convolutions[index](value)
            value = self.convolutions[index + 1](value)
        features = F.adaptive_avg_pool2d(value, 1).flatten(1)
        return self.classifier(features), features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs)[0]


__all__ = ["MCLDCECifarCNN"]
