from __future__ import annotations

"""Seven-layer CIFAR classifier used by the standalone SED+FINE runner."""

import torch
from torch import nn
from torch.nn import functional as F

from .feature_output import FeatureOutput


class FineSevenCNN(nn.Module):
    """Compact seven-convolution network with a feature-output contract.

    The architecture is kept independent from FINE's training lifecycle so it
    can be reused by any CIFAR method requiring the same student network.
    """

    def __init__(
        self,
        num_classes: int = 100,
        *,
        base_width: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if num_classes < 2 or base_width <= 0:
            raise ValueError("num_classes and base_width must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        channels = (
            base_width,
            base_width,
            base_width,
            base_width * 2,
            base_width * 2,
            base_width * 2,
            base_width * 4,
        )
        blocks: list[nn.Module] = []
        incoming = 3
        for index, outgoing in enumerate(channels):
            blocks.extend(
                (
                    nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                    nn.BatchNorm2d(outgoing),
                    nn.LeakyReLU(0.01, inplace=True),
                )
            )
            if index in {2, 5}:
                blocks.extend((nn.MaxPool2d(2), nn.Dropout2d(dropout)))
            incoming = outgoing
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Linear(channels[-1], num_classes)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_with_features(self, inputs: torch.Tensor) -> FeatureOutput:
        values = self.features(inputs)
        features = F.adaptive_avg_pool2d(values, 1).flatten(1)
        return FeatureOutput(self.classifier(features), features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs).logits


__all__ = ["FineSevenCNN"]
