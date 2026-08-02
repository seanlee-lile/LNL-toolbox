from __future__ import annotations

"""Compact six-convolution CIFAR classifier shared by dual-network methods."""

import torch
from torch import nn
from torch.nn import functional as F


class CifarSixConvNet(nn.Module):
    """Six-convolution network used by JoCoR and related CIFAR baselines."""

    def __init__(
        self,
        num_classes: int = 10,
        *,
        input_channels: int = 3,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        channels = (64, 64, 128, 128, 196, 16)
        convolutions: list[nn.Module] = []
        incoming = int(input_channels)
        for outgoing in channels:
            convolutions.extend((
                nn.Conv2d(incoming, outgoing, kernel_size=3, padding=1),
                nn.BatchNorm2d(outgoing, momentum=float(batch_norm_momentum)),
                nn.ReLU(inplace=True),
            ))
            incoming = outgoing
        self.convolutions = nn.ModuleList(convolutions)
        self.classifier = nn.Linear(4 * 4 * channels[-1], int(num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs
        convolution_index = 0
        for block in range(3):
            for _ in range(2):
                value = self.convolutions[convolution_index](value)
                value = self.convolutions[convolution_index + 1](value)
                value = self.convolutions[convolution_index + 2](value)
                convolution_index += 3
            value = F.max_pool2d(value, kernel_size=2, stride=2)
        return self.classifier(value.flatten(1))


__all__ = ["CifarSixConvNet"]
