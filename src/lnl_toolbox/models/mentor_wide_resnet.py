from __future__ import annotations

"""PyTorch translation of MentorNet's official CIFAR Wide-ResNet student."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _WideResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        activate_before_residual: bool,
        leakiness: float,
    ) -> None:
        super().__init__()
        self.activate_before_residual = bool(activate_before_residual)
        self.leakiness = float(leakiness)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.activate_before_residual:
            activated = F.leaky_relu(inputs, negative_slope=self.leakiness)
            activated = self.bn1(activated)
            residual = activated
        else:
            residual = inputs
            activated = self.bn1(inputs)
            activated = F.leaky_relu(activated, negative_slope=self.leakiness)
        values = self.conv1(activated)
        values = self.bn2(values)
        values = F.leaky_relu(values, negative_slope=self.leakiness)
        values = self.conv2(values)

        if self.in_channels != self.out_channels:
            residual = F.avg_pool2d(
                residual, kernel_size=self.stride, stride=self.stride
            )
            difference = self.out_channels - self.in_channels
            if difference < 0 or difference % 2:
                raise ValueError("official Wide-ResNet channel projection is invalid")
            residual = F.pad(
                residual,
                (0, 0, 0, 0, difference // 2, difference // 2),
            )
        return values + residual


class MentorWideResNet101(nn.Module):
    """The official MentorNet CIFAR student: 9 units, filters 16/160/320/640."""

    def __init__(
        self,
        num_classes: int = 100,
        *,
        num_residual_units: int = 9,
        leakiness: float = 0.1,
        width_multiplier: float = 1.0,
        weight_decay: float = 0.0002,
    ) -> None:
        super().__init__()
        if num_classes < 2 or num_residual_units <= 0 or width_multiplier <= 0:
            raise ValueError("invalid MentorNet Wide-ResNet dimensions")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        base = [16, 160, 320, 640]
        filters = [max(1, int(round(value * width_multiplier))) for value in base]
        self.num_classes = int(num_classes)
        self.num_residual_units = int(num_residual_units)
        self.width_multiplier = float(width_multiplier)
        self.leakiness = float(leakiness)
        self.weight_decay = float(weight_decay)
        self.initial = nn.Conv2d(3, filters[0], 3, padding=1, bias=False)
        stages: list[nn.Module] = []
        for stage in range(3):
            in_channels, out_channels = filters[stage], filters[stage + 1]
            stride = 1 if stage == 0 else 2
            for unit in range(self.num_residual_units):
                stages.append(
                    _WideResidualUnit(
                        in_channels if unit == 0 else out_channels,
                        out_channels,
                        stride if unit == 0 else 1,
                        activate_before_residual=stage == 0 and unit == 0,
                        leakiness=self.leakiness,
                    )
                )
        self.stages = nn.Sequential(*stages)
        self.final_bn = nn.BatchNorm2d(filters[-1], eps=0.001, momentum=0.1)
        self.classifier = nn.Linear(filters[-1], num_classes)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                bound = math.sqrt(3.0 / module.in_features)
                nn.init.uniform_(module.weight, -bound, bound)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.stages(self.initial(inputs))
        values = self.final_bn(values)
        values = F.leaky_relu(values, negative_slope=self.leakiness)
        values = values.mean(dim=(2, 3))
        return self.classifier(values)

    def weighted_parameter_decay(self, sample_weights: torch.Tensor) -> torch.Tensor:
        """Match the official weighted convolution-kernel decay objective."""

        if sample_weights.ndim != 1 or sample_weights.numel() == 0:
            raise ValueError("sample_weights must be a non-empty vector")
        kernels = [
            module.weight for module in self.modules() if isinstance(module, nn.Conv2d)
        ]
        penalty = sum((weight.pow(2).sum() * 0.5 for weight in kernels))
        return penalty * self.weight_decay * sample_weights.mean()


__all__ = ["MentorWideResNet101"]
