from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .feature_output import FeatureOutput


class BasicBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.shortcut = nn.Identity() if stride == 1 and incoming == outgoing else nn.Sequential(
            nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False), nn.BatchNorm2d(outgoing)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        output = self.bn2(self.conv2(output))
        return F.relu(output + self.shortcut(inputs), inplace=True)


class PreActBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.shortcut = None if stride == 1 and incoming == outgoing else nn.Conv2d(
            incoming, outgoing, 1, stride=stride, bias=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activated = F.relu(self.bn1(inputs), inplace=True)
        shortcut = inputs if self.shortcut is None else self.shortcut(activated)
        output = self.conv1(activated)
        output = self.conv2(F.relu(self.bn2(output), inplace=True))
        return output + shortcut


class BottleneckBlock(nn.Module):
    """Three-convolution bottleneck used by the explicit CIFAR ResNet-50."""

    expansion = 4

    def __init__(self, incoming: int, outgoing: int, stride: int = 1) -> None:
        super().__init__()
        expanded = outgoing * self.expansion
        self.conv1 = nn.Conv2d(incoming, outgoing, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(
            outgoing,
            outgoing,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.conv3 = nn.Conv2d(outgoing, expanded, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(expanded)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and incoming == expanded
            else nn.Sequential(
                nn.Conv2d(
                    incoming,
                    expanded,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        output = F.relu(self.bn2(self.conv2(output)), inplace=True)
        output = self.bn3(self.conv3(output))
        return F.relu(output + self.shortcut(inputs), inplace=True)


class CifarResNetBottleneck(nn.Module):
    """Four-stage CIFAR bottleneck network used only when explicitly selected."""

    def __init__(
        self,
        num_classes: int = 10,
        base_width: int = 64,
        layer_counts: tuple[int, int, int, int] = (3, 4, 6, 3),
        *,
        stem_padding: int = 1,
        initialization: str = "kaiming",
    ) -> None:
        super().__init__()
        if len(layer_counts) != 4 or any(count <= 0 for count in layer_counts):
            raise ValueError("layer_counts must contain four positive integers")
        if stem_padding < 0:
            raise ValueError("stem_padding must be non-negative")
        initialization = str(initialization).strip().lower()
        if initialization not in {"kaiming", "torch_default"}:
            raise ValueError(
                "initialization must be 'kaiming' or 'torch_default'"
            )
        self.incoming = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                base_width,
                3,
                padding=stem_padding,
                bias=False,
            ),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_width, layer_counts[0], 1)
        self.layer2 = self._make_layer(base_width * 2, layer_counts[1], 2)
        self.layer3 = self._make_layer(base_width * 4, layer_counts[2], 2)
        self.layer4 = self._make_layer(base_width * 8, layer_counts[3], 2)
        feature_width = base_width * 8 * BottleneckBlock.expansion
        self.classifier = nn.Linear(feature_width, num_classes)
        if initialization == "kaiming":
            self._initialize()

    def _make_layer(
        self,
        outgoing: int,
        count: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [BottleneckBlock(self.incoming, outgoing, stride)]
        self.incoming = outgoing * BottleneckBlock.expansion
        for _ in range(count - 1):
            layers.append(BottleneckBlock(self.incoming, outgoing))
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _representation(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layer4(
            self.layer3(
                self.layer2(
                    self.layer1(self.stem(inputs))
                )
            )
        )
        return F.adaptive_avg_pool2d(output, 1).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._representation(inputs))

    def forward_with_features(
        self,
        inputs: torch.Tensor,
    ) -> FeatureOutput:
        features = self._representation(inputs)
        return FeatureOutput(self.classifier(features), features)


class CifarResNet(nn.Module):
    """Residual network adapted to 32x32 CIFAR images."""

    def __init__(self, block: type[nn.Module], num_classes: int = 10,
                 base_width: int = 64, preactivation: bool = False,
                 layer_counts: tuple[int, int, int, int] = (2, 2, 2, 2)) -> None:
        super().__init__()
        if len(layer_counts) != 4 or any(count <= 0 for count in layer_counts):
            raise ValueError("layer_counts must contain four positive integers")
        self.incoming = base_width
        stem: nn.Module = nn.Conv2d(3, base_width, 3, padding=1, bias=False)
        if not preactivation:
            stem = nn.Sequential(stem, nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.stem = stem
        self.layer1 = self._make_layer(block, base_width, layer_counts[0], 1)
        self.layer2 = self._make_layer(block, base_width * 2, layer_counts[1], 2)
        self.layer3 = self._make_layer(block, base_width * 4, layer_counts[2], 2)
        self.layer4 = self._make_layer(block, base_width * 8, layer_counts[3], 2)
        self.final_bn = nn.BatchNorm2d(base_width * 8) if preactivation else nn.Identity()
        self.preactivation = preactivation
        self.classifier = nn.Linear(base_width * 8, num_classes)
        self._initialize()

    def _make_layer(self, block: type[nn.Module], outgoing: int, count: int, stride: int) -> nn.Sequential:
        layers = []
        for current_stride in [stride] + [1] * (count - 1):
            layers.append(block(self.incoming, outgoing, current_stride))
            self.incoming = outgoing
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._representation(inputs))

    def _representation(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(inputs)))))
        if self.preactivation:
            output = F.relu(self.final_bn(output), inplace=True)
        return F.adaptive_avg_pool2d(output, 1).flatten(1)

    def forward_with_features(
        self,
        inputs: torch.Tensor,
    ) -> FeatureOutput:
        features = self._representation(inputs)
        return FeatureOutput(self.classifier(features), features)


def cifar_resnet18(num_classes: int = 10, base_width: int = 64) -> CifarResNet:
    return CifarResNet(BasicBlock, num_classes, base_width, preactivation=False)


def cifar_resnet34(num_classes: int = 10, base_width: int = 64) -> CifarResNet:
    """Return the [3, 4, 6, 3] ResNet-34 used by the GCE CIFAR experiments."""

    return CifarResNet(
        BasicBlock,
        num_classes,
        base_width,
        preactivation=False,
        layer_counts=(3, 4, 6, 3),
    )


def cifar_resnet50(
    num_classes: int = 10,
    base_width: int = 64,
    *,
    stem_padding: int = 1,
    initialization: str = "kaiming",
) -> CifarResNetBottleneck:
    """Return ResNet-50 without changing existing model defaults."""

    return CifarResNetBottleneck(
        num_classes,
        base_width,
        stem_padding=stem_padding,
        initialization=initialization,
    )


def cifar_resnet101(
    num_classes: int = 10,
    base_width: int = 64,
    *,
    stem_padding: int = 1,
    initialization: str = "kaiming",
) -> CifarResNetBottleneck:
    """Return a reusable CIFAR ResNet-101 bottleneck StudentNet."""

    return CifarResNetBottleneck(
        num_classes,
        base_width,
        layer_counts=(3, 4, 23, 3),
        stem_padding=stem_padding,
        initialization=initialization,
    )


def preact_resnet18(num_classes: int = 10, base_width: int = 64) -> CifarResNet:
    return CifarResNet(PreActBlock, num_classes, base_width, preactivation=True)


class CifarResNetDepth(nn.Module):
    """The standard three-stage CIFAR depth family (depth = 6n + 2)."""

    def __init__(self, depth: int, num_classes: int = 10, base_width: int = 16) -> None:
        super().__init__()
        if depth < 8 or (depth - 2) % 6:
            raise ValueError("CIFAR depth must satisfy depth = 6*n + 2 and be at least 8")
        blocks = (depth - 2) // 6
        widths = (base_width, base_width * 2, base_width * 4)
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(widths[0])
        incoming = widths[0]
        stages = []
        for stage, width in enumerate(widths):
            layers = []
            for block in range(blocks):
                stride = 1 if stage == 0 or block else 2
                layers.append(BasicBlock(incoming, width, stride))
                incoming = width
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.classifier = nn.Linear(widths[-1], num_classes)
        self.depth = int(depth)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs).logits

    def forward_with_features(self, inputs: torch.Tensor) -> FeatureOutput:
        output = F.relu(self.bn(self.stem(inputs)), inplace=True)
        for stage in self.stages:
            output = stage(output)
        features = F.adaptive_avg_pool2d(output, 1).flatten(1)
        return FeatureOutput(self.classifier(features), features)


def cifar_resnet_depth(depth: int, num_classes: int = 10, base_width: int = 16) -> CifarResNetDepth:
    return CifarResNetDepth(depth, num_classes, base_width)


def cifar_resnet14(num_classes: int = 10, base_width: int = 16) -> CifarResNetDepth:
    return cifar_resnet_depth(14, num_classes, base_width)


def cifar_resnet32(num_classes: int = 10, base_width: int = 16) -> CifarResNetDepth:
    return cifar_resnet_depth(32, num_classes, base_width)
