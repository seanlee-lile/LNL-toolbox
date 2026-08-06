from __future__ import annotations

import math
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
                 layer_counts: tuple[int, int, int, int] = (2, 2, 2, 2),
                 initialization: str = "kaiming",
                 classifier_bias: bool = True,
                 stem_padding: int = 1) -> None:
        super().__init__()
        if len(layer_counts) != 4 or any(count <= 0 for count in layer_counts):
            raise ValueError("layer_counts must contain four positive integers")
        initialization = str(initialization).strip().lower()
        if initialization not in {"kaiming", "torch_default"}:
            raise ValueError("initialization must be 'kaiming' or 'torch_default'")
        if int(stem_padding) < 0:
            raise ValueError("stem_padding must be non-negative")
        self.incoming = base_width
        stem: nn.Module = nn.Conv2d(
            3, base_width, 3, padding=int(stem_padding), bias=False
        )
        if not preactivation:
            stem = nn.Sequential(stem, nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.stem = stem
        self.layer1 = self._make_layer(block, base_width, layer_counts[0], 1)
        self.layer2 = self._make_layer(block, base_width * 2, layer_counts[1], 2)
        self.layer3 = self._make_layer(block, base_width * 4, layer_counts[2], 2)
        self.layer4 = self._make_layer(block, base_width * 8, layer_counts[3], 2)
        self.final_bn = nn.BatchNorm2d(base_width * 8) if preactivation else nn.Identity()
        self.preactivation = preactivation
        self.classifier = nn.Linear(
            base_width * 8,
            num_classes,
            bias=bool(classifier_bias),
        )
        if initialization == "kaiming":
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


def cifar_resnet18(
    num_classes: int = 10,
    base_width: int = 64,
    *,
    initialization: str = "kaiming",
) -> CifarResNet:
    return CifarResNet(
        BasicBlock,
        num_classes,
        base_width,
        preactivation=False,
        initialization=initialization,
    )


def cifar_resnet34(
    num_classes: int = 10,
    base_width: int = 64,
    *,
    initialization: str = "kaiming",
    bias: bool = True,
    stem_padding: int = 1,
) -> CifarResNet:
    """Return the [3, 4, 6, 3] ResNet-34 used by the GCE CIFAR experiments."""

    return CifarResNet(
        BasicBlock,
        num_classes,
        base_width,
        preactivation=False,
        layer_counts=(3, 4, 6, 3),
        initialization=initialization,
        classifier_bias=bias,
        stem_padding=stem_padding,
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


class _L2RWBatchNorm2d(nn.Module):
    """Batch-statistics BN used by Uber's assigned-weight meta replicas."""

    def __init__(self, num_features: int, epsilon: float = 0.001) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(int(num_features)))
        self.epsilon = float(epsilon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=(0, 2, 3), keepdim=True)
        variance = inputs.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        bias = self.bias.view(1, -1, 1, 1)
        return (inputs - mean) * torch.rsqrt(variance + self.epsilon) + bias


class _L2RWResidualUnit(nn.Module):
    """One residual unit from Uber's CIFAR ResNet module."""

    def __init__(
        self,
        incoming: int,
        outgoing: int,
        stride: int,
        *,
        activate_input: bool,
        meta_batch_statistics: bool = False,
    ) -> None:
        super().__init__()
        batch_norm = _L2RWBatchNorm2d if meta_batch_statistics else nn.BatchNorm2d
        self.bn1 = batch_norm(incoming) if activate_input else None
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.bn2 = batch_norm(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.incoming = int(incoming)
        self.outgoing = int(outgoing)
        self.stride = int(stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs if self.bn1 is None else F.relu(self.bn1(inputs), inplace=True)
        value = self.conv1(value)
        value = self.conv2(F.relu(self.bn2(value), inplace=True))
        shortcut = inputs
        if self.stride > 1:
            shortcut = F.avg_pool2d(shortcut, self.stride, self.stride, ceil_mode=False)
        if self.incoming < self.outgoing:
            difference = self.outgoing - self.incoming
            if difference % 2:
                raise ValueError("official L2RW channel padding requires an even difference")
            side = difference // 2
            shortcut = F.pad(shortcut, (0, 0, 0, 0, side, side))
        return value + shortcut


class L2RWResNet32(nn.Module):
    """Exact CIFAR ResNet-32 topology used by the official L2RW config."""

    def __init__(
        self,
        num_classes: int = 10,
        base_width: int = 16,
        *,
        meta_batch_statistics: bool = False,
    ) -> None:
        super().__init__()
        self.meta_batch_statistics = bool(meta_batch_statistics)
        batch_norm = _L2RWBatchNorm2d if self.meta_batch_statistics else nn.BatchNorm2d
        widths = (int(base_width), int(base_width) * 2, int(base_width) * 4)
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.stem_bn = batch_norm(widths[0])
        stages: list[nn.Module] = []
        incoming = widths[0]
        for stage, outgoing in enumerate(widths):
            for unit in range(5):
                stride = 1 if stage == 0 or unit else 2
                stages.append(_L2RWResidualUnit(
                    incoming, outgoing, stride,
                    activate_input=not (stage == 0 and unit == 0),
                    meta_batch_statistics=self.meta_batch_statistics,
                ))
                incoming = outgoing
        self.stages = nn.ModuleList(stages)
        self.final_bn = batch_norm(widths[-1])
        self.classifier = nn.Linear(widths[-1], int(num_classes))
        self._initialize_official()

    def _initialize_official(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                std = math.sqrt(2.0 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels))
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        limit = math.sqrt(3.0 / self.classifier.in_features)
        nn.init.uniform_(self.classifier.weight, -limit, limit)
        nn.init.zeros_(self.classifier.bias)

    def forward_with_features(self, inputs: torch.Tensor) -> FeatureOutput:
        value = F.relu(self.stem_bn(self.stem(inputs)), inplace=True)
        for unit in self.stages:
            value = unit(value)
        value = F.relu(self.final_bn(value), inplace=True)
        features = F.adaptive_avg_pool2d(value, 1).flatten(1)
        return FeatureOutput(self.classifier(features), features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs).logits


def share_l2rw_meta_parameters(
    weighted_model: L2RWResNet32,
    meta_model: L2RWResNet32,
) -> None:
    """Tie the official assigned-weight branch to the weighted model.

    Uber's TensorFlow graph builds the assigned-weight A/B branches by
    reusing the current variables of model C.  Only the BatchNorm behavior
    differs: the assigned branch computes batch statistics and exposes beta,
    while model C keeps moving-statistics BatchNorm.  The PyTorch replica
    therefore shares every parameter that exists in both models instead of
    maintaining a second, independent set of meta parameters.
    """

    if not isinstance(weighted_model, L2RWResNet32) or not isinstance(
        meta_model, L2RWResNet32
    ):
        raise TypeError("L2RW parameter sharing requires two L2RWResNet32 models")
    if not meta_model.meta_batch_statistics:
        raise ValueError("meta_model must use batch-statistics BatchNorm")
    if weighted_model.meta_batch_statistics:
        raise ValueError("weighted_model must use moving-statistics BatchNorm")

    weighted_parameters = dict(weighted_model.named_parameters())

    def replace_parameter(module: nn.Module, name: str, parameter: nn.Parameter) -> None:
        parent = module
        components = name.split(".")
        for component in components[:-1]:
            parent = getattr(parent, component)
        setattr(parent, components[-1], parameter)

    for name, meta_parameter in list(meta_model.named_parameters()):
        try:
            weighted_parameter = weighted_parameters[name]
        except KeyError as exc:
            raise ValueError(
                f"L2RW meta parameter {name!r} is absent from weighted model"
            ) from exc
        if meta_parameter.shape != weighted_parameter.shape:
            raise ValueError(f"L2RW parameter shape mismatch for {name!r}")
        replace_parameter(meta_model, name, weighted_parameter)


def l2rw_resnet32(num_classes: int = 10, base_width: int = 16) -> L2RWResNet32:
    return L2RWResNet32(num_classes, base_width)
