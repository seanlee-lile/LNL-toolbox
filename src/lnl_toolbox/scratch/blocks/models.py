"""Model construction blocks implemented independently of the legacy model package."""

from __future__ import annotations

from typing import Any

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Model blocks require PyTorch; install the `train` extra.") from exc
    return torch, nn


def _resnet18(num_classes: int, preactivation: bool = False):
    torch, nn = _torch()
    import torch.nn.functional as F

    class BasicBlock(nn.Module):
        def __init__(self, incoming: int, outgoing: int, stride: int = 1) -> None:
            super().__init__()
            self.pre = bool(preactivation)
            self.bn1 = nn.BatchNorm2d(incoming)
            self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(outgoing)
            self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
            self.shortcut = nn.Identity() if incoming == outgoing and stride == 1 else nn.Conv2d(incoming, outgoing, 1, stride, bias=False)

        def forward(self, x):
            if self.pre:
                value = self.conv1(F.relu(self.bn1(x), inplace=True))
                value = self.conv2(F.relu(self.bn2(value), inplace=True))
                return value + self.shortcut(x)
            value = F.relu(self.bn1(self.conv1(x)), inplace=True)
            value = self.bn2(self.conv2(value))
            return F.relu(value + self.shortcut(x), inplace=True)

    class ScratchResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
            self.layer1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
            self.layer2 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64))
            self.layer3 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
            self.layer4 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
            self.classifier = nn.Linear(256, num_classes)

        def forward_with_features(self, x):
            value = self.layer1(self.stem(x))
            value = self.layer2(value)
            value = self.layer3(value)
            value = self.layer4(value)
            features = F.adaptive_avg_pool2d(value, 1).flatten(1)
            return self.classifier(features), features

        def forward(self, x):
            return self.forward_with_features(x)[0]

    return ScratchResNet()


@block(
    id="create_model",
    name="Create Model",
    category="Model",
    description="Create a Scratch-owned classifier by name and store it in a Context slot.",
    params={
        "model": {"type": "str", "default": "resnet18"},
        "num_classes": {"type": "int", "default": 10, "min": 2},
        "input_dim": {"type": "int", "default": 4, "min": 1},
        "hidden": {"type": "int", "default": 128, "min": 1},
        "device": {"type": "slot", "default": "device"},
        "save_as": {"type": "slot", "default": "model"},
    },
    requires=("device",),
    provides=("save_as",),
)
def create_model(
    ctx: ScratchContext,
    model: str = "resnet18",
    num_classes: int = 10,
    input_dim: int = 4,
    hidden: int = 128,
    device: str = "device",
    save_as: str = "model",
) -> None:
    _, nn = _torch()
    name = str(model).strip().lower().replace("-", "_")
    if name in {"resnet18", "preact_resnet18"}:
        network = _resnet18(num_classes, preactivation=name.startswith("preact"))
    elif name in {"linear", "linear_classifier"}:
        network = nn.Linear(input_dim, num_classes)
    elif name == "mlp":
        network = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
    else:
        raise ValueError(f"unknown Scratch model `{model}`")
    network.to(ctx[device] if device in ctx else device)
    ctx[save_as] = network
