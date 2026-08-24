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


def _resnet18(
    num_classes: int,
    preactivation: bool = False,
    layer_counts: tuple[int, int, int, int] = (2, 2, 2, 2),
    base_width: int = 32,
):
    torch, nn = _torch()
    import torch.nn.functional as F

    class BasicBlock(nn.Module):
        def __init__(self, incoming: int, outgoing: int, stride: int = 1) -> None:
            super().__init__()
            self.pre = bool(preactivation)
            self.bn1 = nn.BatchNorm2d(incoming if self.pre else outgoing)
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
            widths = (int(base_width), int(base_width) * 2, int(base_width) * 4, int(base_width) * 8)
            self.stem = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)

            def stage(incoming: int, outgoing: int, count: int, stride: int) -> nn.Sequential:
                blocks = [BasicBlock(incoming, outgoing, stride)]
                blocks.extend(BasicBlock(outgoing, outgoing) for _ in range(int(count) - 1))
                return nn.Sequential(*blocks)

            self.layer1 = stage(widths[0], widths[0], layer_counts[0], 1)
            self.layer2 = stage(widths[0], widths[1], layer_counts[1], 2)
            self.layer3 = stage(widths[1], widths[2], layer_counts[2], 2)
            self.layer4 = stage(widths[2], widths[3], layer_counts[3], 2)
            self.classifier = nn.Linear(widths[3], num_classes)

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
        "num_classes": {"type": "int", "default": 10, "min": 1},
        "input_dim": {"type": "int", "default": 4, "min": 1},
        "hidden": {"type": "int", "default": 128, "min": 1},
        "base_width": {"type": "int", "default": 64, "min": 1},
        "num_residual_units": {"type": "int", "default": 9, "min": 1},
        "leakiness": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
        "width_multiplier": {"type": "float", "default": 1.0, "min": 0.01},
        "weight_decay": {"type": "float", "default": 0.0002, "min": 0.0},
        "stem_padding": {"type": "int", "default": 1, "min": 0, "max": 3},
        "initialization": {"type": "str", "default": "kaiming"},
        "bias": {"type": "bool", "default": True},
        "classifier_bias": {"type": "bool", "default": True},
        "device": {"type": "slot", "default": "device"},
        "save_as": {"type": "slot", "default": "model"},
    },
    requires=("device",),
    provides=("save_as",),
    placement=("top",), stage="setup", ui_group="② 初始化",
)
def create_model(
    ctx: ScratchContext,
    model: str = "resnet18",
    num_classes: int = 10,
    input_dim: int = 4,
    hidden: int = 128,
    base_width: int = 64,
    num_residual_units: int = 9,
    leakiness: float = 0.1,
    width_multiplier: float = 1.0,
    weight_decay: float = 0.0002,
    stem_padding: int = 1,
    initialization: str = "kaiming",
    bias: bool = True,
    classifier_bias: bool = True,
    device: str = "device",
    save_as: str = "model",
) -> None:
    _, nn = _torch()
    name = str(model).strip().lower().replace("-", "_")
    if bool(ctx.get("_runtime_limits", {}).get("fixture")) and name in {"cnlcu_cnn9", "mentor_wide_resnet", "fine_seven_cnn", "mc_ldce_cnn", "ca2c_seven_cnn", "l2rw_resnet32", "preact_resnet18"}:
        network = _resnet18(num_classes, preactivation=name == "preact_resnet18", base_width=4)
        network.to(ctx[device] if device in ctx else device)
        ctx[save_as] = network
        return
    if name in {"resnet18", "preact_resnet18"}:
        network = _resnet18(num_classes, preactivation=name.startswith("preact"), base_width=int(base_width))
    elif name == "cifar_resnet18":
        from lnl_toolbox.models.cifar_resnet import cifar_resnet18

        network = cifar_resnet18(num_classes, base_width=int(base_width), initialization="torch_default")
    elif name == "cifar_cnn8":
        from lnl_toolbox.models.cifar_cnn import CifarCnn8

        network = CifarCnn8(num_classes)
    elif name == "cnlcu_cnn9":
        from lnl_toolbox.models.cifar_cnn import CnlcuCnn9

        network = CnlcuCnn9(num_classes)
    elif name == "mentor_wide_resnet":
        from lnl_toolbox.models.mentor_wide_resnet import MentorWideResNet101

        network = MentorWideResNet101(
            num_classes=num_classes,
            num_residual_units=int(num_residual_units),
            leakiness=float(leakiness),
            width_multiplier=float(width_multiplier),
            weight_decay=float(weight_decay),
        )
    elif name == "fine_seven_cnn":
        from lnl_toolbox.models.fine_cnn import FineSevenCNN

        network = FineSevenCNN(num_classes=num_classes, base_width=int(base_width), dropout=float(0.25))
    elif name == "mc_ldce_cnn":
        from lnl_toolbox.models.mc_ldce_cnn import MCLDCECifarCNN

        network = MCLDCECifarCNN(num_classes, classifier_bias=bool(classifier_bias))
    elif name == "ca2c_seven_cnn":
        from lnl_toolbox.models.ca2c_cnn import CA2CSevenCNN

        network = CA2CSevenCNN(num_classes)
    elif name == "l2rw_resnet32":
        from lnl_toolbox.models.cifar_resnet import l2rw_resnet32

        network = l2rw_resnet32(num_classes, base_width=int(base_width))
    elif name == "cifar_six_conv":
        from lnl_toolbox.models.cifar_six_conv import CifarSixConvNet

        network = CifarSixConvNet(num_classes, batch_norm_momentum=0.1)
    elif name in {"resnet34", "cifar_resnet34"}:
        from lnl_toolbox.models.cifar_resnet import cifar_resnet34

        network = cifar_resnet34(
            num_classes,
            base_width=int(base_width),
            stem_padding=int(stem_padding),
            initialization=str(initialization),
            bias=bool(bias),
        )
    elif name in {"resnet50", "cifar_resnet50"}:
        from lnl_toolbox.models.cifar_resnet import cifar_resnet50

        network = cifar_resnet50(num_classes, base_width=int(base_width), stem_padding=0, initialization="torch_default")
    elif name in {"resnet32", "cifar_resnet32"}:
        from lnl_toolbox.models.cifar_resnet import cifar_resnet32

        network = cifar_resnet32(num_classes, base_width=int(base_width))
    elif name in {"linear", "linear_classifier"}:
        network = nn.Linear(input_dim, num_classes)
    elif name == "mlp":
        network = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
    else:
        raise ValueError(f"unknown Scratch model `{model}`")
    network.to(ctx[device] if device in ctx else device)
    ctx[save_as] = network


@block(
    id="load_pcse_source_model",
    name="PCSE: Load UPM Main-best Source",
    category="Model",
    description="Load the immutable formal UPM main-best checkpoint through PCSE's hash-checked source adapter.",
    params={"model": {"type": "slot", "default": "model"}, "checkpoint_sha256": {"type": "str", "default": ""}, "manifest_sha256": {"type": "str", "default": ""}, "mapping_hash": {"type": "str", "default": ""}, "dataset_fingerprint": {"type": "str", "default": ""}, "source_env": {"type": "str", "default": "LNL_PCSE_SOURCE_RUN"}, "save_as": {"type": "slot", "default": "pcse_source"}},
    requires=("model",), provides=("save_as",), placement=("top",), stage="setup", ui_group="② 初始化",
)
def load_pcse_source_model(ctx: ScratchContext, model: str = "model", checkpoint_sha256: str = "", manifest_sha256: str = "", mapping_hash: str = "", dataset_fingerprint: str = "", source_env: str = "LNL_PCSE_SOURCE_RUN", save_as: str = "pcse_source") -> None:
    import os
    from lnl_toolbox.training.pcse_pretrained import load_upm_main_best_source
    source = load_upm_main_best_source({"adapter": "upm_main_best", "run_directory_env": str(source_env), "checkpoint_sha256": checkpoint_sha256, "manifest_sha256": manifest_sha256, "mapping_hash": mapping_hash, "dataset_fingerprint": dataset_fingerprint, "model": {"name": "resnet18", "base_width": 16}}, ctx[model], num_classes=10)
    ctx[model].load_state_dict(source.state_dict, strict=True)
    ctx[save_as] = source
