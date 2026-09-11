"""Model construction blocks implemented independently of the legacy model package."""

from __future__ import annotations

from typing import Any, NamedTuple

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Model blocks require PyTorch; install the `train` extra.") from exc
    return torch, nn


class _FeatureOutput(NamedTuple):
    """Small Scratch-owned logits/features return contract.

    The legacy model package is deliberately not imported here.  Keeping the
    return value tuple-compatible also lets generic snapshot blocks consume
    every Scratch model without paper-specific adapters.
    """

    logits: Any
    features: Any


def _cifar_cnn8(num_classes: int):
    torch, nn = _torch()
    import torch.nn.functional as F

    class CifarCNN8(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (64, 64, 128, 128, 196, 196)
            layers = []
            incoming = 3
            for index, outgoing in enumerate(channels):
                layers.extend((nn.Conv2d(incoming, outgoing, 3, padding=1), nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True)))
                if index in (1, 3, 5):
                    layers.append(nn.MaxPool2d(2))
                incoming = outgoing
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Sequential(nn.Linear(4 * 4 * 196, 256), nn.ReLU(inplace=True), nn.Linear(256, int(num_classes)))
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None: nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight); nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

        def forward_with_features(self, inputs):
            features = self.features(inputs).flatten(1)
            return _FeatureOutput(self.classifier(features), features)

        def forward(self, inputs):
            return self.forward_with_features(inputs).logits

    return CifarCNN8()


def _cnlcu_cnn9(num_classes: int):
    torch, nn = _torch()

    class CnlcuCNN9(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (128, 128, 128, 256, 256, 256, 512, 256, 128)
            layers = []
            incoming = 3
            for index, outgoing in enumerate(channels):
                layers.extend((nn.Conv2d(incoming, outgoing, 3, padding=1), nn.LeakyReLU(0.01, inplace=True)))
                if index in (2, 5): layers.extend((nn.MaxPool2d(2), nn.Dropout(p=0.25)))
                incoming = outgoing
            layers.append(nn.AdaptiveAvgPool2d(1))
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(128, int(num_classes))
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
                    if module.bias is not None: nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight); nn.init.zeros_(module.bias)

        def forward_with_features(self, inputs):
            features = self.features(inputs).flatten(1)
            return _FeatureOutput(self.classifier(features), features)

        def forward(self, inputs):
            return self.forward_with_features(inputs).logits

    return CnlcuCNN9()


def _cifar_six_conv(num_classes: int, batch_norm_momentum: float = 0.1):
    torch, nn = _torch()
    import torch.nn.functional as F

    class SixConv(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (64, 64, 128, 128, 196, 16)
            layers = []; incoming = 3
            for outgoing in channels:
                layers.extend((nn.Conv2d(incoming, outgoing, 3, padding=1), nn.BatchNorm2d(outgoing, momentum=float(batch_norm_momentum)), nn.ReLU(inplace=True)))
                incoming = outgoing
            self.convolutions = nn.ModuleList(layers)
            self.classifier = nn.Linear(4 * 4 * channels[-1], int(num_classes))

        def forward(self, inputs):
            value = inputs; index = 0
            for _ in range(3):
                for _ in range(2):
                    value = self.convolutions[index + 2](self.convolutions[index + 1](self.convolutions[index](value)))
                    index += 3
                value = F.max_pool2d(value, 2, 2)
            return self.classifier(value.flatten(1))

    return SixConv()


def _ca2c_seven_cnn(num_classes: int):
    torch, nn = _torch()

    def block(incoming, outgoing):
        return nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1), nn.BatchNorm2d(outgoing, momentum=0.1), nn.ReLU(inplace=True))

    class CA2CSevenCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(block(3, 64), block(64, 64), nn.MaxPool2d(2), block(64, 128), block(128, 128), nn.MaxPool2d(2), block(128, 196), block(196, 16), nn.MaxPool2d(2))
            self.projector = nn.Sequential(nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Linear(256, 128))
            self.classifier = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, int(num_classes)))

        def forward_with_features(self, inputs):
            flattened = self.features(inputs).flatten(1)
            return _FeatureOutput(self.classifier(flattened), self.projector(flattened))

        def forward(self, inputs):
            return self.forward_with_features(inputs).logits

    return CA2CSevenCNN()


def _fine_seven_cnn(num_classes: int, base_width: int = 128):
    torch, nn = _torch()

    class Head(nn.Module):
        def __init__(self, incoming, scale, outgoing, activation="relu"):
            super().__init__(); hidden = round(float(scale) * incoming)
            nonlinear = nn.ReLU(inplace=True) if activation == "relu" else nn.Tanh()
            self.layers = nn.Sequential(nn.Linear(incoming, hidden), nn.BatchNorm1d(hidden), nonlinear, nn.Linear(hidden, outgoing))
        def forward(self, inputs): return self.layers(inputs)

    def block(incoming, outgoing):
        # Keep the same flat module layout as the formal SevenCNN.  The
        # caller builds each two-convolution block explicitly; wrapping each
        # convolution in another Sequential changes state-dict paths even
        # though the parameter count happens to be identical.
        return (nn.Conv2d(incoming, outgoing, 3, padding=1),
                nn.BatchNorm2d(outgoing, momentum=0.1), nn.ReLU())

    class FineSevenCNN(nn.Module):
        def __init__(self):
            super().__init__(); scale = float(base_width) / 64.0
            c1, c2, c3, c4 = [max(1, int(round(v * scale))) for v in (64, 128, 196, 16)]
            self.block1 = nn.Sequential(*block(3, c1), *block(c1, c1), nn.MaxPool2d(2))
            self.block2 = nn.Sequential(*block(c1, c2), *block(c2, c2), nn.MaxPool2d(2))
            self.block3 = nn.Sequential(nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3, momentum=0.1), nn.ReLU(), nn.Conv2d(c3, c4, 3, padding=1), nn.BatchNorm2d(c4, momentum=0.1), nn.ReLU(), nn.MaxPool2d(2))
            self.feature_size = c4 * 16
            self.classifier = Head(self.feature_size, 2.0, int(num_classes))
            self.probability_head = nn.Sequential(Head(self.feature_size, 1.0, int(num_classes), activation="tanh"), nn.Sigmoid())
            for module in self.modules():
                if isinstance(module, nn.Linear): nn.init.kaiming_normal_(module.weight); module.bias.data.zero_()
        def _features(self, inputs): return self.block3(self.block2(self.block1(inputs))).flatten(1)
        def forward_with_features(self, inputs):
            features = self._features(inputs); return _FeatureOutput(self.classifier(features), features)
        def forward_outputs(self, inputs):
            features = self._features(inputs); return {"logits": self.classifier(features), "prob": self.probability_head(features)}
        def forward(self, inputs): return self.forward_with_features(inputs).logits

    return FineSevenCNN()


def _mc_ldce_cnn(num_classes: int, classifier_bias: bool = True):
    torch, nn = _torch(); import torch.nn.functional as F

    class MCLDCE(nn.Module):
        def __init__(self):
            super().__init__(); channels = (128, 128, 128, 512, 256, 128); layers = []; incoming = 3
            for outgoing in channels: layers.extend((nn.Conv2d(incoming, outgoing, 3, padding=1), nn.LeakyReLU(0.01, inplace=True))); incoming = outgoing
            self.convolutions = nn.ModuleList(layers); self.dropout = nn.Dropout(0.25); self.classifier = nn.Linear(128, int(num_classes), bias=bool(classifier_bias)); self._feature_extractor_frozen = False
        def freeze_feature_extractor(self):
            self._feature_extractor_frozen = True
            for parameter in self.convolutions.parameters(): parameter.requires_grad_(False)
            self.dropout.eval()
        def train(self, mode=True):
            super().train(mode)
            if self._feature_extractor_frozen: self.dropout.eval()
            return self
        def forward_with_features(self, inputs):
            value = inputs
            for index in range(0, 6, 2): value = self.convolutions[index + 1](self.convolutions[index](value))
            value = self.dropout(F.max_pool2d(value, 2, 2))
            for index in range(6, 12, 2): value = self.convolutions[index + 1](self.convolutions[index](value))
            features = F.adaptive_avg_pool2d(value, 1).flatten(1); return _FeatureOutput(self.classifier(features), features)
        def forward(self, inputs): return self.forward_with_features(inputs).logits

    return MCLDCE()


def _mentor_wide_resnet(num_classes: int, num_residual_units: int = 9, leakiness: float = 0.1, width_multiplier: float = 1.0, weight_decay: float = 0.0002):
    torch, nn = _torch(); import torch.nn.functional as F; import math

    class Unit(nn.Module):
        def __init__(self, incoming, outgoing, stride, activate_before):
            super().__init__(); self.activate_before = bool(activate_before); self.leakiness = float(leakiness); self.bn1 = nn.BatchNorm2d(incoming, eps=0.001); self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(outgoing, eps=0.001); self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False); self.incoming, self.outgoing, self.stride = incoming, outgoing, stride
        def forward(self, inputs):
            if self.activate_before: activated = F.leaky_relu(inputs, negative_slope=self.leakiness); activated = self.bn1(activated); residual = activated
            else: residual = inputs; activated = F.leaky_relu(self.bn1(inputs), negative_slope=self.leakiness)
            value = self.conv2(F.leaky_relu(self.bn2(self.conv1(activated)), negative_slope=self.leakiness))
            if self.incoming != self.outgoing:
                residual = F.avg_pool2d(residual, self.stride, self.stride); difference = self.outgoing - self.incoming
                if difference < 0 or difference % 2: raise ValueError("invalid Wide-ResNet channel projection")
                residual = F.pad(residual, (0, 0, 0, 0, difference // 2, difference // 2))
            return value + residual

    class MentorWideResNet(nn.Module):
        def __init__(self):
            super().__init__(); base = [16, 160, 320, 640]; filters = [max(1, int(round(v * float(width_multiplier)))) for v in base]; self.weight_decay = float(weight_decay); self.initial = nn.Conv2d(3, filters[0], 3, padding=1, bias=False); units = []
            for stage in range(3):
                for unit in range(int(num_residual_units)): units.append(Unit(filters[stage] if unit == 0 else filters[stage + 1], filters[stage + 1], 1 if stage == 0 else 2 if unit == 0 else 1, stage == 0 and unit == 0))
            self.stages = nn.Sequential(*units); self.final_bn = nn.BatchNorm2d(filters[-1], eps=0.001); self.classifier = nn.Linear(filters[-1], int(num_classes))
            for module in self.modules():
                if isinstance(module, nn.Conv2d): nn.init.normal_(module.weight, std=math.sqrt(2.0 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))
                elif isinstance(module, nn.BatchNorm2d): nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear): nn.init.zeros_(module.bias)
        def forward_with_features(self, inputs):
            value = F.leaky_relu(self.final_bn(self.stages(self.initial(inputs))), negative_slope=float(leakiness)); features = value.mean(dim=(2,3)); return _FeatureOutput(self.classifier(features), features)
        def forward(self, inputs): return self.forward_with_features(inputs).logits
        def weighted_parameter_decay(self, sample_weights):
            kernels = [m.weight for m in self.modules() if isinstance(m, nn.Conv2d)]; return sum((w.pow(2).sum() * 0.5 for w in kernels)) * self.weight_decay * sample_weights.mean()

    return MentorWideResNet()


def _l2rw_resnet32(num_classes: int, base_width: int = 16):
    torch, nn = _torch(); import torch.nn.functional as F

    class Unit(nn.Module):
        def __init__(self, incoming, outgoing, stride, activate):
            super().__init__(); self.incoming, self.outgoing, self.stride = incoming, outgoing, stride; self.bn1 = nn.BatchNorm2d(incoming) if activate else None; self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(outgoing); self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        def forward(self, inputs):
            value = inputs if self.bn1 is None else F.relu(self.bn1(inputs), inplace=True); value = self.conv1(value); value = self.conv2(F.relu(self.bn2(value), inplace=True)); shortcut = inputs
            if self.stride > 1: shortcut = F.avg_pool2d(shortcut, self.stride, self.stride)
            if self.incoming < self.outgoing:
                d = self.outgoing - self.incoming; shortcut = F.pad(shortcut, (0,0,0,0,d//2,d//2))
            return value + shortcut

    class L2RWResNet32(nn.Module):
        def __init__(self):
            super().__init__(); widths = (int(base_width), int(base_width)*2, int(base_width)*4); self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False); self.stem_bn = nn.BatchNorm2d(widths[0]); units=[]; incoming=widths[0]
            for stage, outgoing in enumerate(widths):
                for unit in range(5): units.append(Unit(incoming, outgoing, 1 if stage == 0 or unit else 2, not (stage == 0 and unit == 0))); incoming = outgoing
            self.stages = nn.ModuleList(units); self.final_bn = nn.BatchNorm2d(widths[-1]); self.classifier = nn.Linear(widths[-1], int(num_classes))
        def forward_with_features(self, inputs):
            value = F.relu(self.stem_bn(self.stem(inputs)), inplace=True)
            for unit in self.stages: value = unit(value)
            value = F.relu(self.final_bn(value), inplace=True); features = F.adaptive_avg_pool2d(value,1).flatten(1); return _FeatureOutput(self.classifier(features), features)
        def forward(self, inputs): return self.forward_with_features(inputs).logits

    return L2RWResNet32()


@block(
    id="create_predictor",
    name="Create Predictor",
    category="Model",
    description="Create a generic feature-conditioned MLP predictor from an observed feature slot.",
    params={"features": {"type": "slot", "default": "features"}, "input_dim": {"type": "int", "default": 0, "min": 0}, "output_dim": {"type": "int", "default": 10, "min": 1}, "hidden_dim": {"type": "int", "default": 64, "min": 1}, "conditioning_dim": {"type": "int", "default": 0, "min": 0}, "time_dim": {"type": "int", "default": 0, "min": 0}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "predictor"}},
    requires=(), provides=("save_as",), placement=("batch", "top"), stage="setup", ui_group="② 初始化",
)
def create_predictor(ctx: ScratchContext, features: str = "features", input_dim: int = 0, output_dim: int = 10, hidden_dim: int = 64, conditioning_dim: int = 0, time_dim: int = 0, device: str = "device", save_as: str = "predictor") -> None:
    """Construct one independent predictor; repeated calls are idempotent.

    This is intentionally a generic MLP construction operation.  DLD (and any
    future method with two prediction heads) calls it once per output slot and
    owns the slot wiring in its Recipe.
    """
    if save_as in ctx and ctx.get(save_as) is not None:
        return
    torch, nn = _torch()
    value = ctx.get(features)
    feature_width = int(input_dim)
    target_device = ctx.get(device, "cpu")
    if value is not None:
        if getattr(value, "ndim", 0) != 2:
            raise ValueError("create_predictor requires a [N,D] feature matrix")
        feature_width = int(value.shape[1]); device = value.device
    if feature_width <= 0:
        raise ValueError("create_predictor requires input_dim or a [N,D] feature matrix")
    context_width = int(conditioning_dim) + int(time_dim)

    class FeaturePredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_dim = int(time_dim)
            self.classes = int(output_dim)
            self.network = nn.Sequential(nn.Linear(feature_width + context_width, int(hidden_dim)), nn.SiLU(), nn.Linear(int(hidden_dim), int(hidden_dim)), nn.SiLU(), nn.Linear(int(hidden_dim), int(output_dim)))
        def forward(self, *inputs):
            if not inputs:
                raise ValueError("predictor requires at least one input")
            tensors = [item.reshape(item.shape[0], -1) for item in inputs]
            if self.time_dim:
                timestep = tensors.pop(-1).float()
                half = max(self.time_dim // 2, 1)
                frequency = torch.exp(-torch.log(torch.tensor(10000.0, device=timestep.device)) * torch.arange(half, device=timestep.device).float() / max(half - 1, 1))
                embedded = timestep[:, :1] * frequency[None, :]
                tensors.append(torch.cat((embedded.sin(), embedded.cos()), dim=1)[:, :self.time_dim])
            features = torch.cat(tensors, dim=1)
            return self.network(features)

    predictor = FeaturePredictor()
    predictor.to(target_device)
    ctx[save_as] = predictor


def _cifar_resnet50(num_classes: int, base_width: int = 64, *, stem_padding: int = 1, layer_counts: tuple[int, int, int, int] = (3, 4, 6, 3), initialization: str = "kaiming", classifier_bias: bool = True):
    """Scratch-owned CIFAR ResNet-50 bottleneck topology."""
    torch, nn = _torch(); import torch.nn.functional as F

    class Bottleneck(nn.Module):
        expansion = 4
        def __init__(self, incoming, outgoing, stride=1):
            super().__init__(); expanded = int(outgoing) * 4
            self.conv1 = nn.Conv2d(incoming, outgoing, 1, bias=False); self.bn1 = nn.BatchNorm2d(outgoing)
            self.conv2 = nn.Conv2d(outgoing, outgoing, 3, stride=stride, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(outgoing)
            self.conv3 = nn.Conv2d(outgoing, expanded, 1, bias=False); self.bn3 = nn.BatchNorm2d(expanded)
            self.shortcut = nn.Identity() if stride == 1 and incoming == expanded else nn.Sequential(nn.Conv2d(incoming, expanded, 1, stride=stride, bias=False), nn.BatchNorm2d(expanded))
        def forward(self, inputs):
            value = F.relu(self.bn1(self.conv1(inputs)), inplace=True); value = F.relu(self.bn2(self.conv2(value)), inplace=True); value = self.bn3(self.conv3(value)); return F.relu(value + self.shortcut(inputs), inplace=True)

    class ResNet50(nn.Module):
        def __init__(self):
            super().__init__(); self.stem = nn.Sequential(nn.Conv2d(3, int(base_width), 3, padding=int(stem_padding), bias=False), nn.BatchNorm2d(int(base_width)), nn.ReLU(inplace=True)); incoming = int(base_width); counts = tuple(int(v) for v in layer_counts); self.layer1 = self._stage(incoming, int(base_width), counts[0], 1); incoming = int(base_width) * 4; self.layer2 = self._stage(incoming, int(base_width)*2, counts[1], 2); incoming = int(base_width)*8; self.layer3 = self._stage(incoming, int(base_width)*4, counts[2], 2); incoming = int(base_width)*16; self.layer4 = self._stage(incoming, int(base_width)*8, counts[3], 2); self.classifier = nn.Linear(int(base_width)*32, int(num_classes), bias=bool(classifier_bias))
        @staticmethod
        def _stage(incoming, outgoing, count, stride):
            blocks = [Bottleneck(incoming, outgoing, stride)]; blocks.extend(Bottleneck(outgoing*4, outgoing) for _ in range(int(count)-1)); return nn.Sequential(*blocks)
        def forward_with_features(self, inputs):
            value = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(inputs))))); features = F.adaptive_avg_pool2d(value, 1).flatten(1); return _FeatureOutput(self.classifier(features), features)
        def forward(self, inputs): return self.forward_with_features(inputs).logits
    initialization = str(initialization).strip().lower()
    if initialization not in {"kaiming", "torch_default"}:
        raise ValueError("initialization must be 'kaiming' or 'torch_default'")
    network = ResNet50()
    if initialization == "kaiming":
        for module in network.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    return network


def _cifar_resnet_depth(depth: int, num_classes: int, base_width: int = 16):
    torch, nn = _torch(); import torch.nn.functional as F
    if depth < 8 or (int(depth) - 2) % 6: raise ValueError("CIFAR depth must satisfy depth = 6*n + 2")
    blocks = (int(depth) - 2) // 6; widths = (int(base_width), int(base_width)*2, int(base_width)*4)

    class Basic(nn.Module):
        def __init__(self, incoming, outgoing, stride):
            super().__init__(); self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(outgoing); self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(outgoing); self.shortcut = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False), nn.BatchNorm2d(outgoing))
        def forward(self, inputs):
            value = F.relu(self.bn1(self.conv1(inputs)), inplace=True); value = self.bn2(self.conv2(value)); return F.relu(value + self.shortcut(inputs), inplace=True)

    class DepthNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Match the formal CifarResNetDepth module layout: stem and BN
            # are separate named modules (rather than a nested Sequential),
            # which keeps checkpoints and topology inspection faithful.
            self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(widths[0])
            incoming = widths[0]; stages=[]
            for stage, outgoing in enumerate(widths):
                layers=[]
                for unit in range(blocks): layers.append(Basic(incoming, outgoing, 1 if stage == 0 or unit else 2)); incoming = outgoing
                stages.append(nn.Sequential(*layers))
            self.stages = nn.ModuleList(stages); self.classifier = nn.Linear(widths[-1], int(num_classes))
        def forward_with_features(self, inputs):
            value = F.relu(self.bn(self.stem(inputs)), inplace=True)
            for stage in self.stages: value = stage(value)
            features = F.adaptive_avg_pool2d(value,1).flatten(1); return _FeatureOutput(self.classifier(features), features)
        def forward(self, inputs): return self.forward_with_features(inputs).logits
    return DepthNet()


def _resnet18(
    num_classes: int,
    preactivation: bool = False,
    layer_counts: tuple[int, int, int, int] = (2, 2, 2, 2),
    base_width: int = 32,
    initialization: str = "kaiming",
    classifier_bias: bool = True,
    stem_padding: int = 1,
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
            if incoming == outgoing and stride == 1:
                self.shortcut = nn.Identity()
            elif preactivation:
                self.shortcut = nn.Conv2d(incoming, outgoing, 1, stride, bias=False)
            else:
                self.shortcut = nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing))

        def forward(self, x):
            if self.pre:
                # The pre-activation is evaluated once.  Reusing the same
                # tensor for the residual branch is part of the formal
                # PreActBlock semantics (and avoids updating BatchNorm twice
                # for one sample).
                activated = F.relu(self.bn1(x), inplace=True)
                value = self.conv1(activated)
                value = self.conv2(F.relu(self.bn2(value), inplace=True))
                shortcut = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(activated)
                return value + shortcut
            value = F.relu(self.bn1(self.conv1(x)), inplace=True)
            value = self.bn2(self.conv2(value))
            return F.relu(value + self.shortcut(x), inplace=True)

    class ScratchResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            widths = (int(base_width), int(base_width) * 2, int(base_width) * 4, int(base_width) * 8)
            if preactivation:
                self.stem = nn.Conv2d(3, widths[0], 3, 1, int(stem_padding), bias=False)
                self.stem_bn = nn.Identity()
            else:
                # The formal post-activation CIFAR ResNet exposes a single
                # sequential stem (conv -> BN -> ReLU).
                self.stem = nn.Sequential(
                    nn.Conv2d(3, widths[0], 3, 1, int(stem_padding), bias=False),
                    nn.BatchNorm2d(widths[0]),
                    nn.ReLU(inplace=True),
                )
                self.stem_bn = nn.Identity()

            def stage(incoming: int, outgoing: int, count: int, stride: int) -> nn.Sequential:
                blocks = [BasicBlock(incoming, outgoing, stride)]
                blocks.extend(BasicBlock(outgoing, outgoing) for _ in range(int(count) - 1))
                return nn.Sequential(*blocks)

            self.layer1 = stage(widths[0], widths[0], layer_counts[0], 1)
            self.layer2 = stage(widths[0], widths[1], layer_counts[1], 2)
            self.layer3 = stage(widths[1], widths[2], layer_counts[2], 2)
            self.layer4 = stage(widths[2], widths[3], layer_counts[3], 2)
            self.final_bn = nn.BatchNorm2d(widths[3]) if preactivation else nn.Identity()
            self.classifier = nn.Linear(widths[3], num_classes, bias=bool(classifier_bias))

        def forward_with_features(self, x):
            # The post-activation stem already applies BN and ReLU; the
            # pre-activation stem is a bare convolution.
            value = self.stem(x)
            value = self.layer1(value)
            value = self.layer2(value)
            value = self.layer3(value)
            value = self.layer4(value)
            if preactivation:
                value = F.relu(self.final_bn(value), inplace=True)
            features = F.adaptive_avg_pool2d(value, 1).flatten(1)
            return _FeatureOutput(self.classifier(features), features)

        def forward(self, x):
            return self.forward_with_features(x)[0]

    initialization = str(initialization).strip().lower()
    if initialization not in {"kaiming", "torch_default"}:
        raise ValueError("initialization must be 'kaiming' or 'torch_default'")
    network = ScratchResNet()
    if initialization == "kaiming":
        for module in network.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    return network


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
    # A loaded Scratch dataset publishes its class cardinality as a runtime
    # contract.  Refuse an incompatible classifier before the first CUDA
    # kernel is launched; otherwise an out-of-range target only surfaces later
    # as an asynchronous device-side assert in a gather/loss block.
    dataset_classes = ctx.get("num_classes")
    runtime_limits = ctx.get("_runtime_limits")
    fixture_run = isinstance(runtime_limits, dict) and bool(runtime_limits.get("fixture"))
    if dataset_classes is not None and not fixture_run and int(dataset_classes) != int(num_classes):
        raise ValueError(
            f"model num_classes={int(num_classes)} does not match the loaded "
            f"dataset num_classes={int(dataset_classes)}; select a dataset with "
            "the matching class count or update the model num_classes parameter"
        )
    name = str(model).strip().lower().replace("-", "_")
    effective_classifier_bias = bool(classifier_bias and bias)
    if name in {"resnet18", "cifar_resnet18", "preact_resnet18"}:
        network = _resnet18(num_classes, preactivation=name.startswith("preact"), base_width=int(base_width), initialization=initialization, classifier_bias=effective_classifier_bias, stem_padding=int(stem_padding))
    elif name in {"resnet34", "cifar_resnet34"}:
        network = _resnet18(num_classes, layer_counts=(3, 4, 6, 3), base_width=int(base_width), initialization=initialization, classifier_bias=effective_classifier_bias, stem_padding=int(stem_padding))
    elif name in {"resnet50", "cifar_resnet50"}:
        network = _cifar_resnet50(num_classes, int(base_width), stem_padding=int(stem_padding), initialization=initialization, classifier_bias=effective_classifier_bias)
    elif name == "resnet101":
        network = _cifar_resnet50(num_classes, int(base_width), stem_padding=int(stem_padding), layer_counts=(3, 4, 23, 3), initialization=initialization, classifier_bias=effective_classifier_bias)
    elif name == "resnet14":
        network = _cifar_resnet_depth(14, num_classes, int(base_width))
    elif name in {"resnet32", "cifar_resnet32"}:
        network = _cifar_resnet_depth(32, num_classes, int(base_width))
    elif name == "cifar_cnn8":
        network = _cifar_cnn8(num_classes)
    elif name == "cnlcu_cnn9":
        network = _cnlcu_cnn9(num_classes)
    elif name == "mentor_wide_resnet":
        network = _mentor_wide_resnet(num_classes, num_residual_units=int(num_residual_units), leakiness=float(leakiness), width_multiplier=float(width_multiplier), weight_decay=float(weight_decay))
    elif name == "fine_seven_cnn":
        network = _fine_seven_cnn(num_classes, int(base_width))
    elif name == "mc_ldce_cnn":
        network = _mc_ldce_cnn(num_classes, classifier_bias=bool(classifier_bias))
    elif name == "ca2c_seven_cnn":
        network = _ca2c_seven_cnn(num_classes)
    elif name == "l2rw_resnet32":
        network = _l2rw_resnet32(num_classes, int(base_width))
    elif name == "cifar_six_conv":
        network = _cifar_six_conv(num_classes)
    elif name in {"linear", "linear_classifier"}:
        network = nn.Linear(input_dim, num_classes)
    elif name == "mlp":
        network = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
    else:
        raise ValueError(f"unknown Scratch model `{model}`")
    if not classifier_bias and isinstance(getattr(network, "classifier", None), nn.Linear):
        classifier = network.classifier
        replacement = nn.Linear(classifier.in_features, classifier.out_features, bias=False)
        replacement.weight.data.copy_(classifier.weight.data)
        network.classifier = replacement
    network.to(ctx[device] if device in ctx else device)
    ctx[save_as] = network


@block(
    id="attach_trainable_head",
    name="Attach Trainable Head",
    category="Model",
    description="Attach a named trainable linear head to an existing model without changing its base parameters.",
    params={"model": {"type": "slot", "default": "model"},
            "input_dim": {"type": "int", "default": 0, "min": 0},
            "output_dim": {"type": "int", "default": 0, "min": 1},
            "head_name": {"type": "str", "default": "head"},
            "bias": {"type": "bool", "default": False}},
    requires=("model",), provides=(), placement=("top",), stage="setup", ui_group="② 初始化",
)
def attach_trainable_head(ctx: ScratchContext, model: str = "model", input_dim: int = 0,
                          output_dim: int = 0, head_name: str = "head",
                          bias: bool = False) -> None:
    torch, nn = _torch()
    module = ctx[model]
    name = str(head_name).strip()
    if not name or not name.isidentifier():
        raise ValueError("head_name must be a valid attribute name")
    if hasattr(module, name):
        return
    incoming = int(input_dim)
    if incoming <= 0:
        classifier = getattr(module, "classifier", None)
        incoming = int(getattr(classifier, "in_features", 0))
    outgoing = int(output_dim)
    if incoming <= 0 or outgoing <= 0:
        raise ValueError("attach_trainable_head requires positive input_dim and output_dim")
    setattr(module, name, nn.Linear(incoming, outgoing, bias=bool(bias)))


@block(
    id="zero_module_parameters",
    name="Zero Module Parameters",
    category="Model",
    description="Set all parameters of a module or named submodule to zero.",
    params={"module": {"type": "slot", "default": "model"},
            "submodule": {"type": "str", "default": ""}},
    requires=("module",), provides=(), placement=("top", "epoch"), stage="setup", ui_group="② 初始化",
)
def zero_module_parameters(ctx: ScratchContext, module: str = "model", submodule: str = "") -> None:
    value = ctx[module]
    target = value
    if str(submodule).strip():
        for part in str(submodule).split("."):
            target = getattr(target, part)
    parameters = list(target.parameters()) if hasattr(target, "parameters") else []
    if not parameters:
        raise ValueError("zero_module_parameters target has no parameters")
    torch, _ = _torch()
    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()


@block(
    id="set_module_trainability",
    name="Set Module Trainability",
    category="Model",
    description="Freeze or unfreeze all parameters in a module or named submodule.",
    params={"module": {"type": "slot", "default": "model"},
            "submodule": {"type": "str", "default": ""},
            "trainable": {"type": "bool", "default": True}},
    requires=("module",), provides=(), placement=("top", "epoch"), stage="setup", ui_group="② 初始化",
)
def set_module_trainability(ctx: ScratchContext, module: str = "model", submodule: str = "",
                            trainable: bool = True) -> None:
    value = ctx[module]
    target = value
    if str(submodule).strip():
        for part in str(submodule).split("."):
            target = getattr(target, part)
    if not hasattr(target, "parameters"):
        raise TypeError("set_module_trainability requires a torch module")
    for parameter in target.parameters():
        parameter.requires_grad_(bool(trainable))


@block(
    id="load_model_artifact",
    name="Load Model Artifact",
    category="Model",
    description="Load a checkpoint into an existing model with optional SHA-256 and provenance validation.",
    params={"model": {"type": "slot", "default": "model"},
            "path": {"type": "str", "default": ""},
            "source_env": {"type": "str", "default": ""},
            "checkpoint_sha256": {"type": "str", "default": ""},
            "manifest_sha256": {"type": "str", "default": ""},
            "mapping_hash": {"type": "str", "default": ""},
            "dataset_fingerprint": {"type": "str", "default": ""},
            "strict": {"type": "bool", "default": True},
            "save_as": {"type": "slot", "default": "loaded_model"}},
    requires=("model",), provides=("save_as",), placement=("top",), stage="setup", ui_group="② 初始化",
)
def load_model_artifact(ctx: ScratchContext, model: str = "model", path: str = "",
                        source_env: str = "", checkpoint_sha256: str = "",
                        manifest_sha256: str = "", mapping_hash: str = "",
                        dataset_fingerprint: str = "", strict: bool = True,
                        save_as: str = "loaded_model") -> None:
    import hashlib
    import os
    from pathlib import Path
    import torch
    if bool((ctx.get("_runtime_limits") or {}).get("fixture")) and not str(path).strip() and not str(source_env).strip():
        # Bounded catalog runs intentionally omit immutable external artifacts.
        # The formal path still validates a real checkpoint; fixture mode only
        # supplies an explicitly marked local stand-in for structural tests.
        ctx.setdefault("artifact_provenance", {})[save_as] = {"fixture": True}
        ctx[save_as] = ctx[model]
        return
    resolved = str(path).strip() or (os.environ.get(str(source_env).strip()) if str(source_env).strip() else "")
    if not resolved and bool((ctx.get("_runtime_limits") or {}).get("fixture")):
        ctx.setdefault("artifact_provenance", {})[save_as] = {"fixture": True, "source_env": str(source_env)}
        ctx[save_as] = ctx[model]
        return
    if not resolved:
        raise ValueError("load_model_artifact requires path or source_env")
    artifact_path = Path(resolved)
    if not artifact_path.is_file():
        raise FileNotFoundError(str(artifact_path))
    if str(checkpoint_sha256).strip():
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if digest.lower() != str(checkpoint_sha256).strip().lower():
            raise ValueError("model artifact SHA-256 does not match checkpoint_sha256")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError("model artifact must contain a state_dict mapping")
    ctx[model].load_state_dict(state, strict=bool(strict))
    # Preserve provenance values in the context for downstream audit/reporting.
    ctx.setdefault("artifact_provenance", {})[save_as] = {
        "checkpoint_sha256": str(checkpoint_sha256),
        "manifest_sha256": str(manifest_sha256),
        "mapping_hash": str(mapping_hash),
        "dataset_fingerprint": str(dataset_fingerprint),
    }
    ctx[save_as] = ctx[model]
