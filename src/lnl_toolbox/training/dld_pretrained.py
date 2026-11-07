from __future__ import annotations

"""Verified pretrained feature source for real-data DLD runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import torch
from torch import nn
from torch.nn import functional as F

from lnl_toolbox.models.feature_output import FeatureOutput
from lnl_toolbox.training.noisy_labels import file_sha256


@dataclass(frozen=True)
class DLDSourceFileIdentity:
    path: str
    sha256: str
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: str | Path) -> "DLDSourceFileIdentity":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"DLD pretrained source file does not exist: {source}")
        stat = source.stat()
        return cls(
            str(source), file_sha256(source), int(stat.st_size), int(stat.st_mtime_ns)
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }

    def assert_unchanged(self) -> None:
        if DLDSourceFileIdentity.capture(self.path) != self:
            raise RuntimeError(f"DLD pretrained source changed: {self.path}")






class _TorchvisionResNet34FeatureModel(nn.Module):
    """Frozen ImageNet ResNet34 with an explicit CIFAR input contract."""

    _CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    _CIFAR_STD = (0.2470, 0.2435, 0.2616)
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model: nn.Module, input_contract: str = "cifar10_standard_normalized") -> None:
        super().__init__()
        self.model = model
        self.input_contract = input_contract
        if input_contract == "cifar100_standard_normalized":
            self._CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
            self._CIFAR_STD = (0.2675, 0.2565, 0.2761)

    @staticmethod
    def _values(values: tuple[float, ...], inputs: torch.Tensor) -> torch.Tensor:
        return inputs.new_tensor(values).view(1, 3, 1, 1)

    def _preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("DLD pretrained ResNet34 requires NCHW RGB inputs")
        values = inputs * self._values(self._CIFAR_STD, inputs)
        values = values + self._values(self._CIFAR_MEAN, inputs)
        values = values.clamp(0.0, 1.0)
        values = F.interpolate(
            values, size=(256, 256), mode="bilinear", align_corners=False
        )
        values = values[:, :, 16:240, 16:240]
        return (
            values - self._values(self._IMAGENET_MEAN, values)
        ) / self._values(self._IMAGENET_STD, values)

    def forward_with_features(self, inputs: torch.Tensor) -> FeatureOutput:
        values = self._preprocess(inputs)
        model = self.model
        values = model.conv1(values)
        values = model.bn1(values)
        values = model.relu(values)
        values = model.maxpool(values)
        values = model.layer1(values)
        values = model.layer2(values)
        values = model.layer3(values)
        values = model.layer4(values)
        features = torch.flatten(model.avgpool(values), 1)
        return FeatureOutput(model.fc(features), features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs).logits


@dataclass(frozen=True)
class DLDTorchvisionResNet34Source:
    checkpoint: DLDSourceFileIdentity
    model: nn.Module

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "adapter": "torchvision_resnet34_imagenet1k_v1",
            "source": "torchvision.models.ResNet34_Weights.IMAGENET1K_V1",
            "weights": "IMAGENET1K_V1",
            "checkpoint": self.checkpoint.state_dict(),
            "feature_output": "global_average_pool(layer4)",
            "feature_dimension": 512,
            "input_contract": self.model.input_contract,
            "preprocessing": {
                "restore": self.model.input_contract + " to [0,1]",
                "resize": 256,
                "center_crop": 224,
                "normalization": "ImageNet-1K V1",
            },
            "frozen": True,
        }

    def assert_unchanged(self) -> None:
        self.checkpoint.assert_unchanged()


def _cached_torchvision_weight(url: str) -> Path:
    filename = Path(urlparse(url).path).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file():
        raise FileNotFoundError(
            "DLD needs the official torchvision ResNet34 IMAGENET1K_V1 weights at "
            f"{path}. Download resnet34-b627a593.pth from the official PyTorch "
            "weight URL before running this recipe."
        )
    return path


def load_torchvision_resnet34_imagenet1k_v1_source(
    source_config: Mapping[str, Any],
) -> DLDTorchvisionResNet34Source:
    if str(source_config.get("adapter", "")).strip().lower() != (
        "torchvision_resnet34_imagenet1k_v1"
    ):
        raise ValueError("DLD torchvision feature adapter identity mismatch")
    if str(source_config.get("weights", "")).strip() != "IMAGENET1K_V1":
        raise ValueError("DLD torchvision ResNet34 weights must be IMAGENET1K_V1")
    if str(source_config.get("input_contract", "")).strip() not in {"cifar10_standard_normalized", "cifar100_standard_normalized"}:
        raise ValueError("DLD torchvision ResNet34 input contract mismatch")

    from torchvision.models import ResNet34_Weights, resnet34

    weights = ResNet34_Weights.IMAGENET1K_V1
    cached = _cached_torchvision_weight(weights.url)
    identity = DLDSourceFileIdentity.capture(cached)
    # The official cached checkpoint is fingerprinted before loading.
    model = _TorchvisionResNet34FeatureModel(resnet34(weights=weights), str(source_config["input_contract"])).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = DLDTorchvisionResNet34Source(identity, model)
    result.assert_unchanged()
    return result


def load_dld_feature_source(
    extractor_config: Mapping[str, Any],
    *,
    num_classes: int,
) -> DLDTorchvisionResNet34Source:
    external = extractor_config.get("external")
    if not isinstance(external, Mapping):
        raise TypeError("DLD external feature source must be a mapping")
    adapter = str(external.get("adapter", "")).strip().lower()
    if adapter == "torchvision_resnet34_imagenet1k_v1":
        return load_torchvision_resnet34_imagenet1k_v1_source(external)
    raise ValueError(f"unsupported DLD feature adapter: {adapter}")


__all__ = [
    "DLDSourceFileIdentity",
    "DLDTorchvisionResNet34Source",
    "load_dld_feature_source",
    "load_torchvision_resnet34_imagenet1k_v1_source",
]
