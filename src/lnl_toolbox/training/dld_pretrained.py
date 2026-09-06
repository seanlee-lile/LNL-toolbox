from __future__ import annotations

"""Strict, read-only pretrained feature source for real-data DLD runs."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import torch
from torch import nn
from torch.nn import functional as F

from lnl_toolbox.models.feature_output import FeatureOutput
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.noisy_labels import file_sha256


def _digest(value: Any, owner: str) -> str:
    parsed = str(value).strip().lower()
    if len(parsed) != 64 or any(char not in "0123456789abcdef" for char in parsed):
        raise ValueError(f"{owner} must be a 64-character SHA-256 digest")
    return parsed


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


@dataclass(frozen=True)
class DLDUPMMainBestSource:
    run_dir: Path
    checkpoint: DLDSourceFileIdentity
    manifest: DLDSourceFileIdentity
    checkpoint_payload: Mapping[str, Any]
    noise_manifest: NoiseManifest
    model: nn.Module

    @property
    def provenance(self) -> dict[str, Any]:
        noise = self.checkpoint_payload["noise"]
        config = self.checkpoint_payload["config"]
        state = self.checkpoint_payload["upm_state"]
        return {
            "adapter": "upm_main_best",
            "method": "upm",
            "checkpoint_role": "main_best",
            "run_dir": str(self.run_dir),
            "checkpoint": self.checkpoint.state_dict(),
            "manifest": self.manifest.state_dict(),
            "mapping_hash": str(noise["mapping_hash"]),
            "dataset_fingerprint": str(noise["dataset_fingerprint"]),
            "model": dict(config["upm"]["main"]["model"]),
            "completed_epochs": int(state["main_completed_epochs"]),
            "global_step": int(state["main_global_step"]),
            "best_epoch": int(state["main_best_epoch"]),
            "best_validation_accuracy": float(state["main_best_validation_accuracy"]),
        }

    def assert_unchanged(self) -> None:
        self.checkpoint.assert_unchanged()
        self.manifest.assert_unchanged()


def load_upm_main_best_feature_source(
    source_config: Mapping[str, Any],
    model: nn.Module,
    *,
    num_classes: int,
) -> DLDUPMMainBestSource:
    """Load the one explicitly supported source schema without guessing."""

    if str(source_config.get("adapter", "")).strip().lower() != "upm_main_best":
        raise ValueError("DLD external feature adapter must be upm_main_best")
    environment_name = str(source_config.get("run_directory_env", "")).strip()
    run_value = os.environ.get(environment_name, "").strip()
    if not environment_name or not run_value:
        raise ValueError(f"DLD source environment variable is not set: {environment_name}")
    run_dir = Path(run_value).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"DLD pretrained source run does not exist: {run_dir}")
    checkpoint = DLDSourceFileIdentity.capture(run_dir / "best.pt")
    manifest_identity = DLDSourceFileIdentity.capture(run_dir / "noise_manifest.npz")
    if checkpoint.sha256 != _digest(source_config.get("checkpoint_sha256"), "checkpoint_sha256"):
        raise ValueError("DLD source checkpoint SHA-256 mismatch")
    if manifest_identity.sha256 != _digest(source_config.get("manifest_sha256"), "manifest_sha256"):
        raise ValueError("DLD source manifest SHA-256 mismatch")

    payload = read_checkpoint(checkpoint.path, "cpu")
    if payload.get("method") != "upm" or payload.get("checkpoint_role") != "main_best":
        raise ValueError("DLD source must be an UPM main_best checkpoint")
    config = payload.get("config")
    noise = payload.get("noise")
    state = payload.get("upm_state")
    model_state = payload.get("best_main_model_state")
    if not all(isinstance(item, Mapping) for item in (config, noise, state, model_state)):
        raise ValueError("DLD UPM source checkpoint schema is incomplete")
    expected_model = dict(source_config.get("model", {}))
    actual_model = dict(config.get("upm", {}).get("main", {}).get("model", {}))
    if expected_model != actual_model:
        raise ValueError("DLD source model architecture mismatch")
    if str(noise.get("dataset")) != "cifar10" or int(noise.get("num_classes", -1)) != num_classes:
        raise ValueError("DLD source dataset or class count mismatch")
    expected_mapping = _digest(source_config.get("mapping_hash"), "mapping_hash")
    expected_fingerprint = _digest(
        source_config.get("dataset_fingerprint"), "dataset_fingerprint"
    )
    if str(noise.get("mapping_hash", "")).lower() != expected_mapping:
        raise ValueError("DLD source mapping hash mismatch")
    if str(noise.get("dataset_fingerprint", "")).lower() != expected_fingerprint:
        raise ValueError("DLD source dataset fingerprint mismatch")
    if str(noise.get("manifest_sha256", "")).lower() != manifest_identity.sha256:
        raise ValueError("DLD source checkpoint manifest identity mismatch")
    manifest = NoiseManifest.load(manifest_identity.path)
    if manifest.mapping_hash != expected_mapping or manifest.dataset_fingerprint != expected_fingerprint:
        raise ValueError("DLD source manifest provenance mismatch")
    if manifest.dataset != "cifar10" or manifest.num_classes != num_classes:
        raise ValueError("DLD source manifest dataset contract mismatch")
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise ValueError("DLD source state_dict is incompatible") from error
    result = DLDUPMMainBestSource(
        run_dir, checkpoint, manifest_identity, payload, manifest, model
    )
    result.assert_unchanged()
    return result


class _TorchvisionResNet34FeatureModel(nn.Module):
    """Frozen ImageNet ResNet34 with an explicit CIFAR input contract."""

    _CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    _CIFAR_STD = (0.2470, 0.2435, 0.2616)
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

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
            "input_contract": "cifar10_standard_normalized",
            "preprocessing": {
                "restore": "CIFAR-10 standard normalized tensor to [0,1]",
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
            "DLD pretrained ResNet34 weights are not cached; expected official "
            f"torchvision weights at: {path}. Cache IMAGENET1K_V1 explicitly "
            "before validating this conditional recipe."
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
    if str(source_config.get("input_contract", "")).strip() != (
        "cifar10_standard_normalized"
    ):
        raise ValueError("DLD torchvision ResNet34 input contract mismatch")

    from torchvision.models import ResNet34_Weights, resnet34

    weights = ResNet34_Weights.IMAGENET1K_V1
    cached = _cached_torchvision_weight(weights.url)
    identity = DLDSourceFileIdentity.capture(cached)
    # Cache presence is checked first so this call never initiates a download.
    model = _TorchvisionResNet34FeatureModel(resnet34(weights=weights)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = DLDTorchvisionResNet34Source(identity, model)
    result.assert_unchanged()
    return result


def load_dld_feature_source(
    extractor_config: Mapping[str, Any],
    *,
    num_classes: int,
) -> DLDUPMMainBestSource | DLDTorchvisionResNet34Source:
    external = extractor_config.get("external")
    if not isinstance(external, Mapping):
        raise TypeError("DLD external feature source must be a mapping")
    adapter = str(external.get("adapter", "")).strip().lower()
    if adapter == "torchvision_resnet34_imagenet1k_v1":
        return load_torchvision_resnet34_imagenet1k_v1_source(external)
    if adapter == "upm_main_best":
        from lnl_toolbox.training.experiment import build_model

        model = build_model(dict(extractor_config.get("model", {})), num_classes)
        return load_upm_main_best_feature_source(
            external, model, num_classes=num_classes
        )
    raise ValueError(f"unsupported DLD feature adapter: {adapter}")


__all__ = [
    "DLDSourceFileIdentity",
    "DLDUPMMainBestSource",
    "DLDTorchvisionResNet34Source",
    "load_dld_feature_source",
    "load_torchvision_resnet34_imagenet1k_v1_source",
    "load_upm_main_best_feature_source",
]
