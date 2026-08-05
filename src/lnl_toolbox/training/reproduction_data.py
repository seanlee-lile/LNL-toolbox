from __future__ import annotations

"""Small reusable data/model assembly for dedicated paper workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lnl_toolbox.data import NoisyTargetDataset
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.multi_view import IndexedMultiViewCifarDataset, build_strong_cifar_transform
from lnl_toolbox.data.multiclass_synthetic import MulticlassTensorDataset, generate_synthetic_multiclass
from lnl_toolbox.data.torch_cifar import TorchCifarDataset, build_cifar_transform, train_validation_split
from lnl_toolbox.noise.generators import generate_pairflip, generate_symmetric
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.training.experiment import build_model


class FeatureMLP(nn.Module):
    def __init__(self, dimension: int, hidden_width: int, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(dimension, hidden_width), nn.ReLU())
        self.classifier = nn.Linear(hidden_width, classes)

    def forward_with_features(self, inputs: torch.Tensor):
        features = self.features(inputs)
        return self.classifier(features), features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(inputs)[0]


@dataclass(slots=True)
class PreparedNoisyClassification:
    train_loader: DataLoader
    snapshot_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    dataset: str
    train_indices: np.ndarray
    noisy_targets: np.ndarray
    manifest: NoiseManifest


def _loader(dataset, config: Mapping[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 128)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        drop_last=bool(config.get("drop_last", False)) if shuffle else False,
        generator=torch.Generator().manual_seed(seed),
    )


def _cifar_transform(data: Mapping[str, Any], *, training: bool):
    normalization = data.get("normalization", {}) or {}
    return build_cifar_transform(
        training,
        bool(data.get("augment", True)) if training else False,
        normalization_mean=normalization.get("mean"),
        normalization_std=normalization.get("std"),
    )


def _manifest(
    clean: np.ndarray,
    indices: np.ndarray,
    classes: int,
    noise: Mapping[str, Any],
    dataset: str,
) -> NoiseManifest:
    kind = str(noise.get("type", "symmetric")).lower()
    rate = float(noise.get("rate", 0.0))
    seed = int(noise.get("seed", 1))
    if kind == "symmetric":
        result = generate_symmetric(
            clean, classes, rate, seed, dataset,
            sampling=str(noise.get("sampling", "per_class")),
            rng=str(noise.get("rng", "default_rng")),
        )
    elif kind == "pairflip":
        result = generate_pairflip(clean, classes, rate, seed, dataset)
    elif kind == "external_torch":
        path = noise.get("path")
        if not path:
            raise ValueError("external_torch noise requires path")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError("external_torch noise file must contain a mapping")
        clean_key = str(noise.get("clean_key", "clean_label_train"))
        noisy_key = str(noise.get("noisy_key", "noise_label_train"))
        external_clean = np.asarray(payload[clean_key], dtype=np.int64)
        external_noisy = np.asarray(payload[noisy_key], dtype=np.int64)
        if external_clean.shape != external_noisy.shape or external_clean.ndim != 1:
            raise ValueError("external clean/noisy labels must be aligned vectors")
        if indices.size and external_clean.size > int(indices.max()):
            selected_clean = external_clean[indices]
            selected_noisy = external_noisy[indices]
        elif external_clean.size == clean.size:
            selected_clean, selected_noisy = external_clean, external_noisy
        else:
            raise ValueError("external noise labels do not cover training indices")
        if not np.array_equal(selected_clean, clean):
            raise ValueError("external noise clean labels do not match the dataset")
        realized = float(np.mean(selected_clean != selected_noisy))
        result = NoiseManifest(
            dataset, "external_instance_dependent", seed, realized,
            selected_clean, selected_noisy,
            metadata={"source": str(path), "requested_rate": rate},
        )
    else:
        raise ValueError(
            "dedicated workflow noise must be symmetric, pairflip, or external_torch"
        )
    result.global_indices = indices.astype(np.int64, copy=True)
    return result


def prepare_noisy_classification(
    config: Mapping[str, Any], run_dir: Path, seed: int
) -> PreparedNoisyClassification:
    data = dict(config["data"])
    loader_config = dict(config.get("loader", {}))
    name = str(data["name"]).lower()
    if name == "synthetic_multiclass":
        classes = int(data["num_classes"])
        dimension = int(data["dimension"])
        train = generate_synthetic_multiclass(int(data["train_size"]), dimension, classes, seed + 1, start_index=0, split="train")
        validation = generate_synthetic_multiclass(int(data["validation_size"]), dimension, classes, seed + 2, start_index=len(train.labels), split="validation")
        test = generate_synthetic_multiclass(int(data["test_size"]), dimension, classes, seed + 3, start_index=len(train.labels) + len(validation.labels), split="test")
        manifest = _manifest(train.labels, train.global_indices, classes, dict(config["noise"]), name)
        train_set = MulticlassTensorDataset(train, manifest.noisy_targets)
        snapshot_set = MulticlassTensorDataset(train, manifest.noisy_targets)
        validation_set = MulticlassTensorDataset(validation)
        test_set = MulticlassTensorDataset(test)
        indices = train.global_indices
    elif name in {"cifar10", "cifar100"}:
        corpus = load_cifar10(data.get("root"), "train") if name == "cifar10" else load_cifar100(data.get("root"), "train")
        test_corpus = load_cifar10(data.get("root"), "test") if name == "cifar10" else load_cifar100(data.get("root"), "test")
        classes = 10 if name == "cifar10" else 100
        validation_size = int(data.get("validation_size", 5000))
        if validation_size == 0:
            train_indices = np.arange(len(corpus.labels), dtype=np.int64)
            validation_indices = np.empty(0, dtype=np.int64)
        else:
            train_indices, validation_indices = train_validation_split(
                corpus.labels, validation_size, seed,
                strategy=str(data.get("split_strategy", "stratified")),
                rng=str(data.get("split_rng", "default_rng")),
            )
        maximum = data.get("max_train_samples")
        if maximum is not None and int(maximum) < train_indices.size:
            random = np.random.default_rng(seed + 9)
            selected = []
            per_class = max(1, int(maximum) // classes)
            for class_index in range(classes):
                candidates = train_indices[corpus.labels[train_indices] == class_index].copy()
                random.shuffle(candidates)
                selected.append(candidates[:per_class])
            train_indices = np.sort(np.concatenate(selected))
        manifest = _manifest(corpus.labels[train_indices], train_indices, classes, dict(config["noise"]), name)
        train_base = TorchCifarDataset(
            corpus, train_indices, transform=_cifar_transform(data, training=True)
        )
        snapshot_base = TorchCifarDataset(
            corpus, train_indices, transform=_cifar_transform(data, training=False)
        )
        if bool(data.get("strong_augment", False)):
            target_map = {
                int(index): int(target)
                for index, target in zip(train_indices, manifest.noisy_targets)
            }
            train_set = IndexedMultiViewCifarDataset(
                corpus, train_indices,
                weak_transform=_cifar_transform(data, training=True),
                strong_transform=build_strong_cifar_transform(
                    magnitude=int(data.get("strong_magnitude", 10))
                ),
                targets_by_index=target_map,
            )
        else:
            train_set = NoisyTargetDataset(train_base, train_indices, manifest.noisy_targets)
        snapshot_set = NoisyTargetDataset(snapshot_base, train_indices, manifest.noisy_targets)
        test_set = TorchCifarDataset(
            test_corpus, transform=_cifar_transform(data, training=False)
        )
        validation_set = (
            TorchCifarDataset(
                corpus,
                validation_indices,
                transform=_cifar_transform(data, training=False),
            )
            if validation_indices.size else test_set
        )
        indices = train_indices
    else:
        raise ValueError("dedicated workflow supports synthetic_multiclass or CIFAR")
    manifest_path = run_dir / "noise_manifest.npz"
    if not manifest_path.exists():
        manifest.save(manifest_path)
    return PreparedNoisyClassification(
        _loader(train_set, loader_config, shuffle=True, seed=seed + 21),
        _loader(snapshot_set, loader_config, shuffle=False, seed=seed + 22),
        _loader(validation_set, loader_config, shuffle=False, seed=seed + 23),
        _loader(test_set, loader_config, shuffle=False, seed=seed + 24),
        classes, name, indices, manifest.noisy_targets.copy(), manifest,
    )


def build_reproduction_model(config: Mapping[str, Any], data: Mapping[str, Any], classes: int) -> nn.Module:
    if str(data["name"]).lower() == "synthetic_multiclass":
        return FeatureMLP(int(data["dimension"]), int(config.get("hidden_width", 16)), classes)
    name = str(config.get("name", "")).lower()
    if name == "mc_ldce_cnn":
        from lnl_toolbox.models.mc_ldce_cnn import MCLDCECifarCNN
        return MCLDCECifarCNN(classes)
    if name == "ca2c_seven_cnn":
        from lnl_toolbox.models.ca2c_cnn import CA2CSevenCNN
        return CA2CSevenCNN(classes)
    return build_model(config, classes)


__all__ = ["FeatureMLP", "PreparedNoisyClassification", "build_reproduction_model", "prepare_noisy_classification"]
