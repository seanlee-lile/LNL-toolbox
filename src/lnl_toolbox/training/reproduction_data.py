from __future__ import annotations

"""Small reusable data/model assembly for dedicated paper workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.training.experiment import build_model
from lnl_toolbox.training.data_service import prepare_experiment_data


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


def prepare_noisy_classification(
    config: Mapping[str, Any], run_dir: Path, seed: int
) -> PreparedNoisyClassification:
    views = ("weak", "strong") if bool(config["data"].get("strong_augment", False)) else ("weak",)
    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({
                DataRole.TRAIN,
                DataRole.TRAIN_EVAL,
                DataRole.CLEAN_VALIDATION,
                DataRole.TEST,
            }),
            views=views,
            validation_targets="clean",
        ),
        run_dir=run_dir,
        seed=seed,
    )
    if prepared.manifest is None:
        clean = prepared.train_split.clean_targets
        if clean is None:
            raise ValueError("noisy classification requires observed or generated noisy labels")
        manifest = NoiseManifest(
            prepared.dataset,
            "clean",
            seed,
            0.0,
            clean,
            clean,
            global_indices=prepared.train_split.global_indices,
            num_classes=prepared.num_classes,
        )
    else:
        manifest = prepared.manifest
    return PreparedNoisyClassification(
        prepared.loader(DataRole.TRAIN, stream=21),
        prepared.loader(DataRole.TRAIN_EVAL, stream=22, shuffle=False),
        prepared.loader(DataRole.CLEAN_VALIDATION, stream=23, shuffle=False),
        prepared.loader(DataRole.TEST, stream=24, shuffle=False),
        prepared.num_classes,
        prepared.dataset,
        prepared.train_indices.copy(),
        prepared.noisy_targets,
        manifest,
    )


def build_reproduction_model(config: Mapping[str, Any], data: Mapping[str, Any], classes: int) -> nn.Module:
    if str(data["name"]).lower() == "synthetic_multiclass":
        return FeatureMLP(int(data["dimension"]), int(config.get("hidden_width", 16)), classes)
    name = str(config.get("name", "")).lower()
    if name == "mc_ldce_cnn":
        from lnl_toolbox.models.mc_ldce_cnn import MCLDCECifarCNN
        return MCLDCECifarCNN(classes)
    if name == "l2rw_resnet32":
        from lnl_toolbox.models.cifar_resnet import l2rw_resnet32
        return l2rw_resnet32(classes, int(config.get("base_width", 16)))
    if name == "ca2c_seven_cnn":
        from lnl_toolbox.models.ca2c_cnn import CA2CSevenCNN
        return CA2CSevenCNN(classes)
    return build_model(config, classes)


__all__ = ["FeatureMLP", "PreparedNoisyClassification", "build_reproduction_model", "prepare_noisy_classification"]
