from __future__ import annotations

"""Experiment assembly for binary asymmetric-RCN Importance Reweighting."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml

from lnl_toolbox.algorithms.importance_reweighting import (
    ImportanceReweightingAlgorithm,
    ImportanceReweightingConfig,
)
from lnl_toolbox.data.binary_synthetic import (
    BinaryTensorDataset,
    generate_synthetic_binary_2d,
    generate_synthetic_binary_high_dim,
)
from lnl_toolbox.noise.binary_rcn import (
    generate_binary_asymmetric_rcn,
    validate_binary_rcn_manifest,
)
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.experiment import (
    _environment,
    _loader,
    build_optimizer,
    build_scheduler,
)


def run_importance_reweighting_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Run the paper-scoped binary KDE/raw-min/weighted-CE workflow."""

    config = deepcopy(config)
    method_config = ImportanceReweightingConfig.from_mapping(config)
    seed_everything(method_config.seed)
    device = resolve_device(str(method_config.trainer.get("device", "cpu")))

    if resume is not None:
        run_dir = Path(resume).resolve().parent
    elif output_dir is not None:
        run_dir = Path(output_dir).resolve()
    else:
        run_dir = Path(config.get("output_root", "artifacts/runs")) / (
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    sizes = method_config.data
    data_name = str(sizes["name"]).strip().lower()
    dimension = int(sizes["dimension"])
    generator = (
        generate_synthetic_binary_2d
        if data_name == "synthetic_binary_2d"
        else generate_synthetic_binary_high_dim
    )
    dimension_args = () if data_name == "synthetic_binary_2d" else (dimension,)
    train = generator(
        int(sizes["train_size"]), *dimension_args, method_config.seed + 11,
        start_index=0, split="train",
    )
    validation = generator(
        int(sizes["validation_size"]), *dimension_args, method_config.seed + 12,
        start_index=len(train.labels), split="validation",
    )
    test = generator(
        int(sizes["test_size"]), *dimension_args, method_config.seed + 13,
        start_index=len(train.labels) + len(validation.labels), split="test",
    )
    population_clean = np.concatenate((train.labels, validation.labels))
    population_indices = np.concatenate((
        train.global_indices, validation.global_indices
    ))
    manifest_path = run_dir / "noise_manifest.npz"
    if resume is None:
        manifest = generate_binary_asymmetric_rcn(
            population_clean,
            population_indices,
            rho_positive=float(method_config.noise["rho_positive"]),
            rho_negative=float(method_config.noise["rho_negative"]),
            seed=int(method_config.noise.get("seed", method_config.seed + 21)),
        )
        manifest.save(manifest_path)
    else:
        if not manifest_path.exists():
            raise FileNotFoundError("resume requires noise_manifest.npz")
        manifest = NoiseManifest.load(manifest_path)
    validate_binary_rcn_manifest(
        manifest,
        expected_indices=population_indices,
        rho_positive=float(method_config.noise["rho_positive"]),
        rho_negative=float(method_config.noise["rho_negative"]),
    )
    if not np.array_equal(manifest.clean_targets, population_clean):
        raise ValueError("importance reweighting manifest clean-label identity mismatch")

    position = {
        int(index): row
        for row, index in enumerate(manifest.global_indices.tolist())
    }
    def noisy_targets(indices: np.ndarray) -> np.ndarray:
        try:
            rows = np.asarray([position[int(index)] for index in indices])
        except KeyError as exc:
            raise ValueError("noise manifest is missing a stable sample index") from exc
        return manifest.noisy_targets[rows]

    train_noisy = noisy_targets(train.global_indices)
    validation_noisy = noisy_targets(validation.global_indices)
    train_dataset = BinaryTensorDataset(train, train_noisy)
    validation_dataset = BinaryTensorDataset(validation, validation_noisy)
    test_dataset = BinaryTensorDataset(test)
    loader_config = method_config.loader

    def train_loader_factory(epoch: int):
        return _loader(
            train_dataset,
            loader_config,
            shuffle=True,
            seed=method_config.seed + 1000 + int(epoch),
        )

    validation_loader = _loader(
        validation_dataset, loader_config, shuffle=False,
        seed=method_config.seed + 2000,
    )
    test_loader = _loader(
        test_dataset, loader_config, shuffle=False,
        seed=method_config.seed + 3000,
    )

    def model_factory() -> nn.Module:
        model = nn.Linear(dimension, 2)
        if model.out_features != 2:
            raise ValueError("importance reweighting model must output two logits")
        return model

    def optimizer_factory(model: nn.Module):
        return build_optimizer(model, method_config.optimizer)

    def scheduler_factory(optimizer):
        return build_scheduler(
            optimizer, method_config.scheduler, method_config.epochs
        )

    manifest_identity = {
        "dataset": manifest.dataset,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "mapping_hash": manifest.mapping_hash,
        "noise_type": manifest.noise_type,
        "num_classes": manifest.num_classes,
        "label_convention": manifest.metadata.get("label_convention"),
    }
    algorithm = ImportanceReweightingAlgorithm(
        config=config,
        run_dir=run_dir,
        manifest_identity=manifest_identity,
        posterior_features=train.features,
        posterior_targets=train_noisy,
        posterior_indices=train.global_indices,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        train_loader_factory=train_loader_factory,
        validation_loader=validation_loader,
        test_loader=test_loader,
        loss=build_builtin_loss(method_config.loss),
        device=device,
    )
    if resume is not None:
        algorithm.resume(resume)

    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(method_config.seed, device), indent=2),
        encoding="utf-8",
    )
    (run_dir / "noise_summary.json").write_text(
        json.dumps({
            "rho_positive": float(method_config.noise["rho_positive"]),
            "rho_negative": float(method_config.noise["rho_negative"]),
            "manifest_actual_rate": manifest.actual_rate,
            "mapping_hash": manifest.mapping_hash,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return algorithm.run()
