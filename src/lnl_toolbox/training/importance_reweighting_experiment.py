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
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.data_service import prepare_experiment_data
from lnl_toolbox.training.experiment import _environment, build_optimizer, build_scheduler
from lnl_toolbox.training.snapshots import collect_feature_snapshot


def run_importance_reweighting_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *,
    requirements: DataRequirements | None = None,
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
    if requirements is None:
        from lnl_toolbox.training.runners import resolve_data_requirements

        requirements = resolve_data_requirements(
            config, expected_runner="importance_reweighting"
        )

    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=method_config.seed,
    )
    data_name = prepared.dataset
    manifest = prepared.manifest
    if manifest is None:
        raise ValueError("importance reweighting requires binary asymmetric noise")
    realized_rho_positive = float(method_config.noise["rho_positive"])
    realized_rho_negative = float(method_config.noise["rho_negative"])

    def train_loader_factory(epoch: int):
        return prepared.loader(DataRole.TRAIN, epoch=int(epoch), stream=1000)

    validation_loader = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False, stream=2000)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False, stream=3000)
    feature_snapshot = collect_feature_snapshot(
        nn.Flatten(start_dim=1),
        prepared.loader(DataRole.TRAIN, shuffle=False, stream=4000),
        "cpu",
        dataset=prepared.dataset,
        split="train",
    )
    posterior_inputs = feature_snapshot.features.astype(np.float32, copy=False)
    posterior_targets = feature_snapshot.noisy_targets
    posterior_indices = feature_snapshot.global_indices
    dimension = int(posterior_inputs.shape[1])

    def model_factory() -> nn.Module:
        return nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(dimension, 2))

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
        "data_fingerprint": prepared.data_fingerprint,
        "realized_rho_positive": realized_rho_positive,
        "realized_rho_negative": realized_rho_negative,
    }
    for artifact_name, hash_name in (
        ("preprocessing_state.json", "preprocessing_state_hash"),
        ("split_manifest.json", "split_hash"),
    ):
        artifact_path = run_dir / artifact_name
        if artifact_path.is_file():
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            manifest_identity[hash_name] = artifact.get(
                "state_hash" if artifact_name.startswith("preprocessing") else "split_hash",
                "",
            )
    algorithm = ImportanceReweightingAlgorithm(
        config=config,
        run_dir=run_dir,
        manifest_identity=manifest_identity,
        posterior_features=posterior_inputs,
        posterior_targets=posterior_targets,
        posterior_indices=posterior_indices,
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
            "data_fingerprint": prepared.data_fingerprint,
            "realized_rho_positive": realized_rho_positive,
            "realized_rho_negative": realized_rho_negative,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result = algorithm.run()
    return result
