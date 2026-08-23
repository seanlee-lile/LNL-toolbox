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

    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
            validation_targets="noisy",
        ),
        run_dir=run_dir,
        seed=method_config.seed,
    )
    data_name = prepared.dataset
    manifest = prepared.manifest
    if manifest is None:
        raise ValueError("importance reweighting requires binary asymmetric noise")
    positive_rows = manifest.clean_targets == 1
    negative_rows = manifest.clean_targets == 0
    realized_rho_positive = float(np.mean(
        manifest.noisy_targets[positive_rows] == 0
    ))
    realized_rho_negative = float(np.mean(
        manifest.noisy_targets[negative_rows] == 1
    ))

    def train_loader_factory(epoch: int):
        return prepared.loader(DataRole.TRAIN, epoch=int(epoch), stream=1000)

    validation_loader = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False, stream=2000)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False, stream=3000)
    train_dataset = prepared.dataset_for(DataRole.TRAIN)
    posterior_inputs = np.stack([
        np.asarray(train_dataset[index]["input"], dtype=np.float32)
        for index in range(len(train_dataset))
    ])
    posterior_targets = np.asarray([
        int(train_dataset[index]["target"]) for index in range(len(train_dataset))
    ], dtype=np.int64)
    posterior_indices = prepared.train_indices.copy()
    dimension = int(posterior_inputs.shape[1])

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
