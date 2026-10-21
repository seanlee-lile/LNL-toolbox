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
from lnl_toolbox.training.experiment import _environment, bind_model_input, build_model, build_optimizer, build_scheduler
from lnl_toolbox.training.snapshots import collect_feature_snapshot, collect_posterior_snapshot, pretrain_noisy_classifier


def _realized_noise_rates(manifest) -> tuple[float | None, float | None]:
    """Diagnostic class-conditional flip rates; never used for training weights."""
    if manifest.clean_targets is None:
        return None, None
    clean = np.asarray(manifest.clean_targets)
    noisy = np.asarray(manifest.noisy_targets)
    rates = []
    for label in (1, 0):
        mask = clean == label
        rates.append(float(np.mean(noisy[mask] != label)) if mask.any() else None)
    return rates[0], rates[1]


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
    prepared.loader_config = {**prepared.loader_config, "persistent_workers": True}
    train_loader = prepared.loader(DataRole.TRAIN, stream=1000)
    data_name = prepared.dataset
    manifest = prepared.manifest
    if manifest is None:
        raise ValueError("importance reweighting requires binary asymmetric noise")
    realized_rho_positive, realized_rho_negative = _realized_noise_rates(manifest)

    def train_loader_factory(epoch: int):
        train_loader.generator.manual_seed(prepared.seed + int(epoch) + 1000)
        return train_loader

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
    if dimension != int(method_config.data["dimension"]):
        raise ValueError(f"data.dimension={method_config.data['dimension']} does not match actual input dimension {dimension}")

    def model_factory() -> nn.Module:
        if str(method_config.model.get("name", "linear")) == "linear":
            return nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(dimension, 2))
        return build_model(bind_model_input(method_config.model, prepared.input_spec), 2)

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
    }
    # Missing observations must not add non-reflexive NaN checkpoint identity.
    for name, value in (("realized_rho_positive", realized_rho_positive),
                        ("realized_rho_negative", realized_rho_negative)):
        if value is not None:
            manifest_identity[name] = value
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
    posterior_backend = None
    if method_config.posterior_stage["name"] == "neural":
        from lnl_toolbox.algorithms.importance_reweighting.estimation import SnapshotBinaryNoisyPosteriorEstimator
        from lnl_toolbox.noise.estimators import PosteriorSnapshot

        stage = dict(method_config.posterior_stage)
        stage_dir = run_dir / "posterior_stage"
        stage_dir.mkdir(exist_ok=True)
        stage_model_config = dict(stage.get("model", method_config.model))
        stage_optimizer_config = dict(stage.get("optimizer", method_config.optimizer))
        stage_scheduler_config = dict(stage.get("scheduler", {"name": "none"}))
        identity = {"stage": stage, "model": stage_model_config,
                    "optimizer": stage_optimizer_config, "scheduler": stage_scheduler_config,
                    "seed": method_config.seed, "manifest": manifest_identity}
        if stage_model_config.get("name", "linear") == "linear":
            stage_model = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(dimension, 2))
        else:
            stage_model = build_model(bind_model_input(stage_model_config, prepared.input_spec), 2)
        stage_model.to(device)
        stage_optimizer = build_optimizer(stage_model, stage_optimizer_config)
        summary = pretrain_noisy_classifier(
            stage_model, stage_optimizer, train_loader_factory, device,
            epochs=int(stage["epochs"]), validation_loader=validation_loader,
            scheduler=build_scheduler(stage_optimizer, stage_scheduler_config, int(stage["epochs"])),
            checkpoint_dir=stage_dir, checkpoint_identity=identity,
            early_stopping=stage.get("early_stopping", False),
        )
        snapshot_path = stage_dir / "posterior_snapshot.npz"
        # Reuse the exact persisted snapshot on resume; do not change backend identity.
        if snapshot_path.is_file():
            snapshot = PosteriorSnapshot.load(snapshot_path)
        else:
            snapshot = collect_posterior_snapshot(
                stage_model, prepared.loader(DataRole.TRAIN, shuffle=False, stream=4000),
                device, dataset=data_name, split="train")
            snapshot.save(snapshot_path)
        (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        posterior_backend = SnapshotBinaryNoisyPosteriorEstimator(snapshot, identity)
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
        posterior_backend=posterior_backend,
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
    # Publish JSON null (unknown) instead of legacy diagnostic NaN values.
    metrics_path = run_dir / "final_metrics.json"
    if metrics_path.is_file() and (realized_rho_positive is None or realized_rho_negative is None):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["noise_rates"]["realized_rho_positive"] = realized_rho_positive
        metrics["noise_rates"]["realized_rho_negative"] = realized_rho_negative
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return result
