from __future__ import annotations

"""Experiment assembly for the first multiclass PCSE workflow."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import yaml

from lnl_toolbox.algorithms.pcse import PCSEAlgorithm, PCSEConfig
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.noise.generators import generate_pairflip, generate_symmetric
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.experiment import (
    _environment,
    bind_model_input,
    build_model,
    build_optimizer,
    build_scheduler,
)
from lnl_toolbox.training.data_service import prepare_experiment_data
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    effective_subset_actual_rate,
    file_sha256,
)


class _PCSEMultilayerPerceptron(nn.Module):
    """Small method-local model exposing two named hidden representations."""

    def __init__(
        self,
        dimension: int,
        hidden_width: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(dimension, hidden_width),
            nn.ReLU(),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_width, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.hidden2(self.hidden1(inputs)))


def _build_pcse_pretraining_model(
    model_config: Mapping[str, Any], input_spec: Any, num_classes: int
) -> nn.Module:
    """Build the internal PCSE pretraining model from prepared input facts."""

    bound_model_config = bind_model_input(model_config, input_spec)
    if str(model_config.get("name", "")).strip().lower() == "pcse_mlp":
        dimension = input_spec.feature_dim
        if dimension is None:
            raise ValueError(
                "PCSE pcse_mlp pretraining requires tabular input with "
                "feature_dim"
            )
        return _PCSEMultilayerPerceptron(
            dimension,
            int(model_config.get("hidden_width", 16)),
            num_classes,
        )
    return build_model(bound_model_config, num_classes)


def _resolve_run_dir(
    config: Mapping[str, Any],
    output_dir: str | Path | None,
    resume: str | Path | None,
) -> Path:
    if resume is not None:
        result = Path(resume).resolve().parent
    elif output_dir is not None:
        result = Path(output_dir).expanduser().resolve()
    else:
        result = (
            Path(config.get("output_root", "artifacts/runs"))
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        ).resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result


def _noisy_targets(
    manifest: NoiseManifest, indices: np.ndarray
) -> np.ndarray:
    positions = {
        int(index): position
        for position, index in enumerate(manifest.global_indices.tolist())
    }
    try:
        rows = np.asarray(
            [positions[int(index)] for index in indices], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError(
            "PCSE noise manifest is missing a stable sample index"
        ) from exc
    return manifest.noisy_targets[rows]




def _prepare_synthetic_manifest(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    clean_targets: np.ndarray,
    global_indices: np.ndarray,
    num_classes: int,
    resume_payload: Mapping[str, Any] | None,
) -> tuple[NoiseManifest, Path]:
    noise = config.get("noise")
    if not isinstance(noise, Mapping):
        raise TypeError("PCSE noise configuration must be a mapping")
    mode = str(noise.get("mode", "generated")).strip().lower()
    noise_type = str(noise.get("type", "")).strip().lower()
    if mode != "generated" or noise_type not in {"symmetric", "pairflip"}:
        raise ValueError(
            "PCSE synthetic first version requires generated symmetric or "
            "pairflip noise"
        )
    rate = float(noise["rate"])
    seed = int(noise["seed"])
    manifest_path = run_dir / str(
        noise.get("manifest_filename", "noise_manifest.npz")
    )
    if resume_payload is None:
        generator = (
            generate_symmetric if noise_type == "symmetric"
            else generate_pairflip
        )
        if noise_type == "symmetric":
            manifest = generator(
                clean_targets,
                num_classes,
                rate,
                seed,
                "synthetic_multiclass",
                sampling=str(noise.get("sampling", "per_class")),
                rng=str(noise.get("rng", "default_rng")),
            )
        else:
            manifest = generator(
                clean_targets,
                num_classes,
                rate,
                seed,
                "synthetic_multiclass",
            )
        manifest.global_indices = np.asarray(
            global_indices, dtype=np.int64
        ).copy()
        manifest.metadata = {**manifest.metadata, "source": "generated"}
        manifest.save(manifest_path)
    else:
        checkpoint_noise = resume_payload.get("noise")
        if not isinstance(checkpoint_noise, Mapping):
            raise ValueError("PCSE resume checkpoint has no noise metadata")
        if not manifest_path.is_file():
            raise FileNotFoundError("PCSE resume requires noise_manifest.npz")
        manifest = NoiseManifest.load(manifest_path)
        checks = {
            "mapping hash": checkpoint_noise.get("mapping_hash")
            == manifest.mapping_hash,
            "manifest SHA-256": checkpoint_noise.get("manifest_sha256")
            == file_sha256(manifest_path),
            "noise type": manifest.noise_type == noise_type,
            "requested rate": np.isclose(manifest.requested_rate, rate),
            "seed": manifest.seed == seed,
            "class count": manifest.num_classes == num_classes,
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            raise ValueError(
                "PCSE resume noise validation failed: " + ", ".join(failed)
            )
    if not np.array_equal(manifest.global_indices, global_indices):
        raise ValueError("PCSE manifest stable-index identity mismatch")
    if not np.array_equal(manifest.clean_targets, clean_targets):
        raise ValueError("PCSE manifest dataset identity mismatch")
    if manifest.num_classes != num_classes:
        raise ValueError("PCSE manifest class count mismatch")
    return manifest, manifest_path


def run_pcse_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *,
    requirements: DataRequirements | None = None,
) -> Path:
    """Run noisy pretraining, estimated transition and PCSE post-processing."""

    config = deepcopy(config)
    method_config = PCSEConfig.from_mapping(config)
    pretraining_stage = dict(config.get("pretraining_stage", {}))
    pretraining_stage["method"] = method_config.pretraining.method
    config["pretraining_stage"] = pretraining_stage
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    trainer = config.get("trainer", {})
    if not isinstance(trainer, Mapping):
        raise TypeError("PCSE trainer configuration must be a mapping")
    device = resolve_device(str(trainer.get("device", "cpu")))
    run_dir = _resolve_run_dir(config, output_dir, resume)

    resume_payload = None
    if resume is not None:
        resume_payload = read_checkpoint(resume, "cpu")
        if resume_payload.get("method") != "pcse":
            raise ValueError("Resume checkpoint is not a PCSE run")

    data = config.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("PCSE data configuration must be a mapping")
    loader = config.get("loader")
    if not isinstance(loader, Mapping):
        raise TypeError("PCSE loader configuration must be a mapping")
    model_config = method_config.pretraining.model
    if requirements is None:
        from lnl_toolbox.training.runners import resolve_data_requirements
        requirements = resolve_data_requirements(config, expected_runner="pcse")
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=seed + 10,
        checkpoint_payload=resume_payload,
    )
    prepared.loader_config = {**prepared.loader_config, "persistent_workers": True}
    data_name, num_classes = prepared.dataset, prepared.num_classes
    model = _build_pcse_pretraining_model(
        model_config, prepared.input_spec, num_classes
    )
    manifest, manifest_path = prepared.manifest, prepared.manifest_path
    if manifest is None or manifest_path is None:
        raise ValueError("PCSE requires noisy train and validation labels")
    manifest_mode = "generated"
    if manifest.noise_type == "clean":
        manifest_mode = "clean"
    elif manifest.noise_type == "real_world":
        manifest_mode = "native"
    effective_train_rate = effective_subset_actual_rate(manifest, prepared.train_indices)
    # Validation may be an independently namespaced source split.  Its
    # realized rate must be measured from the materialized role view rather
    # than by indexing the training-scoped noise manifest.
    # The training-scoped manifest can only describe a validation subset when
    # validation is carved from the same train split.  Independent validation
    # sources have their own namespace and must not be measured against the
    # train manifest (there is no meaningful manifest-aligned rate to report).
    effective_validation_rate = (
        effective_subset_actual_rate(manifest, prepared.validation_indices)
        if prepared.validation_uses_train_manifest
        else None
    )
    train_loader = prepared.loader(DataRole.TRAIN, generator_seed=seed + 101)
    statistics_loader = prepared.loader(DataRole.TRAIN, shuffle=False, generator_seed=seed + 102)
    validation_loader = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False, generator_seed=seed + 103)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False, generator_seed=seed + 104)

    optimizer = build_optimizer(
        model, method_config.pretraining.optimizer
    )
    scheduler = (
        build_scheduler(
            optimizer,
            method_config.pretraining.scheduler,
            method_config.pretraining.epochs,
        )
    )
    loss = build_builtin_loss({"name": "ce"}).to(device)

    noise_metadata = checkpoint_noise_metadata(
        manifest,
        manifest_path,
        run_dir,
        effective_train_rate,
        mode=manifest_mode,
        validation_targets="noisy",
        effective_validation_rate=effective_validation_rate,
    )
    config["noise"] = {**dict(config["noise"]), **noise_metadata}
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(seed, device), indent=2), encoding="utf-8"
    )
    (run_dir / "noise_summary.json").write_text(
        json.dumps(noise_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    algorithm = PCSEAlgorithm(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        train_loader=train_loader,
        statistics_loader=statistics_loader,
        noisy_validation_loader=validation_loader,
        clean_test_loader=test_loader,
        device=device,
        run_dir=run_dir,
        config=config,
        dataset=data_name,
        num_classes=num_classes,
        noise_metadata=noise_metadata,
    )
    try:
        if resume is not None:
            algorithm.resume(resume)
        algorithm.run()
    finally:
        algorithm.close()
    return run_dir
