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
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.data.multiclass_synthetic import (
    MulticlassTensorDataset,
    generate_synthetic_multiclass,
)
from lnl_toolbox.data.cifar import load_cifar10
from lnl_toolbox.data.noisy_dataset import NoisyTargetDataset
from lnl_toolbox.data.torch_cifar import (
    TorchCifarDataset,
    build_cifar_transform,
    train_validation_split,
)
from lnl_toolbox.noise.generators import generate_pairflip, generate_symmetric
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import read_checkpoint
from lnl_toolbox.training.experiment import (
    _environment,
    _loader,
    build_model,
    build_optimizer,
    build_scheduler,
)
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    effective_subset_actual_rate,
    file_sha256,
)
from lnl_toolbox.training.pcse_pretrained import load_upm_main_best_source


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


def _persist_source_manifest(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"PCSE temporary source manifest already exists: {temporary}"
        )
    try:
        temporary.write_bytes(source.read_bytes())
        loaded = NoiseManifest.load(temporary)
        if loaded.mapping_hash != NoiseManifest.load(source).mapping_hash:
            raise ValueError("PCSE source manifest temporary validation failed")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


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
) -> Path:
    """Run noisy pretraining, estimated transition and PCSE post-processing."""

    config = deepcopy(config)
    method_config = PCSEConfig.from_mapping(config)
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
    data_name = str(data.get("name", "")).strip().lower()
    if data_name not in {"synthetic_multiclass", "cifar10"}:
        raise ValueError("PCSE runner supports synthetic_multiclass and cifar10")
    num_classes = int(data.get("num_classes", 0))
    if num_classes < 3:
        raise ValueError("PCSE requires at least three classes")
    loader = config.get("loader")
    if not isinstance(loader, Mapping):
        raise TypeError("PCSE loader configuration must be a mapping")
    model_config = method_config.pretraining.model
    external_source = None
    source_model = None
    if data_name == "synthetic_multiclass":
        if method_config.pretraining.mode != "train":
            raise ValueError("synthetic PCSE requires pretraining mode train")
        dimension = int(data.get("dimension", 0))
        train = generate_synthetic_multiclass(
            int(data["train_size"]), dimension, num_classes, seed + 11,
            start_index=0, split="train",
        )
        validation = generate_synthetic_multiclass(
            int(data["validation_size"]), dimension, num_classes, seed + 12,
            start_index=len(train.labels), split="validation",
        )
        test = generate_synthetic_multiclass(
            int(data["test_size"]), dimension, num_classes, seed + 13,
            start_index=len(train.labels) + len(validation.labels), split="test",
        )
        population_clean = np.concatenate((train.labels, validation.labels))
        population_indices = np.concatenate(
            (train.global_indices, validation.global_indices)
        )
        manifest, manifest_path = _prepare_synthetic_manifest(
            config=config, run_dir=run_dir, clean_targets=population_clean,
            global_indices=population_indices, num_classes=num_classes,
            resume_payload=resume_payload,
        )
        train_observed = _noisy_targets(manifest, train.global_indices)
        validation_observed = _noisy_targets(manifest, validation.global_indices)
        train_set = MulticlassTensorDataset(train, train_observed)
        statistics_set = train_set
        validation_set = MulticlassTensorDataset(validation, validation_observed)
        clean_test_set = MulticlassTensorDataset(test)
        if str(model_config.get("name", "")).strip().lower() != "pcse_mlp":
            raise ValueError("PCSE synthetic runner requires model name pcse_mlp")
        model = _PCSEMultilayerPerceptron(
            dimension, int(model_config.get("hidden_width", 16)), num_classes
        )
        effective_train_rate = manifest.actual_rate
        effective_validation_rate = float(
            np.mean(validation_observed != validation.labels)
        )
        manifest_mode = "generated"
    else:
        if method_config.pretraining.mode != "external_checkpoint":
            raise ValueError(
                "PCSE CIFAR-10 requires pretraining mode external_checkpoint"
            )
        if num_classes != 10:
            raise ValueError("PCSE CIFAR-10 requires data.num_classes: 10")
        model = build_model(model_config, num_classes)
        external_source = load_upm_main_best_source(
            method_config.pretraining.source, model, num_classes=num_classes
        )
        source_model = build_model(model_config, num_classes)
        source_model.load_state_dict(external_source.state_dict, strict=True)
        source_model.to(device)
        train_data = load_cifar10(data.get("root"), "train")
        test_data = load_cifar10(data.get("root"), "test")
        train_indices, validation_indices = train_validation_split(
            train_data.labels,
            int(data["validation_size"]),
            seed,
            strategy=str(data.get("split_strategy", "stratified")),
            rng=str(data.get("split_rng", "default_rng")),
        )
        expected_indices = np.sort(
            np.concatenate((train_indices, validation_indices))
        )
        manifest = external_source.noise_manifest
        if not np.array_equal(manifest.global_indices, expected_indices):
            raise ValueError("PCSE source manifest stable-index mismatch")
        if not np.array_equal(
            manifest.clean_targets, train_data.labels[expected_indices]
        ):
            raise ValueError("PCSE source manifest clean-label identity mismatch")
        manifest_path = run_dir / "noise_manifest.npz"
        if resume_payload is None:
            _persist_source_manifest(
                Path(external_source.manifest.path), manifest_path
            )
        if not manifest_path.is_file() or file_sha256(manifest_path) != (
            external_source.manifest.sha256
        ):
            raise ValueError("PCSE run-local source manifest identity mismatch")
        transform = build_cifar_transform(False, preprocessing="standard")
        train_clean = TorchCifarDataset(
            train_data, train_indices, transform=transform
        )
        validation_clean = TorchCifarDataset(
            train_data, validation_indices, transform=transform
        )
        train_set = NoisyTargetDataset(
            train_clean, manifest.global_indices, manifest.noisy_targets
        )
        statistics_set = NoisyTargetDataset(
            TorchCifarDataset(train_data, train_indices, transform=transform),
            manifest.global_indices,
            manifest.noisy_targets,
        )
        validation_set = NoisyTargetDataset(
            validation_clean, manifest.global_indices, manifest.noisy_targets
        )
        clean_test_set = TorchCifarDataset(test_data, transform=transform)
        effective_train_rate = effective_subset_actual_rate(
            manifest, train_indices
        )
        effective_validation_rate = effective_subset_actual_rate(
            manifest, validation_indices
        )
        manifest_mode = "external"

    train_loader = _loader(train_set, loader, shuffle=True, seed=seed + 101)
    statistics_loader = _loader(
        statistics_set, loader, shuffle=False, seed=seed + 102
    )
    validation_loader = _loader(
        validation_set, loader, shuffle=False, seed=seed + 103
    )
    test_loader = _loader(clean_test_set, loader, shuffle=False, seed=seed + 104)

    optimizer = build_optimizer(
        model, method_config.pretraining.optimizer
    )
    scheduler = (
        build_scheduler(
            optimizer,
            method_config.pretraining.scheduler,
            method_config.pretraining.epochs,
        )
        if method_config.pretraining.mode == "train"
        else None
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
        external_source_provenance=(
            None if external_source is None else external_source.provenance
        ),
        external_source_model=source_model,
    )
    try:
        if resume is not None:
            algorithm.resume(resume)
        elif external_source is not None:
            validation = evaluate_classification(
                model, validation_loader, loss, device
            )
            algorithm.adopt_external_pretrained(
                completed_epochs=external_source.source_completed_epochs,
                global_step=external_source.source_global_step,
                best_epoch=external_source.best_epoch,
                validation_accuracy=external_source.best_validation_accuracy,
                validation_loss=float(validation["loss"]),
            )
        algorithm.run()
    finally:
        algorithm.close()
        if external_source is not None:
            external_source.assert_unchanged()
    return run_dir
