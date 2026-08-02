from __future__ import annotations

"""Unified clean/noisy supervised CIFAR experiment runner."""

from copy import deepcopy
from datetime import datetime
import json
import math
import platform
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.data import NoisyTargetDataset
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.torch_cifar import (
    TorchCifarDataset,
    build_cifar_transform,
    cifar_pixel_mean,
    stratified_split,
    train_validation_split,
)
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.models.cifar_resnet import (
    cifar_resnet18,
    cifar_resnet34,
    cifar_resnet50,
    preact_resnet18,
)
from lnl_toolbox.models.cifar_cnn import CifarCnn8
from lnl_toolbox.models.tiny_cnn import TinyCNN
from lnl_toolbox.plugins.builtin import (
    build_builtin_pipeline,
    build_builtin_loss,
    build_builtin_parameter_update_policy,
    build_builtin_selector,
)
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from lnl_toolbox.training.early_stopping import EarlyStopping
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    effective_subset_actual_rate,
    noise_mode,
    prepare_noise_manifest,
)
from lnl_toolbox.training.progress import (
    TerminalTrainingProgress,
    write_training_curves_svg,
)


def build_model(config: Mapping[str, Any], num_classes: int) -> nn.Module:
    name = str(config.get("name", "preact_resnet18")).lower()
    if name == "tiny_cnn":
        return TinyCNN(num_classes, int(config.get("width", 64)))
    if name == "cifar_cnn8":
        return CifarCnn8(num_classes)
    if name == "resnet18":
        return cifar_resnet18(num_classes, int(config.get("base_width", 64)))
    if name == "resnet34":
        return cifar_resnet34(num_classes, int(config.get("base_width", 64)))
    if name == "resnet50":
        return cifar_resnet50(
            num_classes,
            int(config.get("base_width", 64)),
            stem_padding=int(config.get("stem_padding", 1)),
            initialization=str(config.get("initialization", "kaiming")),
        )
    if name == "preact_resnet18":
        return preact_resnet18(num_classes, int(config.get("base_width", 64)))
    raise ValueError(f"Unsupported model: {name}")


def build_optimizer(model: nn.Module, config: Mapping[str, Any]):
    name = str(config.get("name", "sgd")).lower()
    common = {
        "lr": float(config["lr"]),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            momentum=float(config.get("momentum", 0.9)),
            nesterov=bool(config.get("nesterov", False)),
            **common,
        )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **common)
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            betas=(
                float(config.get("beta1", 0.9)),
                float(config.get("beta2", 0.999)),
            ),
            eps=float(config.get("eps", 1e-8)),
            **common,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, config: Mapping[str, Any] | None, epochs: int):
    if not config or str(config.get("name", "none")).lower() == "none":
        return None
    name = str(config["name"]).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", epochs)),
            eta_min=float(config.get("eta_min", 0.0)),
        )
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(value) for value in config["milestones"]],
            gamma=float(config.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def _subset(indices: np.ndarray, labels: np.ndarray, size: int | None, seed: int) -> np.ndarray:
    if size is None or size >= len(indices):
        return indices
    _, selected = stratified_split(labels[indices], size, seed)
    return indices[selected]


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader(dataset, config: Mapping[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", True)),
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers else None,
        generator=torch.Generator().manual_seed(seed),
    )


def _environment(seed: int, device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "seed": seed,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def _normalized_resume_value(config: Mapping[str, Any], key: str) -> Any:
    if key == "loss":
        return dict(config.get("loss", {"name": "ce"}))
    if key == "scheduler":
        return dict(config.get("scheduler", {"name": "none"}) or {"name": "none"})
    if key == "selector":
        return dict(config.get("selector", {"name": "all"}) or {"name": "all"})
    if key == "parameter_update":
        return dict(
            config.get("parameter_update", {"name": "standard"})
            or {"name": "standard"}
        )
    if key == "pipeline":
        return dict(config.get("pipeline", {}) or {})
    if key == "early_stopping":
        return dict(config.get("early_stopping", {}) or {})
    return config.get(key)


def _validate_resume_config(current: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    for key in (
        "seed",
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "selector",
        "parameter_update",
        "pipeline",
        "early_stopping",
    ):
        if _normalized_resume_value(current, key) != _normalized_resume_value(saved, key):
            raise ValueError(f"Resume configuration changed {key}")
    current_dataset = str(current.get("data", {}).get("name", "cifar10")).lower()
    saved_dataset = str(saved.get("data", {}).get("name", "cifar10")).lower()
    if current_dataset != saved_dataset:
        raise ValueError("Resume configuration changed data.name")
    current_preprocessing = str(
        current.get("data", {}).get("preprocessing", "standard")
    ).lower()
    saved_preprocessing = str(
        saved.get("data", {}).get("preprocessing", "standard")
    ).lower()
    if current_preprocessing != saved_preprocessing:
        raise ValueError("Resume configuration changed data.preprocessing")
    for key, default in (
        (
            "validation_split",
            {"strategy": "stratified", "rng": "default_rng"},
        ),
        ("normalization", None),
    ):
        current_value = current.get("data", {}).get(key, default)
        saved_value = saved.get("data", {}).get(key, default)
        if current_value != saved_value:
            raise ValueError(f"Resume configuration changed data.{key}")
    current_validation_targets = str(
        current.get("noise", {}).get("validation_targets", "clean")
    ).lower()
    saved_validation_targets = str(
        saved.get("noise", {}).get("validation_targets", "clean")
    ).lower()
    if current_validation_targets != saved_validation_targets:
        raise ValueError("Resume configuration changed noise.validation_targets")
    current_selection = dict(current.get("evaluation", {}) or {}).get(
        "selection_split", "validation"
    )
    saved_selection = dict(saved.get("evaluation", {}) or {}).get(
        "selection_split", "validation"
    )
    if str(current_selection).lower() != str(saved_selection).lower():
        raise ValueError("Resume configuration changed evaluation.selection_split")


def _resolve_dss_epoch_contract(
    config: dict[str, Any],
    epochs: int,
) -> None:
    """Bind DSS history horizon to the final trainer epoch budget."""

    pipeline = config.get("pipeline")
    if not isinstance(pipeline, Mapping):
        return
    objective = pipeline.get("objective_consumer")
    if not isinstance(objective, Mapping):
        return
    if str(objective.get("name", "")).strip().lower() != "dss":
        return
    required_components = (
        ("loss", "ce"),
        ("selector", "all"),
        ("parameter_update", "standard"),
    )
    for section, required_name in required_components:
        section_config = config.get(section, {"name": required_name})
        if not isinstance(section_config, Mapping):
            raise TypeError(f"{section} configuration must be a mapping")
        actual_name = str(
            section_config.get("name", required_name)
        ).strip().lower()
        if actual_name != required_name:
            raise ValueError(
                f"DSS requires {section}.name={required_name!r}; "
                f"found {actual_name!r}"
            )
    configured = objective.get("total_epochs")
    if configured is not None and int(configured) != epochs:
        raise ValueError(
            "DSS total_epochs must equal the resolved trainer.epochs"
        )
    resolved_pipeline = dict(pipeline)
    resolved_objective = dict(objective)
    resolved_objective["total_epochs"] = epochs
    resolved_pipeline["objective_consumer"] = resolved_objective
    config["pipeline"] = resolved_pipeline


def _resolved_noise_config(
    original: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = dict(original)
    resolved.update({
        "mode": metadata["mode"],
        "run_manifest": metadata["manifest_path"],
        "run_manifest_sha256": metadata["manifest_sha256"],
        "mapping_hash": metadata["mapping_hash"],
        "actual_rate": metadata["manifest_actual_rate"],
        "effective_train_subset_actual_rate": metadata[
            "effective_train_subset_actual_rate"
        ],
        "validation_targets": metadata["validation_targets"],
        "effective_validation_subset_actual_rate": metadata[
            "effective_validation_subset_actual_rate"
        ],
    })
    if "source_manifest_sha256" in metadata:
        resolved["source_manifest_sha256"] = metadata["source_manifest_sha256"]
    return resolved


def _validate_supervised_config(config: Mapping[str, Any]) -> None:
    if "transition_estimator" in config:
        raise ValueError(
            "field 'transition_estimator' is not connected; use "
            "pipeline.transition_estimator"
        )
    pipeline_config = config.get("pipeline", {}) or {}
    if not isinstance(pipeline_config, Mapping):
        raise TypeError("pipeline configuration must be a mapping")
    early_stopping_config = config.get("early_stopping", {}) or {}
    if early_stopping_config not in (False, None) and not isinstance(early_stopping_config, Mapping):
        raise TypeError("early_stopping configuration must be a mapping or false")
    update_config = config.get("parameter_update", {"name": "standard"})
    if not isinstance(update_config, Mapping):
        raise TypeError("parameter_update configuration must be a mapping")
    noise_config = config.get("noise") or {}
    if not isinstance(noise_config, Mapping):
        raise TypeError("noise configuration must be a mapping")
    if str(update_config.get("name", "standard")).strip().lower() == "cdr":
        optimizer_config = config.get("optimizer")
        if not isinstance(optimizer_config, Mapping):
            raise TypeError("CDR optimizer configuration must be a mapping")
        optimizer_name = str(
            optimizer_config.get("name", "sgd")
        ).strip().lower()
        if optimizer_name != "sgd":
            raise ValueError("CDR requires optimizer.name='sgd'")
        compatibility_mode = str(
            update_config.get("compatibility_mode", "paper")
        ).strip().lower()
        if compatibility_mode not in {"paper", "official_code"}:
            raise ValueError(
                "CDR compatibility_mode must be 'paper' or 'official_code'"
            )
        if compatibility_mode == "paper":
            momentum = float(optimizer_config.get("momentum", 0.9))
            if momentum != 0.0:
                raise ValueError(
                    "CDR paper mode requires SGD momentum=0 so that "
                    "non-critical parameters receive only the explicit "
                    "L1 update"
                )
            weight_decay = float(optimizer_config.get("weight_decay", 0.0))
            if weight_decay != 0.0:
                raise ValueError(
                    "CDR paper mode requires optimizer weight_decay=0; "
                    "use parameter_update.l1_decay for Eq. (5)-(6)"
                )
        if "rate" in noise_config and noise_config["rate"] is not None:
            configured_noise_rate = float(noise_config["rate"])
            update_noise_rate = float(update_config["noise_rate"])
            if not math.isclose(
                configured_noise_rate,
                update_noise_rate,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "CDR noise-rate mismatch: "
                    f"noise.rate={configured_noise_rate} but "
                    "parameter_update.noise_rate="
                    f"{update_noise_rate}"
                )
    validation_targets = str(
        noise_config.get("validation_targets", "clean")
    ).strip().lower()
    if validation_targets not in {"clean", "noisy"}:
        raise ValueError("noise.validation_targets must be 'clean' or 'noisy'")
    if validation_targets == "noisy" and noise_mode(config) == "clean":
        raise ValueError("Noisy validation targets require an enabled noise source")
    evaluation = config.get("evaluation", {}) or {}
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation configuration must be a mapping")
    selection_split = str(evaluation.get("selection_split", "validation")).lower()
    if selection_split not in {"validation", "test"}:
        raise ValueError("evaluation.selection_split must be 'validation' or 'test'")
    if selection_split == "test" and not bool(evaluation.get("allow_test_selection", False)):
        raise ValueError("test selection requires evaluation.allow_test_selection=true")


def run_supervised_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Run one reproducible clean or noisy-label supervised experiment."""

    config = deepcopy(config)
    config.setdefault("loss", {"name": "ce"})
    config.setdefault("parameter_update", {"name": "standard"})
    _validate_supervised_config(config)
    config.setdefault("selector", {"name": "all"})
    seed = int(config.get("seed", 1))
    epochs = int(config["trainer"]["epochs"])
    _resolve_dss_epoch_contract(config, epochs)
    seed_everything(seed)
    device = resolve_device(config["trainer"].get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if resume is not None:
        run_dir = Path(resume).resolve().parent
    elif output_dir is not None:
        run_dir = Path(output_dir).resolve()
    else:
        run_dir = Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = None
    if resume is not None:
        checkpoint_payload = read_checkpoint(resume, "cpu")
        saved_config = checkpoint_payload.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("Checkpoint is missing its resolved configuration")
        _validate_resume_config(config, saved_config)

    data_config = config["data"]
    dataset_name = str(data_config.get("name", "cifar10")).lower()
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError("Supervised runner supports cifar10 and cifar100")
    loader_fn = load_cifar10 if dataset_name == "cifar10" else load_cifar100
    num_classes = 10 if dataset_name == "cifar10" else 100
    train_data = loader_fn(data_config.get("root"), "train")
    test_data = loader_fn(data_config.get("root"), "test")
    validation_size = int(data_config["validation_size"])
    if validation_size == 0:
        full_train_indices = np.arange(len(train_data), dtype=np.int64)
        validation_indices = np.empty(0, dtype=np.int64)
    else:
        split_config = data_config.get("validation_split", {}) or {}
        if not isinstance(split_config, Mapping):
            raise TypeError("data.validation_split must be a mapping")
        full_train_indices, validation_indices = train_validation_split(
            train_data.labels,
            validation_size,
            seed,
            strategy=str(split_config.get("strategy", "stratified")),
            rng=str(split_config.get("rng", "default_rng")),
        )
    noise_config = config.get("noise") or {}
    validation_target_source = str(
        noise_config.get("validation_targets", "clean")
    ).strip().lower()
    manifest_indices = full_train_indices
    if validation_target_source == "noisy":
        manifest_indices = np.sort(
            np.concatenate((full_train_indices, validation_indices))
        )

    manifest, manifest_path = prepare_noise_manifest(
        config,
        dataset=dataset_name,
        clean_targets=train_data.labels[manifest_indices],
        global_indices=manifest_indices,
        num_classes=num_classes,
        run_dir=run_dir,
        checkpoint_payload=checkpoint_payload,
        dataset_targets=train_data.labels,
    )
    train_indices = _subset(
        full_train_indices,
        train_data.labels,
        data_config.get("max_train_samples"),
        seed + 1,
    )
    validation_indices = _subset(
        validation_indices,
        train_data.labels,
        data_config.get("max_validation_samples"),
        seed + 2,
    )
    test_indices = _subset(
        np.arange(len(test_data)),
        test_data.labels,
        data_config.get("max_test_samples"),
        seed + 3,
    )
    preprocessing = str(data_config.get("preprocessing", "standard")).lower()
    normalization = data_config.get("normalization")
    if normalization is not None and not isinstance(normalization, Mapping):
        raise TypeError("data.normalization must be a mapping")
    normalization = dict(normalization or {})
    if normalization and set(normalization) != {"mean", "std"}:
        raise ValueError(
            "data.normalization must contain exactly mean and std"
        )
    pixel_mean = (
        cifar_pixel_mean(train_data.images)
        if preprocessing == "gce2018"
        else None
    )
    transform_options = {
        "preprocessing": preprocessing,
        "pixel_mean": pixel_mean,
        "normalization_mean": normalization.get("mean"),
        "normalization_std": normalization.get("std"),
    }

    clean_train_set = TorchCifarDataset(
        train_data,
        train_indices,
        transform=build_cifar_transform(
            True,
            bool(data_config.get("augment", True)),
            **transform_options,
        ),
    )
    train_set = clean_train_set
    noise_metadata = None
    if manifest is not None:
        assert manifest_path is not None
        train_set = NoisyTargetDataset(
            clean_train_set, manifest.global_indices, manifest.noisy_targets
        )
        effective_rate = effective_subset_actual_rate(manifest, train_indices)
    clean_validation_set = TorchCifarDataset(
        train_data,
        validation_indices,
        transform=build_cifar_transform(False, **transform_options),
    )
    validation_set = clean_validation_set
    effective_validation_rate = None
    if manifest is not None and validation_target_source == "noisy":
        validation_set = NoisyTargetDataset(
            clean_validation_set,
            manifest.global_indices,
            manifest.noisy_targets,
        )
        effective_validation_rate = effective_subset_actual_rate(
            manifest, validation_indices
        )
    if manifest is not None:
        assert manifest_path is not None
        noise_metadata = checkpoint_noise_metadata(
            manifest,
            manifest_path,
            run_dir,
            effective_rate,
            mode=noise_mode(config),
            validation_targets=validation_target_source,
            effective_validation_rate=effective_validation_rate,
        )
        config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)

    test_set = TorchCifarDataset(
        test_data,
        test_indices,
        transform=build_cifar_transform(False, **transform_options),
    )
    loader_config = config["loader"]
    train_loader = _loader(train_set, loader_config, shuffle=True, seed=seed)
    validation_loader = _loader(
        validation_set, loader_config, shuffle=False, seed=seed
    )
    test_loader = _loader(test_set, loader_config, shuffle=False, seed=seed)
    evaluation_config = config.get("evaluation", {}) or {}
    selection_split = str(evaluation_config.get("selection_split", "validation")).lower()
    selection_criterion = build_builtin_loss(
        evaluation_config.get("loss", {"name": "ce"})
    ).to(device)
    selection_loader = test_loader if selection_split == "test" else validation_loader

    model = build_model(config["model"], num_classes)
    criterion = build_builtin_loss(config["loss"]).to(device)
    selector = build_builtin_selector(config["selector"])
    update_policy = build_builtin_parameter_update_policy(
        config["parameter_update"]
    )
    optimizer = build_optimizer(model, config["optimizer"])
    pipeline = build_builtin_pipeline(config.get("pipeline"))
    pipeline_compatibility_warnings: list[str] = []
    if resume is not None:
        assert checkpoint_payload is not None
        pipeline_compatibility_warnings = pipeline.restore_for_resume(
            run_dir,
            checkpoint_state=checkpoint_payload.get("pipeline"),
            component_states=checkpoint_payload.get("component_states"),
            dataset=dataset_name,
            split="train",
        )
    else:
        pipeline.prepare_transition(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            dataset=dataset_name,
            split="train",
            run_dir=run_dir,
        )
    scheduler = build_scheduler(optimizer, config.get("scheduler"), epochs)
    algorithm = SupervisedClassificationAlgorithm(
        model,
        optimizer,
        criterion,
        device,
        selector=selector,
        update_policy=update_policy,
        risk_corrector=pipeline.risk_corrector,
        transition=pipeline.artifacts.transition,
        weight_provider=pipeline.weight_provider,
        objective_consumer=pipeline.objective_consumer,
    )
    algorithm.setup(ExperimentContext(run_dir, config, seed))
    state = RunState(phase="train")
    early_stopping = EarlyStopping.from_config(config.get("early_stopping"))
    completed_epoch = -1
    last_completed_epoch = completed_epoch
    best_epoch = -1
    best_accuracy = float("-inf")
    best_selection_accuracy = float("-inf")
    compatibility_warnings: list[str] = list(
        pipeline_compatibility_warnings
    )
    if resume is not None:
        state, completed_epoch, checkpoint_payload = load_checkpoint(
            resume, algorithm, device, scheduler=scheduler
        )
        last_completed_epoch = completed_epoch
        best_epoch = int(checkpoint_payload["best_epoch"])
        best_accuracy = float(checkpoint_payload.get("best_validation_accuracy", float("-inf")))
        best_selection_accuracy = float(
            checkpoint_payload.get("best_selection_accuracy", best_accuracy)
        )
        compatibility_warnings.extend(
            checkpoint_payload.get("_compatibility_warnings", [])
        )
        if early_stopping is not None and checkpoint_payload.get("early_stopping") is not None:
            early_stopping.load_state_dict(checkpoint_payload["early_stopping"])

    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(seed, device), indent=2), encoding="utf-8"
    )
    if noise_metadata is not None:
        (run_dir / "noise_summary.json").write_text(
            json.dumps(noise_metadata, indent=2), encoding="utf-8"
        )

    algorithm.on_run_start(state)
    metrics_path = run_dir / "metrics.jsonl"
    curve_rows: list[dict[str, Any]] = []
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if value.get("event") == "epoch":
                curve_rows.append(value)
    progress_config = config["trainer"].get("progress", {})
    if isinstance(progress_config, bool):
        progress_config = {"enabled": progress_config}
    if not isinstance(progress_config, Mapping):
        raise TypeError("trainer.progress must be a boolean or mapping")
    progress_enabled = bool(progress_config.get("enabled", False))
    progress_interval = int(progress_config.get("update_interval", 20))
    curves_enabled = bool(progress_config.get("curves", progress_enabled))
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        if compatibility_warnings:
            metrics_file.write(json.dumps({
                "event": "checkpoint_compatibility",
                "warnings": compatibility_warnings,
            }) + "\n")
            metrics_file.flush()
        for epoch in range(completed_epoch + 1, epochs):
            state.cycle = epoch
            algorithm.on_cycle_start(state)
            loss_sum = 0.0
            all_sample_loss_sum = 0.0
            correct_weighted = 0.0
            samples = 0.0
            selected_samples = 0.0
            update_metric_sums: dict[str, float] = {}
            update_metric_steps = 0
            learning_rate = float(optimizer.param_groups[0]["lr"])
            progress = TerminalTrainingProgress(
                epoch=epoch + 1,
                total_epochs=epochs,
                total_batches=len(train_loader),
                update_interval=progress_interval,
                enabled=progress_enabled,
            )
            for batch_number, raw_batch in enumerate(train_loader, start=1):
                result = algorithm.step(Batch(raw_batch), state)
                count = result.metrics["samples"]
                selected_count = result.metrics["selected_samples"]
                loss_sum += result.metrics["loss"] * selected_count
                all_sample_loss_sum += result.metrics["all_sample_loss"] * count
                correct_weighted += result.metrics["accuracy"] * count
                samples += count
                selected_samples += selected_count
                update_metrics = {
                    key: float(value)
                    for key, value in result.metrics.items()
                    if key.startswith("update_")
                }
                if update_metrics:
                    update_metric_steps += 1
                    for key, value in update_metrics.items():
                        update_metric_sums[key] = (
                            update_metric_sums.get(key, 0.0) + value
                        )
                progress.update(
                    batch_number,
                    loss=loss_sum / selected_samples,
                    accuracy=correct_weighted / samples,
                )
            algorithm.on_cycle_end(state)
            validation = None
            if selection_split == "validation":
                validation = evaluate_classification(
                    model, validation_loader, selection_criterion, device
                )
                selection = validation
            else:
                selection = evaluate_classification(
                    model, selection_loader, selection_criterion, device
                )
            if scheduler is not None:
                scheduler.step()
            row = {
                "event": "epoch",
                "epoch": epoch + 1,
                "global_step": state.step,
                "learning_rate": learning_rate,
                "train_loss": loss_sum / selected_samples,
                "train_all_sample_loss": all_sample_loss_sum / samples,
                "train_accuracy": correct_weighted / samples,
                "selected_samples": selected_samples,
                "selected_ratio": selected_samples / samples,
                "selection_split": selection_split,
                "selection_loss": selection["loss"],
                "selection_accuracy": selection["accuracy"],
            }
            if validation is not None:
                row.update({
                    "validation_loss": validation["loss"],
                    "validation_accuracy": validation["accuracy"],
                })
            if update_metric_steps:
                row.update({
                    f"train_{key}": value / update_metric_steps
                    for key, value in update_metric_sums.items()
                })
            state.metrics = {
                key: float(value)
                for key, value in row.items()
                if isinstance(value, float)
            }
            improved = selection["accuracy"] > best_selection_accuracy
            if improved:
                best_selection_accuracy = selection["accuracy"]
                best_accuracy = best_selection_accuracy
                best_epoch = epoch
            if early_stopping is not None:
                early_stopping.update(row)
            last_completed_epoch = epoch
            checkpoint_kwargs = {
                "scheduler": scheduler,
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "selection_split": selection_split,
                "best_selection_accuracy": best_selection_accuracy,
                "noise": noise_metadata,
                "pipeline": pipeline.state_dict(),
                "component_states": pipeline.component_state_dict(),
                "early_stopping": None if early_stopping is None else early_stopping.state_dict(),
            }
            save_checkpoint(
                run_dir / "last.pt",
                algorithm,
                state,
                epoch,
                config,
                **checkpoint_kwargs,
            )
            if improved:
                save_checkpoint(
                    run_dir / "best.pt",
                    algorithm,
                    state,
                    epoch,
                    config,
                    **checkpoint_kwargs,
                )
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            curve_rows.append(row)
            if curves_enabled:
                curve_rows_for_svg = [
                    {
                        **curve_row,
                        "validation_loss": curve_row.get(
                            "validation_loss", curve_row["selection_loss"]
                        ),
                        "validation_accuracy": curve_row.get(
                            "validation_accuracy", curve_row["selection_accuracy"]
                        ),
                    }
                    for curve_row in curve_rows
                ]
                write_training_curves_svg(
                    curve_rows_for_svg, run_dir / "training_curves.svg"
                )
            print(json.dumps(row), flush=True)
            if early_stopping is not None and early_stopping.stopped:
                state.stopped = True
                break

        best_path = run_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError(
                "No best checkpoint exists; increase the total epoch target when resuming"
            )
        best_payload = read_checkpoint(best_path, device)
        model.load_state_dict(best_payload["model"])
        test = evaluate_classification(
            model, test_loader,
            selection_criterion if selection_split == "test" else criterion,
            device,
        )
        final: dict[str, Any] = {
            "event": "final",
            "completed_epochs": last_completed_epoch + 1,
            "global_step": state.step,
            "best_epoch": best_epoch + 1,
            "best_validation_accuracy": best_accuracy,
            "selection_split": selection_split,
            "best_selection_accuracy": best_selection_accuracy,
            "test_selection_leakage": selection_split == "test",
            "test_checkpoint": "best.pt",
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
        }
        if noise_metadata is not None:
            final["noise"] = noise_metadata
        if device.type == "cuda":
            final["max_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / (
                1024 ** 2
            )
        metrics_file.write(json.dumps(final) + "\n")
        print(json.dumps(final), flush=True)
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    algorithm.on_run_end(state)
    algorithm.close()
    return run_dir


def run_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Compatibility entry point for the general training CLI."""

    method = config.get("method")
    method_name = (
        str(method.get("name", "")).strip().lower()
        if isinstance(method, Mapping)
        else str(method or "").strip().lower()
    )
    if method_name == "coteaching":
        from lnl_toolbox.training.coteaching_experiment import (
            run_coteaching_experiment,
        )

        return run_coteaching_experiment(config, output_dir, resume)
    if method_name == "cnlcu":
        from lnl_toolbox.training.cnlcu_experiment import run_cnlcu_experiment

        return run_cnlcu_experiment(config, output_dir, resume)
    if method_name == "dual_t_forward":
        raise ValueError("method 'dual_t_forward' was renamed to 'dual_t'")
    if method_name == "dual_t":
        from lnl_toolbox.training.dual_t_experiment import (
            run_dual_t_experiment,
        )

        return run_dual_t_experiment(config, output_dir, resume)
    if method_name == "importance_reweighting":
        from lnl_toolbox.training.importance_reweighting_experiment import (
            run_importance_reweighting_experiment,
        )

        return run_importance_reweighting_experiment(
            config, output_dir, resume
        )
    if method_name == "pcse":
        from lnl_toolbox.training.pcse_experiment import run_pcse_experiment

        return run_pcse_experiment(config, output_dir, resume)
    return run_supervised_experiment(config, output_dir, resume)
