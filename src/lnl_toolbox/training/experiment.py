from __future__ import annotations

"""Unified clean/noisy supervised CIFAR experiment runner."""

from copy import deepcopy
from datetime import datetime
import json
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
from lnl_toolbox.data.torch_cifar import TorchCifarDataset, build_cifar_transform, stratified_split
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.models.cifar_resnet import cifar_resnet18, preact_resnet18
from lnl_toolbox.models.tiny_cnn import TinyCNN
from lnl_toolbox.plugins.builtin import build_builtin_loss, build_builtin_selector
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    effective_subset_actual_rate,
    noise_mode,
    prepare_noise_manifest,
)


def build_model(config: Mapping[str, Any], num_classes: int) -> nn.Module:
    name = str(config.get("name", "preact_resnet18")).lower()
    if name == "tiny_cnn":
        return TinyCNN(num_classes, int(config.get("width", 64)))
    if name == "resnet18":
        return cifar_resnet18(num_classes, int(config.get("base_width", 64)))
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
    return config.get(key)


def _validate_resume_config(current: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    for key in ("seed", "model", "loss", "optimizer", "scheduler", "selector"):
        if _normalized_resume_value(current, key) != _normalized_resume_value(saved, key):
            raise ValueError(f"Resume configuration changed {key}")
    current_dataset = str(current.get("data", {}).get("name", "cifar10")).lower()
    saved_dataset = str(saved.get("data", {}).get("name", "cifar10")).lower()
    if current_dataset != saved_dataset:
        raise ValueError("Resume configuration changed data.name")


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
    })
    if "source_manifest_sha256" in metadata:
        resolved["source_manifest_sha256"] = metadata["source_manifest_sha256"]
    return resolved


def _validate_supervised_config(config: Mapping[str, Any]) -> None:
    field = "transition_estimator"
    if field in config:
        raise ValueError(
            f"Configuration field {field!r} is registered but not connected to "
            "run_supervised_experiment"
        )


def run_supervised_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Run one reproducible clean or noisy-label supervised experiment."""

    config = deepcopy(config)
    config.setdefault("loss", {"name": "ce"})
    _validate_supervised_config(config)
    config.setdefault("selector", {"name": "all"})
    seed = int(config.get("seed", 1))
    epochs = int(config["trainer"]["epochs"])
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
    full_train_indices, validation_indices = stratified_split(
        train_data.labels, int(data_config["validation_size"]), seed
    )

    manifest, manifest_path = prepare_noise_manifest(
        config,
        dataset=dataset_name,
        clean_targets=train_data.labels[full_train_indices],
        global_indices=full_train_indices,
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

    clean_train_set = TorchCifarDataset(
        train_data,
        train_indices,
        transform=build_cifar_transform(
            True, bool(data_config.get("augment", True))
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
        noise_metadata = checkpoint_noise_metadata(
            manifest,
            manifest_path,
            run_dir,
            effective_rate,
            mode=noise_mode(config),
        )
        config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)

    validation_set = TorchCifarDataset(
        train_data, validation_indices, transform=build_cifar_transform(False)
    )
    test_set = TorchCifarDataset(
        test_data, test_indices, transform=build_cifar_transform(False)
    )
    loader_config = config["loader"]
    train_loader = _loader(train_set, loader_config, shuffle=True, seed=seed)
    validation_loader = _loader(
        validation_set, loader_config, shuffle=False, seed=seed
    )
    test_loader = _loader(test_set, loader_config, shuffle=False, seed=seed)

    model = build_model(config["model"], num_classes)
    criterion = build_builtin_loss(config["loss"]).to(device)
    selector = build_builtin_selector(config["selector"])
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(optimizer, config.get("scheduler"), epochs)
    algorithm = SupervisedClassificationAlgorithm(
        model, optimizer, criterion, device, selector=selector
    )
    algorithm.setup(ExperimentContext(run_dir, config, seed))
    state = RunState(phase="train")
    completed_epoch = -1
    best_epoch = -1
    best_accuracy = float("-inf")
    compatibility_warnings: list[str] = []
    if resume is not None:
        state, completed_epoch, checkpoint_payload = load_checkpoint(
            resume, algorithm, device, scheduler=scheduler
        )
        best_epoch = int(checkpoint_payload["best_epoch"])
        best_accuracy = float(checkpoint_payload["best_validation_accuracy"])
        compatibility_warnings = list(
            checkpoint_payload.get("_compatibility_warnings", [])
        )

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
            learning_rate = float(optimizer.param_groups[0]["lr"])
            for raw_batch in train_loader:
                result = algorithm.step(Batch(raw_batch), state)
                count = result.metrics["samples"]
                selected_count = result.metrics["selected_samples"]
                loss_sum += result.metrics["loss"] * selected_count
                all_sample_loss_sum += result.metrics["all_sample_loss"] * count
                correct_weighted += result.metrics["accuracy"] * count
                samples += count
                selected_samples += selected_count
            algorithm.on_cycle_end(state)
            validation = evaluate_classification(
                model, validation_loader, criterion, device
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
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
            }
            state.metrics = {
                key: float(value)
                for key, value in row.items()
                if isinstance(value, float)
            }
            improved = validation["accuracy"] > best_accuracy
            if improved:
                best_accuracy = validation["accuracy"]
                best_epoch = epoch
            checkpoint_kwargs = {
                "scheduler": scheduler,
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "noise": noise_metadata,
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
            print(json.dumps(row), flush=True)

        best_path = run_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError(
                "No best checkpoint exists; increase the total epoch target when resuming"
            )
        best_payload = read_checkpoint(best_path, device)
        model.load_state_dict(best_payload["model"])
        test = evaluate_classification(model, test_loader, criterion, device)
        final: dict[str, Any] = {
            "event": "final",
            "completed_epochs": epochs,
            "global_step": state.step,
            "best_epoch": best_epoch + 1,
            "best_validation_accuracy": best_accuracy,
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

    return run_supervised_experiment(config, output_dir, resume)
