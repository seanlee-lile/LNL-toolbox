from __future__ import annotations

"""End-to-end orchestration for a reproducible CIFAR training run.

This module deliberately keeps experiment I/O and lifecycle management outside
individual algorithms so future LNL methods can share the same data splits,
logging, checkpointing, and evaluation rules.
"""

from dataclasses import asdict
from datetime import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.torch_cifar import TorchCifarDataset, build_cifar_transform, stratified_split
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.models import TinyCNN
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from .checkpoint import load_checkpoint, save_checkpoint


def _subset(indices: np.ndarray, labels: np.ndarray, size: int | None, seed: int) -> np.ndarray:
    if size is None or size >= len(indices):
        return indices
    selected_labels = labels[indices]
    _, chosen = stratified_split(selected_labels, size, seed)
    return indices[chosen]


def _loader(dataset, config: dict[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", True)),
        generator=generator,
        persistent_workers=int(config.get("num_workers", 0)) > 0,
    )


def _environment(seed: int, device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device), "seed": seed,
    }


def _optimizer(model, config: dict[str, Any]):
    name = config.get("name", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                                 weight_decay=float(config.get("weight_decay", 0.0)))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=float(config["lr"]),
                               momentum=float(config.get("momentum", 0.9)),
                               weight_decay=float(config.get("weight_decay", 0.0)))
    raise ValueError(f"Unsupported optimizer: {name}")


def run_experiment(config: dict[str, Any], output_dir: str | Path | None = None,
                   resume: str | Path | None = None) -> Path:
    """Run one configured experiment and return its artifact directory."""
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    device = resolve_device(config["trainer"].get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if resume:
        run_dir = Path(resume).resolve().parent
    elif output_dir:
        run_dir = Path(output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(config.get("output_root", "artifacts/runs")) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    environment = _environment(seed, device)
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    # Build train/validation/test sets once so every algorithm is compared on
    # identical sample identities and splits.
    data_cfg = config["data"]
    dataset_name = data_cfg.get("name", "cifar10")
    loader_fn = load_cifar10 if dataset_name == "cifar10" else load_cifar100
    classes = 10 if dataset_name == "cifar10" else 100
    raw_train = loader_fn(data_cfg.get("root"), "train")
    raw_test = loader_fn(data_cfg.get("root"), "test")
    train_indices, val_indices = stratified_split(raw_train.labels, int(data_cfg["validation_size"]), seed)
    train_indices = _subset(train_indices, raw_train.labels, data_cfg.get("max_train_samples"), seed + 1)
    val_indices = _subset(val_indices, raw_train.labels, data_cfg.get("max_validation_samples"), seed + 2)
    test_indices = _subset(np.arange(len(raw_test)), raw_test.labels, data_cfg.get("max_test_samples"), seed + 3)
    train_set = TorchCifarDataset(raw_train, train_indices, transform=build_cifar_transform(True, data_cfg.get("augment", True)))
    val_set = TorchCifarDataset(raw_train, val_indices, transform=build_cifar_transform(False))
    test_set = TorchCifarDataset(raw_test, test_indices, transform=build_cifar_transform(False))
    loader_cfg = config["loader"]
    train_loader = _loader(train_set, loader_cfg, shuffle=True, seed=seed)
    val_loader = _loader(val_set, loader_cfg, shuffle=False, seed=seed)
    test_loader = _loader(test_set, loader_cfg, shuffle=False, seed=seed)

    model = TinyCNN(num_classes=classes, width=int(config["model"].get("width", 64)))
    criterion = build_builtin_loss(config.get("loss", {"name": "ce"})).to(device)
    algorithm = SupervisedClassificationAlgorithm(model, _optimizer(model, config["optimizer"]), criterion, device)
    algorithm.setup(ExperimentContext(run_dir, config, seed))
    state = RunState()
    completed_epoch = -1
    if resume:
        state, completed_epoch, _ = load_checkpoint(resume, algorithm, device)
    algorithm.on_run_start(state)
    metrics_path = run_dir / "metrics.jsonl"
    epochs = int(config["trainer"]["epochs"])
    # JSON Lines is append-only, human-readable, and survives interrupted runs.
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(completed_epoch + 1, epochs):
            state.cycle = epoch
            algorithm.on_cycle_start(state)
            loss_sum = correct_weighted = samples = 0.0
            for raw_batch in train_loader:
                result = algorithm.step(Batch(raw_batch), state)
                count = result.metrics["samples"]
                loss_sum += result.metrics["loss"] * count
                correct_weighted += result.metrics["accuracy"] * count
                samples += count
            algorithm.on_cycle_end(state)
            validation = evaluate_classification(model, val_loader, criterion, device)
            row = {
                "event": "epoch", "epoch": epoch + 1, "global_step": state.step,
                "train_loss": loss_sum / samples, "train_accuracy": correct_weighted / samples,
                "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
            }
            state.metrics = {key: value for key, value in row.items() if isinstance(value, float)}
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            print(json.dumps(row), flush=True)
            save_checkpoint(run_dir / "last.pt", algorithm, state, epoch, config)

        test = evaluate_classification(model, test_loader, criterion, device)
        final = {"event": "final", "completed_epochs": epochs, "global_step": state.step,
                 "test_loss": test["loss"], "test_accuracy": test["accuracy"]}
        if device.type == "cuda":
            final["max_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        metrics_file.write(json.dumps(final) + "\n")
        print(json.dumps(final), flush=True)
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    algorithm.on_run_end(state)
    algorithm.close()
    return run_dir
