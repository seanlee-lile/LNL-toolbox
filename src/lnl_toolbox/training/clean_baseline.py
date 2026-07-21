from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import csv
import json
import platform
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.core import RunState
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.torch_cifar import TorchCifarDataset, build_cifar_transform, stratified_split
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.losses.torch_losses import validate_per_sample_loss
from lnl_toolbox.models.cifar_resnet import cifar_resnet18, preact_resnet18
from lnl_toolbox.models.tiny_cnn import TinyCNN
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything


def build_clean_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    name = config.get("name", "preact_resnet18").lower()
    if name == "tiny_cnn":
        return TinyCNN(num_classes, int(config.get("width", 64)))
    if name == "resnet18":
        return cifar_resnet18(num_classes, int(config.get("base_width", 64)))
    if name == "preact_resnet18":
        return preact_resnet18(num_classes, int(config.get("base_width", 64)))
    raise ValueError(f"Unsupported model: {name}")


def build_clean_optimizer(model: nn.Module, config: dict[str, Any]):
    name = config.get("name", "sgd").lower()
    common = {"lr": float(config["lr"]), "weight_decay": float(config.get("weight_decay", 0.0))}
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=float(config.get("momentum", 0.9)),
                               nesterov=bool(config.get("nesterov", False)), **common)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **common)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_clean_scheduler(optimizer, config: dict[str, Any] | None, epochs: int):
    if not config or config.get("name", "none").lower() == "none":
        return None
    name = config["name"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config.get("t_max", epochs)), eta_min=float(config.get("eta_min", 0.0))
        )
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(value) for value in config["milestones"]],
            gamma=float(config.get("gamma", 0.1))
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def _subset(indices: np.ndarray, labels: np.ndarray, size: int | None, seed: int) -> np.ndarray:
    if size is None or size >= len(indices):
        return indices
    _, selected = stratified_split(labels[indices], size, seed)
    return indices[selected]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(dataset, config: dict[str, Any], shuffle: bool, seed: int) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=shuffle, num_workers=workers,
        pin_memory=bool(config.get("pin_memory", True)), persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers else None,
        generator=torch.Generator().manual_seed(seed),
    )


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint(model, optimizer, scheduler, state: RunState, completed_epoch: int,
                best_epoch: int, best_accuracy: float, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "run_state": asdict(state), "completed_epoch": completed_epoch,
        "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy, "config": config,
    }


def _load_checkpoint(path: Path, model, optimizer, scheduler, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    return RunState(**payload["run_state"]), payload


def _environment(seed: int, device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "device": str(device), "seed": seed,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def run_clean_experiment(config: dict[str, Any], output_dir: str | Path | None = None,
                         resume: str | Path | None = None) -> Path:
    config = deepcopy(config)
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
        run_dir = Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(_environment(seed, device), indent=2), encoding="utf-8")

    data_config = config["data"]
    dataset_name = data_config.get("name", "cifar10").lower()
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError("Clean baseline supports cifar10 and cifar100")
    loader_fn = load_cifar10 if dataset_name == "cifar10" else load_cifar100
    num_classes = 10 if dataset_name == "cifar10" else 100
    train_data = loader_fn(data_config.get("root"), "train")
    test_data = loader_fn(data_config.get("root"), "test")
    train_indices, validation_indices = stratified_split(
        train_data.labels, int(data_config["validation_size"]), seed
    )
    train_indices = _subset(train_indices, train_data.labels, data_config.get("max_train_samples"), seed + 1)
    validation_indices = _subset(
        validation_indices, train_data.labels, data_config.get("max_validation_samples"), seed + 2
    )
    test_indices = _subset(
        np.arange(len(test_data)), test_data.labels, data_config.get("max_test_samples"), seed + 3
    )
    train_set = TorchCifarDataset(
        train_data, train_indices,
        transform=build_cifar_transform(True, bool(data_config.get("augment", True)))
    )
    validation_set = TorchCifarDataset(train_data, validation_indices, transform=build_cifar_transform(False))
    test_set = TorchCifarDataset(test_data, test_indices, transform=build_cifar_transform(False))
    loader_config = config["loader"]
    train_loader = _make_loader(train_set, loader_config, True, seed)
    validation_loader = _make_loader(validation_set, loader_config, False, seed)
    test_loader = _make_loader(test_set, loader_config, False, seed)

    model = build_clean_model(config["model"], num_classes).to(device)
    criterion = build_builtin_loss(config.get("loss", {"name": "ce"})).to(device)
    optimizer = build_clean_optimizer(model, config["optimizer"])
    scheduler = build_clean_scheduler(optimizer, config.get("scheduler"), epochs)
    state = RunState(phase="train")
    completed_epoch = -1
    best_epoch = -1
    best_accuracy = float("-inf")
    if resume is not None:
        state, payload = _load_checkpoint(Path(resume), model, optimizer, scheduler, device)
        completed_epoch = int(payload["completed_epoch"])
        best_epoch = int(payload.get("best_epoch", completed_epoch))
        best_accuracy = float(payload.get("best_validation_accuracy", float("-inf")))

    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(completed_epoch + 1, epochs):
            state.cycle = epoch
            state.phase = "train"
            model.train()
            loss_sum = 0.0
            correct = 0
            samples = 0
            learning_rate = float(optimizer.param_groups[0]["lr"])
            for batch in train_loader:
                inputs = batch["input"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                count = int(targets.numel())
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                per_sample_loss = validate_per_sample_loss(criterion(logits, targets), count)
                loss = per_sample_loss.mean()
                loss.backward()
                optimizer.step()
                loss_sum += float(per_sample_loss.detach().sum().item())
                correct += int((logits.argmax(1) == targets).sum().item())
                samples += count
                state.step += 1
            validation = evaluate_classification(model, validation_loader, criterion, device)
            if scheduler is not None:
                scheduler.step()
            row = {
                "event": "epoch", "epoch": epoch + 1, "global_step": state.step, "learning_rate": learning_rate,
                "train_loss": loss_sum / samples, "train_accuracy": correct / samples,
                "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
            }
            state.metrics = {key: float(value) for key, value in row.items() if isinstance(value, float)}
            improved = validation["accuracy"] > best_accuracy
            if improved:
                best_accuracy = validation["accuracy"]
                best_epoch = epoch
            payload = _checkpoint(model, optimizer, scheduler, state, epoch, best_epoch, best_accuracy, config)
            _atomic_save(payload, run_dir / "last.pt")
            if improved:
                _atomic_save(payload, run_dir / "best.pt")
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            print(json.dumps(row), flush=True)

        if not (run_dir / "best.pt").is_file():
            raise RuntimeError("No best checkpoint exists; increase the total epoch target when resuming")
        best_payload = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model"])
        test = evaluate_classification(model, test_loader, criterion, device)
        final = {
            "event": "final", "completed_epochs": epochs, "global_step": state.step,
            "best_epoch": best_epoch + 1, "best_validation_accuracy": best_accuracy,
            "test_checkpoint": "best.pt", "test_loss": test["loss"], "test_accuracy": test["accuracy"],
        }
        if device.type == "cuda":
            final["max_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        metrics_file.write(json.dumps(final) + "\n")
        print(json.dumps(final), flush=True)
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return run_dir


def run_seed_suite(config: dict[str, Any], seeds: list[int], output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        current = deepcopy(config)
        current["seed"] = int(seed)
        run_dir = run_clean_experiment(current, root / f"seed-{seed}")
        result = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
        results.append({"seed": seed, **result})
    accuracies = np.asarray([row["test_accuracy"] for row in results], dtype=np.float64)
    summary = {
        "seeds": seeds, "runs": results, "test_accuracy_mean": float(accuracies.mean()),
        "test_accuracy_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "best_epoch", "best_validation_accuracy", "test_accuracy"])
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    return root
