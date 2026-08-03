from __future__ import annotations

"""Standalone CWD lifecycle for the CIFAR airplane/automobile benchmark."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.algorithms.cwd import CWDGlobalObjective
from lnl_toolbox.data.binary_benchmarks import (
    cifar_airplane_automobile_view,
    corrupt_binary_labels,
    stratified_binary_splits,
)
from lnl_toolbox.data.cifar import CifarData, load_cifar10
from lnl_toolbox.data.torch_cifar import TorchCifarDataset, build_cifar_transform
from lnl_toolbox.estimators.cwd import CWDEstimator
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.models.cifar_resnet import cifar_resnet34
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.models.tiny_cnn import TinyCNN
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.snapshots import collect_feature_snapshot


def _loader(dataset, config: Mapping[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", True)),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
        drop_last=bool(config.get("drop_last", False)) if shuffle else False,
    )


def _binary_cifar_corpus(root: str | Path | None) -> tuple[CifarData, np.ndarray]:
    train = load_cifar10(root, "train")
    test = load_cifar10(root, "test")
    train_images, train_labels, train_source = cifar_airplane_automobile_view(train)
    test_images, test_labels, test_source = cifar_airplane_automobile_view(test)
    data = CifarData(
        np.concatenate((train_images, test_images)),
        np.concatenate((train_labels, test_labels)),
        ("airplane", "automobile"),
        "combined",
        "cifar10_airplane_automobile",
    )
    source_indices = np.concatenate((train_source, len(train) + test_source))
    return data, source_indices


def _limit_indices(labels: np.ndarray, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or int(maximum) >= labels.size:
        return np.arange(labels.size, dtype=np.int64)
    if int(maximum) < 4:
        raise ValueError("CWD max_samples must contain at least four samples")
    random = np.random.default_rng(seed)
    parts = []
    for class_index in (0, 1):
        candidates = np.flatnonzero(labels == class_index)
        random.shuffle(candidates)
        parts.append(candidates[: int(maximum) // 2])
    return np.sort(np.concatenate(parts))


def _build_model(config: Mapping[str, Any]):
    name = str(config.get("name", "resnet34")).strip().lower()
    if name in {"resnet34", "cifar_resnet34"}:
        return cifar_resnet34(2, int(config.get("base_width", 64)))
    if name == "tiny_cnn":
        return TinyCNN(2, int(config.get("width", 8)))
    raise ValueError(f"Unsupported CWD model: {name}")


def _train_epoch(model, optimizer, loader, objective, device) -> tuple[float, float]:
    model.train()
    loss_sum = correct = samples = 0.0
    base_loss = CrossEntropyLoss().to(device)
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        indices = batch["index"].to(device, non_blocking=True)
        output = forward_with_features(model, inputs)
        loss = objective.compute(
            model=model,
            logits=output.logits,
            features=output.features,
            noisy_targets=targets,
            sample_indices=indices,
            base_loss=base_loss,
            metadata={},
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count = int(targets.numel())
        loss_sum += float(loss.detach()) * count
        correct += float(output.logits.argmax(1).eq(targets).sum())
        samples += count
    return loss_sum / samples, correct / samples


def run_cwd_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Run one configured fold of the paper's five-fold CIFAR-binary protocol."""

    config = deepcopy(config)
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    device = resolve_device(str(config["trainer"].get("device", "auto")))
    run_dir = (
        Path(resume).resolve().parent
        if resume is not None
        else Path(
            output_dir
            or Path(config.get("output_root", "artifacts/runs"))
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        ).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None if resume is None else read_checkpoint(resume, "cpu")
    if checkpoint is not None and checkpoint.get("config") != config:
        raise ValueError("CWD resume configuration mismatch")

    data_config = config["data"]
    corpus, source_indices = _binary_cifar_corpus(data_config.get("root"))
    limited = _limit_indices(corpus.labels, data_config.get("max_samples"), seed)
    images = corpus.images[limited]
    clean_labels = corpus.labels[limited]
    source_indices = source_indices[limited]
    folds = int(data_config.get("folds", 5))
    fold_index = int(data_config.get("fold_index", 0))
    splits = stratified_binary_splits(clean_labels, folds=folds, seed=seed)
    if not 0 <= fold_index < len(splits):
        raise ValueError("CWD fold_index is outside the configured fold range")
    train_positions, test_positions = splits[fold_index]

    noise_config = config["noise"]
    rho_positive = float(noise_config["rho_positive"])
    rho_negative = float(noise_config["rho_negative"])
    noise_seed = int(noise_config.get("seed", seed))
    manifest = corrupt_binary_labels(
        clean_labels[train_positions], rho_positive, rho_negative, noise_seed
    )
    manifest.dataset = "cifar10_airplane_automobile"
    manifest.metadata.update(
        {
            "folds": folds,
            "fold_index": fold_index,
            "source_global_indices": source_indices[train_positions].tolist(),
        }
    )
    manifest_path = run_dir / "noise_manifest.npz"
    if checkpoint is None:
        manifest.save(manifest_path)
    else:
        expected = checkpoint.get("noise_mapping_hash")
        if expected != manifest.mapping_hash:
            raise ValueError("CWD resume noise manifest mismatch")

    train_data = CifarData(
        images[train_positions],
        manifest.noisy_targets,
        corpus.class_names,
        "train",
        corpus.dataset,
    )
    test_data = CifarData(
        images[test_positions],
        clean_labels[test_positions],
        corpus.class_names,
        "test",
        corpus.dataset,
    )
    transform = build_cifar_transform(
        True, bool(data_config.get("augment", False))
    )
    evaluation_transform = build_cifar_transform(False)
    train_set = TorchCifarDataset(train_data, transform=transform)
    snapshot_set = TorchCifarDataset(train_data, transform=evaluation_transform)
    test_set = TorchCifarDataset(test_data, transform=evaluation_transform)
    loader_config = config["loader"]
    train_loader = _loader(train_set, loader_config, shuffle=True, seed=seed)
    snapshot_loader = _loader(snapshot_set, loader_config, shuffle=False, seed=seed)
    test_loader = _loader(test_set, loader_config, shuffle=False, seed=seed)

    model = _build_model(config["model"]).to(device)
    optimizer_config = config["optimizer"]
    if str(optimizer_config.get("name", "adam")).lower() != "adam":
        raise ValueError("CWD paper runner requires Adam")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    milestones = [int(value) for value in config.get("scheduler", {}).get("milestones", [])]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=float(config.get("scheduler", {}).get("gamma", 0.1)),
    )
    start_epoch = 0
    rows: list[dict[str, Any]] = []
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        rows = list(checkpoint.get("metrics", []))

    transition = np.asarray(
        [
            [1.0 - rho_negative, rho_negative],
            [rho_positive, 1.0 - rho_positive],
        ],
        dtype=np.float64,
    )
    estimator = CWDEstimator(transition, ridge=float(config.get("cwd", {}).get("ridge", 1e-8)))
    criterion = CrossEntropyLoss().to(device)
    epochs = int(config["trainer"]["epochs"])
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(start_epoch, epochs):
        snapshot = collect_feature_snapshot(
            model,
            snapshot_loader,
            device,
            dataset=corpus.dataset,
            split=f"train_fold_{fold_index}",
            feature_extractor=lambda current, inputs: forward_with_features(current, inputs).features,
        )
        artifact = estimator.estimate(snapshot)
        snapshot.save(run_dir / "feature_snapshot.npz")
        artifact.save(run_dir / "statistic_artifact.npz")
        objective = CWDGlobalObjective(artifact)
        train_loss, train_accuracy = _train_epoch(
            model, optimizer, train_loader, objective, device
        )
        test = evaluate_classification(model, test_loader, criterion, device)
        row = standardize_epoch_row(
            {
                "epoch": epoch + 1,
                "phase": "cwd",
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": test["loss"],
                "validation_accuracy": test["accuracy"],
                "test_loss": test["loss"],
                "test_accuracy": test["accuracy"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "statistic_hash": artifact.artifact_hash,
                "feature_snapshot_hash": snapshot.snapshot_hash,
                "fold_index": fold_index,
            }
        )
        rows.append(row)
        metrics_path.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
            encoding="utf-8",
        )
        scheduler.step()
        atomic_save(
            {
                "format_version": 1,
                "method": "cwd",
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "completed_epoch": epoch,
                "metrics": rows,
                "rng_state": capture_rng_state(),
                "noise_mapping_hash": manifest.mapping_hash,
                "statistic_hash": artifact.artifact_hash,
                "feature_snapshot_hash": snapshot.snapshot_hash,
            },
            run_dir / "last.pt",
        )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if rows:
        write_training_curves_svg(rows, run_dir / "training_curves.svg")
    return run_dir


__all__ = ["run_cwd_experiment"]
