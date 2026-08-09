from __future__ import annotations

"""Standalone, modular SED+FINE experiment lifecycle."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.algorithms.fine import FINERegularizer
from lnl_toolbox.data.cifar import load_cifar100
from lnl_toolbox.data.multi_view import (
    IndexedMultiViewCifarDataset,
    build_strong_cifar_transform,
)
from lnl_toolbox.data.torch_cifar import (
    TorchCifarDataset,
    build_cifar_transform,
    stratified_split,
    train_validation_split,
)
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.models.fine_cnn import FineSevenCNN
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.selectors.sed import (
    SelfAdaptiveClassSelector,
    SelfAdaptiveConfidenceReweighting,
)
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.epoch_stream import (
    build_epoch_loader,
    loader_stream_metadata,
    validate_loader_stream,
)
from lnl_toolbox.training.interfaces import RunContext
from lnl_toolbox.training.experiment import build_model, build_optimizer
from lnl_toolbox.training.model_ema import ModelEMA
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    effective_subset_actual_rate,
    noise_mode,
    prepare_noise_manifest,
)
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg


def _build_fine_model(config: Mapping[str, Any], num_classes: int):
    """Build the standalone paper model without extending the main runner."""

    if str(config.get("name", "")).strip().lower() == "fine_seven_cnn":
        return FineSevenCNN(
            num_classes,
            base_width=int(config.get("base_width", 128)),
            dropout=float(config.get("dropout", 0.25)),
        )
    return build_model(config, num_classes)


def _subset(indices: np.ndarray, labels: np.ndarray, size: int | None, seed: int) -> np.ndarray:
    if size is None or int(size) >= indices.size:
        return indices
    _, selected = stratified_split(labels[indices], int(size), seed)
    return indices[selected]


def _loader(
    dataset,
    config: Mapping[str, Any],
    *,
    shuffle: bool,
    seed: int,
    batch_size: int | None = None,
) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"] if batch_size is None else batch_size),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", True)),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
        drop_last=bool(config.get("drop_last", False)) if shuffle else False,
    )


@torch.inference_mode()
def _epoch_predictions(model, ema_model, loader, device, positions: Mapping[int, int], classes: int):
    # The official robust stage performs this no-grad snapshot while both
    # networks are still in train mode.  In particular, BatchNorm buffers are
    # allowed to follow the source implementation during the snapshot pass.
    model.train()
    ema_model.train()
    size = len(positions)
    probabilities = torch.zeros((size, classes), device=device)
    ema_probabilities = torch.zeros_like(probabilities)
    targets = torch.zeros(size, dtype=torch.long, device=device)
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        indices = [positions[int(value)] for value in batch["index"]]
        rows = torch.as_tensor(indices, dtype=torch.long, device=device)
        probabilities[rows] = torch.softmax(model(inputs), dim=1)
        ema_probabilities[rows] = torch.softmax(ema_model(inputs), dim=1)
        targets[rows] = batch["target"].to(device, non_blocking=True)
    return probabilities, ema_probabilities, targets


def _confidence_penalty(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1).clamp_min(1e-12)
    return (probabilities * probabilities.log()).sum(dim=1).mean()


def _train_warmup(model, ema, optimizer, loader, device) -> tuple[float, float]:
    model.train()
    loss_sum = correct = samples = 0.0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets) + _confidence_penalty(logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(model)
        count = targets.numel()
        loss_sum += float(loss.detach()) * count
        correct += float(logits.argmax(1).eq(targets).sum())
        samples += count
    return loss_sum / samples, correct / samples


def _train_robust(
    model,
    ema,
    optimizer,
    loader,
    device,
    *,
    clean_by_index: Mapping[int, bool],
    pseudo_by_index: Mapping[int, int],
    weight_by_index: Mapping[int, float],
    regularizer: FINERegularizer,
    alpha: float,
) -> tuple[float, float]:
    model.train()
    loss_sum = correct = samples = 0.0
    for batch in loader:
        weak = batch["input"].to(device, non_blocking=True)
        strong = batch["strong_input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        indices = [int(value) for value in batch["index"]]
        clean = torch.as_tensor(
            [clean_by_index[index] for index in indices], dtype=torch.bool, device=device
        )
        pseudo = torch.as_tensor(
            [pseudo_by_index[index] for index in indices], dtype=torch.long, device=device
        )
        weights = torch.as_tensor(
            [weight_by_index[index] for index in indices], dtype=torch.float32, device=device
        )
        logits = model(weak)
        strong_logits = model(strong)
        clean_loss = (
            F.cross_entropy(logits[clean], targets[clean])
            if bool(clean.any())
            else logits.sum() * 0.0
        )
        ssl_loss = (F.cross_entropy(strong_logits, pseudo, reduction="none") * weights).mean()
        fine_loss = regularizer(
            logits,
            targets,
            rejected_mask=~clean,
            pseudo_labels=pseudo,
        )
        loss = clean_loss + float(alpha) * ssl_loss + fine_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(model)
        count = targets.numel()
        loss_sum += float(loss.detach()) * count
        correct += float(logits.argmax(1).eq(targets).sum())
        samples += count
    return loss_sum / samples, correct / samples


def run_fine_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *,
    context: RunContext | None = None,
) -> Path:
    """Run the official warm-up -> SED/SCR -> FINE lifecycle."""

    config = deepcopy(config)
    seed = int(config.get("seed", 123))
    seed_everything(seed)
    device = resolve_device(str(config["trainer"].get("device", "auto")))
    run_dir = (
        Path(resume).resolve().parent
        if resume is not None
        else Path(output_dir or Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None if resume is None else read_checkpoint(resume, "cpu")
    if checkpoint is not None and checkpoint.get("config") != config:
        raise ValueError("FINE resume configuration mismatch")

    data_config = config["data"]
    if str(data_config.get("name", "cifar100")).lower() != "cifar100":
        raise ValueError("FINE CIFAR runner currently supports CIFAR-100/CIFAR-100N")
    train_data = load_cifar100(data_config.get("root"), "train")
    test_data = load_cifar100(data_config.get("root"), "test")
    validation_size = int(data_config.get("validation_size", 0))
    if validation_size == 0:
        train_indices = np.arange(len(train_data), dtype=np.int64)
        validation_indices = np.empty(0, dtype=np.int64)
    else:
        train_indices, validation_indices = train_validation_split(
            train_data.labels, validation_size, seed
        )
    train_indices = _subset(train_indices, train_data.labels, data_config.get("max_train_samples"), seed + 1)
    validation_indices = _subset(validation_indices, train_data.labels, data_config.get("max_validation_samples"), seed + 2)
    test_indices = _subset(np.arange(len(test_data)), test_data.labels, data_config.get("max_test_samples"), seed + 3)
    manifest, manifest_path = prepare_noise_manifest(
        config,
        dataset="cifar100",
        clean_targets=train_data.labels[train_indices],
        global_indices=train_indices,
        num_classes=100,
        run_dir=run_dir,
        checkpoint_payload=checkpoint,
        dataset_targets=train_data.labels,
    )
    targets_by_index = None if manifest is None else {
        int(index): int(target)
        for index, target in zip(manifest.global_indices, manifest.noisy_targets)
    }
    noise_metadata = None
    if manifest is not None:
        assert manifest_path is not None
        noise_metadata = checkpoint_noise_metadata(
            manifest,
            manifest_path,
            run_dir,
            effective_subset_actual_rate(manifest, train_indices),
            mode=noise_mode(config),
        )
    normalization = data_config.get("normalization", {}) or {}
    mean = normalization.get("mean")
    std = normalization.get("std")
    weak_transform = build_cifar_transform(
        True,
        bool(data_config.get("augment", True)),
        normalization_mean=mean,
        normalization_std=std,
    )
    strong_transform = build_strong_cifar_transform(
        mean=mean or (0.49139968, 0.48215827, 0.44653124),
        std=std or (0.24703233, 0.24348505, 0.26158768),
        magnitude=int(data_config.get("strong_magnitude", 10)),
        policy=str(data_config.get("strong_policy", "official_cifar10")),
    )
    train_set = IndexedMultiViewCifarDataset(
        train_data,
        train_indices,
        weak_transform=weak_transform,
        strong_transform=strong_transform,
        targets_by_index=targets_by_index,
    )
    evaluation_transform = build_cifar_transform(
        False, normalization_mean=mean, normalization_std=std
    )
    test_set = TorchCifarDataset(test_data, test_indices, transform=evaluation_transform)
    validation_set = (
        TorchCifarDataset(train_data, validation_indices, transform=evaluation_transform)
        if validation_indices.size
        else None
    )
    loader_config = config["loader"]
    evaluation_batch_size = int(
        config.get("evaluation", {}).get("batch_size", loader_config["batch_size"])
    )
    validation_loader = (
        None
        if validation_set is None
        else _loader(
            validation_set,
            loader_config,
            shuffle=False,
            seed=seed,
            batch_size=evaluation_batch_size,
        )
    )
    test_loader = _loader(
        test_set,
        loader_config,
        shuffle=False,
        seed=seed,
        batch_size=evaluation_batch_size,
    )

    model = _build_fine_model(config["model"], 100).to(device)
    optimizer_config = dict(config["optimizer"])
    fine_config = config["fine"]
    warmup_epochs = int(fine_config["warmup_epochs"])
    epochs = int(config["trainer"]["epochs"])
    if not 0 <= warmup_epochs <= epochs:
        raise ValueError("FINE warmup_epochs must be in [0, epochs]")
    optimizer_config["lr"] = float(fine_config.get("warmup_lr", optimizer_config["lr"]))
    optimizer = build_optimizer(model, optimizer_config)
    ema = ModelEMA(
        model,
        float(fine_config.get("ema_momentum", 0.95)),
        update_buffers=False,
    )
    ema.model.to(device)
    # ModelEMA intentionally defaults to eval mode for generic teacher use;
    # FINE's official ``model_ema`` remains in train mode during snapshots.
    ema.model.train()
    scs = SelfAdaptiveClassSelector(
        100,
        float(fine_config.get("momentum_scs", 0.999)),
        quantile=float(fine_config.get("quantile", 0.8)) if fine_config.get("use_quantile", True) else None,
        maximum_threshold=float(fine_config.get("maximum_threshold", 0.95)) if fine_config.get("clip_threshold", True) else None,
    )
    scr = SelfAdaptiveConfidenceReweighting(
        100, float(fine_config.get("momentum_scr", 0.99))
    )
    regularizer = FINERegularizer(
        beta=float(fine_config.get("beta", 0.1)),
        gamma=float(fine_config.get("gamma", 0.002)),
        probability_floor=float(fine_config.get("probability_floor", 1e-7)),
        seed=seed,
    )
    scheduler = None
    start_epoch = 0
    rows: list[dict[str, Any]] = []
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"])
        scs.load_state_dict(checkpoint["scs"])
        scr.load_state_dict(checkpoint["scr"])
        regularizer.load_state_dict(checkpoint["regularizer"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        exact_loader_resume = validate_loader_stream(
            checkpoint.get("loader_stream"),
            base_seed=seed,
            namespace="fine",
            next_epoch=start_epoch,
        )
        if not exact_loader_resume:
            warnings.warn(
                "Legacy FINE checkpoint has no loader_stream metadata; "
                "epoch-boundary input-stream resume is best effort.",
                RuntimeWarning,
                stacklevel=2,
            )
        rows = list(checkpoint.get("metrics", []))

    criterion = CrossEntropyLoss().to(device)
    positions = {int(index): position for position, index in enumerate(train_indices)}
    metrics_path = run_dir / "metrics.jsonl"
    session = context.session if context is not None and context.state.get("lifecycle_active") else None
    if session is not None:
        session.start_phase("fine_training", total_units=epochs)
    if start_epoch >= warmup_epochs:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=float(config.get("scheduler", {}).get("eta_min", 5e-4)),
        )
        if checkpoint is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
    for epoch in range(start_epoch, epochs):
        if epoch == warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = float(config["optimizer"]["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=float(config.get("scheduler", {}).get("eta_min", 5e-4)),
            )
        if epoch < warmup_epochs:
            train_loader = build_epoch_loader(
                train_set,
                loader_config,
                base_seed=seed,
                namespace="fine.train",
                epoch=epoch,
                drop_last=bool(loader_config.get("drop_last", False)),
            )
            train_loss, train_accuracy = _train_warmup(model, ema, optimizer, train_loader, device)
            clean_ratio = 1.0
        else:
            snapshot_loader = build_epoch_loader(
                train_set,
                loader_config,
                base_seed=seed,
                namespace="fine.snapshot",
                epoch=epoch,
                shuffle=False,
                drop_last=False,
            )
            probabilities, ema_probabilities, snapshot_targets = _epoch_predictions(
                model, ema.model, snapshot_loader, device, positions, 100
            )
            clean_mask = scs.select_epoch(probabilities, snapshot_targets)
            weights = scr.weights(ema_probabilities)
            pseudo = ema_probabilities.argmax(dim=1)
            train_loader = build_epoch_loader(
                train_set,
                loader_config,
                base_seed=seed,
                namespace="fine.train",
                epoch=epoch,
                drop_last=bool(loader_config.get("drop_last", False)),
            )
            clean_by_index = {index: bool(clean_mask[position]) for index, position in positions.items()}
            pseudo_by_index = {index: int(pseudo[position]) for index, position in positions.items()}
            weight_by_index = {index: float(weights[position]) for index, position in positions.items()}
            train_loss, train_accuracy = _train_robust(
                model,
                ema,
                optimizer,
                train_loader,
                device,
                clean_by_index=clean_by_index,
                pseudo_by_index=pseudo_by_index,
                weight_by_index=weight_by_index,
                regularizer=regularizer,
                alpha=float(fine_config.get("alpha", 1.0)),
            )
            clean_ratio = float(clean_mask.float().mean())
        row = {
            "event": "epoch",
            "epoch": epoch + 1,
            "phase": "warmup" if epoch < warmup_epochs else "robust",
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_metric": "unavailable",
            "selected_ratio": clean_ratio,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if validation_loader is not None:
            validation = evaluate_classification(
                model, validation_loader, criterion, device
            )
            row.update({
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_metric": "clean_validation_accuracy",
            })
            row = standardize_epoch_row(row)
        rows.append(row)
        if session is not None:
            session.log_epoch(
                epoch + 1,
                phase=str(row.get("phase", "train")),
                **{key: value for key, value in row.items()
                   if key not in {"event", "epoch", "phase", "seq"}},
            )
        else:
            metrics_path.write_text(
                "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
                encoding="utf-8",
            )
        if scheduler is not None:
            scheduler.step()
        atomic_save({
            "format_version": 1,
            "method": "fine_sed",
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "ema": ema.state_dict(),
            "scs": scs.state_dict(),
            "scr": scr.state_dict(),
            "regularizer": regularizer.state_dict(),
            "completed_epoch": epoch,
            "metrics": rows,
            "rng_state": capture_rng_state(),
            "noise": noise_metadata,
            "loader_stream": loader_stream_metadata(
                base_seed=seed, namespace="fine", next_epoch=epoch + 1
            ),
        }, run_dir / "last.pt")
    test = evaluate_classification(model, test_loader, criterion, device)
    final = {
        "event": "final",
        "method": "fine_sed",
        "completed_epochs": epochs,
        "validation_metric": (
            "clean_validation_accuracy" if validation_loader is not None else "unavailable"
        ),
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
    }
    if session is None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final, sort_keys=True) + "\n")
    if session is not None:
        session.end_phase("fine_training", completed_units=max(0, epochs - start_epoch))
        session.emit(
            "final",
            phase="evaluation",
            **{key: value for key, value in final.items() if key != "event"},
        )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if rows and validation_loader is not None:
        write_training_curves_svg(rows, run_dir / "training_curves.svg")
    return run_dir


__all__ = ["run_fine_experiment"]
