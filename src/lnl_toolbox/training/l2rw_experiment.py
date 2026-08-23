from __future__ import annotations

"""Dedicated L2RW bilevel training lifecycle."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.l2rw import meta_reweight
from lnl_toolbox.data.trusted import TrustedSupervisionManifest
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state, read_checkpoint, restore_rng_state
from lnl_toolbox.training.experiment import build_optimizer, build_scheduler
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.reproduction_data import build_reproduction_model
from lnl_toolbox.training.data_service import prepare_experiment_data


def _official_partition(size: int, parts: list[int], seed: int) -> list[np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    random = np.random.RandomState(seed)
    random.shuffle(indices)
    result, offset = [], 0
    for part in parts:
        result.append(indices[offset:offset + int(part)].copy())
        offset += int(part)
    if offset != size:
        raise ValueError("official L2RW partition sizes do not cover the dataset")
    return result


def _official_train_subsets(
    train_positions: np.ndarray, num_clean: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Match the official generated CIFAR clean/noisy train partition."""

    positions = np.asarray(train_positions, dtype=np.int64)
    noisy, clean = _official_partition(
        int(positions.size),
        [int(positions.size) - int(num_clean), int(num_clean)],
        seed,
    )
    return positions[noisy], positions[clean]


def _official_flip_labels(labels: np.ndarray, num_classes: int, rate: float, seed: int):
    values = np.asarray(labels, dtype=np.int64)
    num_noise = int(values.size * float(rate))
    random = np.random.RandomState(int(seed) + 1)
    replacements = np.floor(random.uniform(0.0, num_classes - 1, [num_noise])).astype(np.int64)
    noisy = np.concatenate((replacements, values[num_noise:]))
    clean_mask = np.concatenate((np.zeros(num_noise, dtype=np.int64), np.ones(values.size - num_noise, dtype=np.int64)))
    order = np.arange(values.size, dtype=np.int64)
    random.shuffle(order)
    return noisy[order], clean_mask[order], order


def _official_l2rw_transform(training: bool, augment: bool = True):
    from torchvision import transforms
    operations = []
    if training and augment:
        operations.extend((transforms.Pad(4), transforms.RandomCrop(32), transforms.RandomHorizontalFlip()))
    operations.extend((transforms.ToTensor(), transforms.Lambda(lambda value: (value - 0.5) * 2.0)))
    return transforms.Compose(operations)


def _trusted_manifest(
    config: Mapping[str, Any], dataset, run_dir: Path, dataset_name: str
) -> TrustedSupervisionManifest:
    trusted = config.get("trusted_validation")
    if not isinstance(trusted, Mapping):
        raise ValueError("L2RW requires an explicit trusted_validation configuration")
    source = str(trusted.get("source", "")).strip().lower()
    if source == "audited_manifest":
        path = trusted.get("manifest")
        if not path:
            raise ValueError("audited trusted supervision requires manifest path")
        manifest = TrustedSupervisionManifest.load(path)
    elif source == "synthetic_fixture":
        if dataset_name != "synthetic_multiclass":
            raise ValueError("synthetic_fixture trusted supervision is smoke-only")
        indices, targets = [], []
        for position in range(len(dataset)):
            sample = dataset[position]
            indices.append(int(sample["index"])); targets.append(int(sample["target"]))
        counts = np.bincount(np.asarray(targets, dtype=np.int64))
        manifest = TrustedSupervisionManifest(
            np.asarray(indices), np.asarray(targets), dataset_name,
            "trusted_validation", "synthetic_fixture",
            bool(counts.size > 0 and np.all(counts == counts[0])),
            {"purpose": "L2RW deterministic smoke only"},
        )
    else:
        raise ValueError(
            "trusted_validation.source must be audited_manifest or synthetic_fixture"
        )
    if not manifest.balanced:
        raise ValueError("L2RW trusted supervision must be class-balanced")
    local = run_dir / "trusted_validation_manifest.npz"
    if local.is_file():
        existing = TrustedSupervisionManifest.load(local)
        if existing.fingerprint != manifest.fingerprint:
            raise ValueError("L2RW run-local trusted manifest identity mismatch")
    else:
        manifest.save(local)
    return manifest


def run_l2rw_experiment(
    config: dict[str, Any], output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    config = deepcopy(config)
    seed = int(config.get("seed", 1)); seed_everything(seed)
    device = resolve_device(str(config.get("trainer", {}).get("device", "auto")))
    run_dir = (
        Path(resume).resolve().parent if resume else
        Path(output_dir or Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    trusted_config = dict(config.get("trusted_validation", {}))
    # Uber's CIFAR launcher has two independent seeds: FLAGS.seed controls
    # generated data, while the protobuf seed controls TensorFlow/model RNG.
    # Keep the top-level seed for the latter and make the data seed explicit.
    data_seed = int(
        config.get("data", {}).get(
            "seed", config.get("noise", {}).get("seed", 0)
        )
    )
    official_generated = str(trusted_config.get("source", "")).strip().lower() == "official_generated"
    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.TRUSTED_VALIDATION, DataRole.CLEAN_VALIDATION, DataRole.TEST}),
            train_drop_last=True if official_generated else None,
        ),
        run_dir=run_dir,
        seed=data_seed if official_generated else seed,
    )
    num_classes, dataset_name = prepared.num_classes, prepared.dataset
    input_seed = int(config.get("data", {}).get("input_seed", seed))
    train_loader = prepared.loader(DataRole.TRAIN, generator_seed=input_seed)
    validation_loader = prepared.loader(DataRole.CLEAN_VALIDATION, shuffle=False, generator_seed=input_seed)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False, generator_seed=input_seed)
    trusted_base = prepared.dataset_for(DataRole.TRUSTED_VALIDATION)
    if official_generated:
        trusted_indices = np.asarray(getattr(trusted_base, "indices"), dtype=np.int64)
        trusted_targets = np.asarray([int(trusted_base[index]["target"]) for index in range(len(trusted_base))], dtype=np.int64)
        counts = np.unique(trusted_targets, return_counts=True)[1]
        manifest = TrustedSupervisionManifest(
            trusted_indices, trusted_targets, dataset_name, "train_clean", "official_generated",
            bool(counts.size and np.all(counts == counts[0])),
            {"source": "generate_noisy_cifar_data.py", "seed": data_seed},
        )
        trusted_path = run_dir / "trusted_validation_manifest.npz"
        if trusted_path.is_file() and TrustedSupervisionManifest.load(trusted_path).fingerprint != manifest.fingerprint:
            raise ValueError("official trusted manifest identity mismatch")
        if not trusted_path.is_file():
            manifest.save(trusted_path)
    else:
        manifest = _trusted_manifest(config, trusted_base, run_dir, dataset_name)
    trusted_loader = prepared.loader(
        DataRole.TRUSTED_VALIDATION,
        batch_size=int(trusted_config.get("batch_size", config.get("loader", {}).get("batch_size", 128))),
        generator_seed=int(trusted_config.get("input_seed", seed + 1000)),
    )
    model = build_reproduction_model(config["model"], config["data"], num_classes).to(device)
    meta_model = model
    if official_generated and str(config["model"].get("name", "")).lower() == "l2rw_resnet32":
        # The official assigned-weight replicas use batch statistics and only
        # an offset beta; model C keeps the regular moving-statistics BN.
        from lnl_toolbox.models.cifar_resnet import (
            L2RWResNet32,
            share_l2rw_meta_parameters,
        )

        meta_model = L2RWResNet32(
            num_classes,
            int(config["model"].get("base_width", 16)),
            meta_batch_statistics=True,
        ).to(device)
        # The official reweight_autodiff graph reuses model C's current
        # variables for the assigned-weight A/B branches.  Keep that exact
        # parameter identity while retaining the A/B BatchNorm semantics.
        share_l2rw_meta_parameters(model, meta_model)
    optimizer = build_optimizer(model, config["optimizer"])
    epochs = int(config["trainer"].get("epochs", 1))
    max_steps = int(config["trainer"].get("max_steps", 0))
    scheduler = build_scheduler(optimizer, config.get("scheduler"), epochs)
    criterion = CrossEntropyLoss().to(device)
    start = 0; rows: list[dict[str, Any]] = []
    payload = read_checkpoint(resume, device) if resume else None
    if payload is not None:
        if payload.get("method") != "l2rw" or payload.get("config") != config:
            raise ValueError("L2RW resume configuration mismatch")
        if payload.get("trusted_fingerprint") != manifest.fingerprint:
            raise ValueError("L2RW trusted supervision resume mismatch")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        if meta_model is not model and payload.get("meta_model") is not None:
            meta_model.load_state_dict(payload["meta_model"])
        if scheduler is not None: scheduler.load_state_dict(payload["scheduler"])
        restore_rng_state(payload["rng_state"])
        if train_loader.generator is not None:
            train_loader.generator.set_state(payload["train_loader_rng"])
        if trusted_loader.generator is not None:
            trusted_loader.generator.set_state(payload["trusted_loader_rng"])
        start = int(payload["completed_epoch"]) + 1
        rows = list(payload.get("metrics", []))
    meta_config = dict(config.get("meta", {}))
    alpha = float(meta_config["virtual_learning_rate"])
    meta_implementation = str(meta_config.get("implementation", "paper"))
    meta_weight_decay = float(config["optimizer"].get("weight_decay", 0.0))
    global_step = int(payload.get("global_step", 0)) if payload is not None else 0
    step_milestones = [int(value) for value in config.get("scheduler", {}).get("step_milestones", [])]
    while (global_step < max_steps) if max_steps else (start < epochs):
        epoch = start
        model.train(); meta_model.train(); trusted_iterator = iter(trusted_loader)
        total = correct = 0; loss_sum = weight_sum = positive_sum = 0.0
        for batch in train_loader:
            if max_steps and global_step >= max_steps:
                break
            if step_milestones:
                # Uber's FixedLearnRateScheduler.step(niter) changes the
                # rate when niter + 1 reaches a decay step, before that
                # optimization update is applied.
                decays = sum(global_step + 1 >= milestone for milestone in step_milestones)
                learning_rate = float(config["optimizer"]["lr"]) * float(config.get("scheduler", {}).get("gamma", 0.1)) ** decays
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            try:
                trusted_batch = next(trusted_iterator)
            except StopIteration:
                trusted_iterator = iter(trusted_loader)
                trusted_batch = next(trusted_iterator)
            inputs = batch["input"].to(device); targets = batch["target"].to(device)
            trusted_inputs = trusted_batch["input"].to(device); trusted_targets = trusted_batch["target"].to(device)
            weights = meta_reweight(
                meta_model, inputs, targets, trusted_inputs, trusted_targets,
                virtual_learning_rate=alpha,
                weight_decay=meta_weight_decay,
                implementation=meta_implementation,
            )
            logits = model(inputs)
            per_sample = criterion(logits, targets)
            objective = torch.sum(weights.sample_weights.to(per_sample) * per_sample)
            optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
            global_step += 1
            count = targets.numel(); total += count; loss_sum += float(objective.detach()) * count
            correct += int(logits.argmax(1).eq(targets).sum())
            weight_sum += float(weights.sample_weights.sum()); positive_sum += weights.metrics["positive_weight_count"]
        validation = evaluate_classification(model, validation_loader, criterion, device)
        test = evaluate_classification(model, test_loader, criterion, device)
        row = standardize_epoch_row({
            "epoch": epoch + 1, "train_loss": loss_sum / total,
            "train_accuracy": correct / total, "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"], "test_loss": test["loss"],
            "test_accuracy": test["accuracy"], "learning_rate": optimizer.param_groups[0]["lr"],
            "method": "l2rw", "mean_weight_sum": weight_sum / len(train_loader),
            "mean_positive_weights": positive_sum / len(train_loader),
            "trusted_fingerprint": manifest.fingerprint, "global_step": global_step,
        })
        rows.append(row)
        print(
            f"L2RW epoch {epoch + 1}/{epochs} steps={global_step} "
            f"loss={row['train_loss']:.5f} val={row['validation_accuracy']:.4f} "
            f"test={row['test_accuracy']:.4f}",
            flush=True,
        )
        if scheduler is not None: scheduler.step()
        atomic_save({
            "method": "l2rw", "config": config, "model": model.state_dict(),
            "meta_model": None if meta_model is model else meta_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "completed_epoch": epoch, "global_step": global_step, "metrics": rows,
            "trusted_fingerprint": manifest.fingerprint,
            "train_loader_rng": None if train_loader.generator is None else train_loader.generator.get_state(),
            "trusted_loader_rng": None if trusted_loader.generator is None else trusted_loader.generator.get_state(),
            "rng_state": capture_rng_state(),
        }, run_dir / "last.pt")
        start += 1
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if rows: write_training_curves_svg(rows, run_dir / "training_curves.svg")
    return run_dir


__all__ = ["run_l2rw_experiment"]
