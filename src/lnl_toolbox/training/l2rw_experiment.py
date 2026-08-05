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
from torch.utils.data import Subset

from lnl_toolbox.algorithms.l2rw import meta_reweight
from lnl_toolbox.data.trusted import TrustedSupervisionManifest, TrustedValidationProvider
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state, read_checkpoint, restore_rng_state
from lnl_toolbox.training.experiment import build_optimizer, build_scheduler
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.reproduction_data import build_reproduction_model, prepare_noisy_classification


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
    data = prepare_noisy_classification(config, run_dir, seed)
    manifest = _trusted_manifest(config, data.validation_loader.dataset, run_dir, data.dataset)
    trusted_base = data.validation_loader.dataset
    base_indices = getattr(trusted_base, "indices", None)
    if base_indices is not None and len(base_indices) != manifest.global_indices.size:
        position = {int(index): offset for offset, index in enumerate(base_indices)}
        try:
            selected_positions = [position[int(index)] for index in manifest.global_indices]
        except KeyError as exc:
            raise ValueError("trusted manifest is outside the validation pool") from exc
        trusted_base = Subset(trusted_base, selected_positions)
    provider = TrustedValidationProvider(trusted_base, manifest)
    trusted_config = dict(config["trusted_validation"])
    trusted_loader = provider.loader(
        batch_size=int(trusted_config.get("batch_size", config.get("loader", {}).get("batch_size", 128))),
        shuffle=True,
        seed=int(trusted_config.get("seed", seed + 1000)),
        num_workers=int(trusted_config.get("num_workers", 0)),
    )
    model = build_reproduction_model(config["model"], config["data"], data.num_classes).to(device)
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
        if scheduler is not None: scheduler.load_state_dict(payload["scheduler"])
        restore_rng_state(payload["rng_state"])
        if data.train_loader.generator is not None:
            data.train_loader.generator.set_state(payload["train_loader_rng"])
        if trusted_loader.generator is not None:
            trusted_loader.generator.set_state(payload["trusted_loader_rng"])
        start = int(payload["completed_epoch"]) + 1
        rows = list(payload.get("metrics", []))
    alpha = float(config["meta"]["virtual_learning_rate"])
    global_step = int(payload.get("global_step", 0)) if payload is not None else 0
    step_milestones = [int(value) for value in config.get("scheduler", {}).get("step_milestones", [])]
    while (global_step < max_steps) if max_steps else (start < epochs):
        epoch = start
        model.train(); trusted_iterator = iter(trusted_loader)
        total = correct = 0; loss_sum = weight_sum = positive_sum = 0.0
        for batch in data.train_loader:
            if max_steps and global_step >= max_steps:
                break
            if step_milestones:
                decays = sum(global_step >= milestone for milestone in step_milestones)
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
                model, inputs, targets, trusted_inputs, trusted_targets,
                virtual_learning_rate=alpha,
            )
            logits = model(inputs)
            per_sample = criterion(logits, targets)
            objective = torch.sum(weights.sample_weights.to(per_sample) * per_sample)
            optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
            global_step += 1
            count = targets.numel(); total += count; loss_sum += float(objective.detach()) * count
            correct += int(logits.argmax(1).eq(targets).sum())
            weight_sum += float(weights.sample_weights.sum()); positive_sum += weights.metrics["positive_weight_count"]
        validation = evaluate_classification(model, data.validation_loader, criterion, device)
        test = evaluate_classification(model, data.test_loader, criterion, device)
        row = standardize_epoch_row({
            "epoch": epoch + 1, "train_loss": loss_sum / total,
            "train_accuracy": correct / total, "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"], "test_loss": test["loss"],
            "test_accuracy": test["accuracy"], "learning_rate": optimizer.param_groups[0]["lr"],
            "method": "l2rw", "mean_weight_sum": weight_sum / len(data.train_loader),
            "mean_positive_weights": positive_sum / len(data.train_loader),
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
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "completed_epoch": epoch, "global_step": global_step, "metrics": rows,
            "trusted_fingerprint": manifest.fingerprint,
            "train_loader_rng": None if data.train_loader.generator is None else data.train_loader.generator.get_state(),
            "trusted_loader_rng": None if trusted_loader.generator is None else trusted_loader.generator.get_state(),
            "rng_state": capture_rng_state(),
        }, run_dir / "last.pt")
        start += 1
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if rows: write_training_curves_svg(rows, run_dir / "training_curves.svg")
    return run_dir


__all__ = ["run_l2rw_experiment"]
