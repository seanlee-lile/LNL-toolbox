from __future__ import annotations

"""Two-stage CAL workflow without clean labels in training batches."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.cal import (
    cal_all_class_losses,
    cal_objective,
    cores2_adjusted_losses,
    resolve_confidence_weight,
)
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.noise.cal import CALProxyArtifact, build_cal_proxy_artifact
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state, read_checkpoint, restore_rng_state
from lnl_toolbox.training.interfaces import RunContext
from lnl_toolbox.training.experiment import (
    build_alpha_scaled_scheduler,
    build_optimizer,
)
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.reproduction_data import build_reproduction_model, prepare_noisy_classification
from lnl_toolbox.training.snapshots import collect_posterior_snapshot


def _build_warmup_scheduler(optimizer, config: dict[str, Any], epochs: int):
    """Build the configured scheduler for the separate CORES² warm-up."""

    return build_alpha_scaled_scheduler(optimizer, config.get("scheduler"))


def _assert_finite_warmup_state(model, loss: torch.Tensor) -> None:
    if not bool(torch.isfinite(loss.detach()).item()):
        raise ValueError("CAL warm-up produced a non-finite loss")
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise ValueError(
                f"CAL warm-up produced non-finite parameter: {name}"
            )


def _assert_finite_warmup_gradients(model) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError(
                f"CAL warm-up produced non-finite gradient: {name}"
            )


def run_cal_experiment(config: dict[str, Any], output_dir=None, resume=None, *, context: RunContext | None = None) -> Path:
    config = deepcopy(config); seed = int(config.get("seed", 1)); seed_everything(seed)
    device = resolve_device(str(config.get("trainer", {}).get("device", "auto")))
    run_dir = Path(resume).resolve().parent if resume else Path(output_dir or Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_noisy_classification(config, run_dir, seed)
    classes = data.num_classes
    noisy_prior = torch.as_tensor(np.bincount(data.noisy_targets, minlength=classes) / len(data.noisy_targets), dtype=torch.float32, device=device)
    proxy_path = run_dir / "cal_proxy_artifact.npz"
    payload = read_checkpoint(resume, device) if resume else None
    if payload is None:
        warmup = build_reproduction_model(config["model"], config["data"], classes).to(device)
        warmup_optimizer = build_optimizer(warmup, config["optimizer"])
        warmup_cfg = dict(config["warmup"])
        warmup_epochs = int(warmup_cfg["epochs"])
        warmup_scheduler = _build_warmup_scheduler(
            warmup_optimizer, config, warmup_epochs
        )
        for epoch in range(warmup_epochs):
            warmup.train()
            confidence_weight = resolve_confidence_weight(
                epoch,
                float(warmup_cfg["confidence_weight"]),
                warmup_cfg.get("confidence_schedule"),
            )
            for batch in data.train_loader:
                inputs, targets = batch["input"].to(device), batch["target"].to(device)
                loss = cores2_adjusted_losses(
                    warmup(inputs), targets, noisy_prior, confidence_weight
                ).mean()
                _assert_finite_warmup_state(warmup, loss)
                warmup_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _assert_finite_warmup_gradients(warmup)
                warmup_optimizer.step()
                _assert_finite_warmup_state(warmup, loss)
            if warmup_scheduler is not None:
                warmup_scheduler.step(resolve_confidence_weight(
                    epoch + 1,
                    float(warmup_cfg["confidence_weight"]),
                    warmup_cfg.get("confidence_schedule"),
                ))
        atomic_save(
            {
                "method": "cal_warmup",
                "config": config,
                "model": warmup.state_dict(),
                "optimizer": warmup_optimizer.state_dict(),
                "scheduler": None if warmup_scheduler is None else warmup_scheduler.state_dict(),
                "completed_epoch": warmup_epochs - 1,
                "rng_state": capture_rng_state(),
            },
            run_dir / "cal_warmup.pt",
        )
        snapshot = collect_posterior_snapshot(warmup, data.snapshot_loader, device, dataset=data.dataset, split="train")
        losses = []
        loss_indices = []
        warmup.eval()
        with torch.inference_mode():
            for batch in data.snapshot_loader:
                logits = warmup(batch["input"].to(device)); targets = batch["target"].to(device)
                losses.append(cores2_adjusted_losses(
                    logits,
                    targets,
                    noisy_prior,
                    resolve_confidence_weight(
                        warmup_epochs - 1,
                        float(warmup_cfg["confidence_weight"]),
                        warmup_cfg.get("confidence_schedule"),
                    ),
                ).cpu().numpy())
                loss_indices.append(batch["index"].cpu().numpy())
        all_losses = np.concatenate(losses)
        all_indices = np.concatenate(loss_indices)
        order = np.argsort(all_indices, kind="stable")
        if not np.array_equal(all_indices[order], snapshot.global_indices):
            raise ValueError("CAL adjusted-loss indices do not match posterior snapshot")
        proxy = build_cal_proxy_artifact(snapshot, all_losses[order], lower_threshold=float(config["sieve"]["lower_threshold"]), upper_threshold=float(config["sieve"]["upper_threshold"])); proxy.save(proxy_path)
    else:
        if payload.get("method") != "cal" or payload.get("config") != config: raise ValueError("CAL resume identity mismatch")
        proxy = CALProxyArtifact.load(proxy_path)
        if proxy.artifact_hash != payload["proxy_hash"]: raise ValueError("CAL proxy resume mismatch")
    model = build_reproduction_model(config["model"], config["data"], classes).to(device)
    optimizer = build_optimizer(model, config["optimizer"]); epochs = int(config["trainer"]["epochs"])
    scheduler = build_alpha_scaled_scheduler(optimizer, config.get("scheduler"))
    cal_cfg = dict(config["cal"])
    cal_schedule = cal_cfg.get("confidence_schedule")
    criterion = CrossEntropyLoss().to(device); means = torch.zeros(classes, classes, device=device); start = 0; rows = []
    retained = proxy.sample_status != 2
    proxy_prior_np = np.bincount(proxy.proxy_targets[retained], minlength=classes).astype(np.float64)
    if proxy_prior_np.sum() <= 0:
        raise ValueError("CAL proxy artifact retained no samples")
    proxy_prior_np /= proxy_prior_np.sum()
    proxy_prior = torch.as_tensor(proxy_prior_np, dtype=torch.float32, device=device)
    if payload is not None:
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None: scheduler.load_state_dict(payload["scheduler"])
        means = payload["reference_loss_means"].to(device); rows = list(payload.get("metrics", [])); start = int(payload["completed_epoch"]) + 1; restore_rng_state(payload["rng_state"])
    session = context.session if context is not None and context.state.get("lifecycle_active") else None
    if session is not None:
        session.start_phase("risk_correction", total_units=epochs)
    for epoch in range(start, epochs):
        model.train(); total = correct = 0; loss_sum = 0.0
        epoch_loss_sums = torch.zeros_like(means)
        epoch_class_counts = torch.zeros(classes, device=device)
        for batch in data.train_loader:
            inputs, targets, indices = batch["input"].to(device), batch["target"].to(device), batch["index"].to(device)
            proxy_targets, mask, _ = proxy.lookup(indices); logits = model(inputs)
            confidence_weight = resolve_confidence_weight(
                epoch,
                float(cal_cfg["confidence_weight"]),
                cal_schedule,
            )
            loss, _ = cal_objective(logits, targets, proxy_targets, mask, noisy_prior, proxy_prior, means, confidence_weight=confidence_weight)
            detached_all_losses = cal_all_class_losses(logits.detach())
            for proxy_class in range(classes):
                class_mask = mask & proxy_targets.eq(proxy_class)
                epoch_loss_sums[proxy_class] += detached_all_losses[class_mask].sum(dim=0)
                epoch_class_counts[proxy_class] += class_mask.sum()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += targets.numel(); loss_sum += float(loss.detach()) * targets.numel(); correct += int(logits.argmax(1).eq(targets).sum())
        observed_classes = epoch_class_counts > 0
        means[observed_classes] = epoch_loss_sums[observed_classes] / epoch_class_counts[observed_classes, None]
        means = means.detach(); validation = evaluate_classification(model, data.validation_loader, criterion, device); test = evaluate_classification(model, data.test_loader, criterion, device)
        row = standardize_epoch_row({"epoch": epoch + 1, "train_loss": loss_sum / total, "train_accuracy": correct / total, "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"], "test_loss": test["loss"], "test_accuracy": test["accuracy"], "learning_rate": optimizer.param_groups[0]["lr"], "method": "cal"}); rows.append(row)
        if session is not None:
            session.log_epoch(epoch + 1, phase="risk_correction", **{key: value for key, value in row.items() if key not in {"event", "epoch", "phase", "seq"}})
        if scheduler is not None:
            scheduler.step(resolve_confidence_weight(
                epoch + 1,
                float(cal_cfg["confidence_weight"]),
                cal_schedule,
            ))
        atomic_save({"method": "cal", "config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None if scheduler is None else scheduler.state_dict(), "completed_epoch": epoch, "metrics": rows, "proxy_hash": proxy.artifact_hash, "reference_loss_means": means.cpu(), "rng_state": capture_rng_state()}, run_dir / "last.pt")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if session is None:
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        session.end_phase("risk_correction", completed_units=max(0, epochs - start))
        session.emit("final", phase="evaluation", method="cal", completed_epochs=epochs,
                     test_accuracy=rows[-1].get("test_accuracy") if rows else None)
    if rows: write_training_curves_svg(rows, run_dir / "training_curves.svg")
    return run_dir


__all__ = ["run_cal_experiment"]
