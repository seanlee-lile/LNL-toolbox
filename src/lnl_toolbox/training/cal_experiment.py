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
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.training.data_service import prepare_experiment_data
from lnl_toolbox.training.experiment import (
    bind_model_input,
    build_alpha_scaled_scheduler,
    build_optimizer,
)
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.reproduction_data import build_reproduction_model
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


def _reference_transition_means(
    proxy: CALProxyArtifact,
    train_indices: np.ndarray,
    noisy_targets: np.ndarray,
    num_classes: int,
) -> torch.Tensor:
    """Estimate fixed proxy-to-noisy transition means over retained samples."""

    indices = np.asarray(train_indices, dtype=np.int64)
    targets = np.asarray(noisy_targets, dtype=np.int64)
    if indices.shape != targets.shape or np.unique(indices).size != indices.size:
        raise ValueError("CAL train indices and noisy targets must align uniquely")
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    positions = np.searchsorted(sorted_indices, proxy.global_indices)
    if (
        np.any(positions >= sorted_indices.size)
        or not np.array_equal(sorted_indices[positions], proxy.global_indices)
    ):
        raise ValueError("CAL proxy indices do not align with noisy targets")
    observed = targets[order[positions]]
    retained = proxy.sample_status != 2
    counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(
        counts,
        (proxy.proxy_targets[retained], observed[retained]),
        1.0,
    )
    totals = counts.sum(axis=1, keepdims=True)
    nonempty = totals[:, 0] > 0
    counts[nonempty] /= totals[nonempty]
    return torch.as_tensor(counts, dtype=torch.float32)


def run_cal_experiment(
    config: dict[str, Any], output_dir=None, resume=None, *,
    requirements: DataRequirements | None = None,
) -> Path:
    config = deepcopy(config); seed = int(config.get("seed", 1)); seed_everything(seed)
    device = resolve_device(str(config.get("trainer", {}).get("device", "auto")))
    run_dir = Path(resume).resolve().parent if resume else Path(output_dir or Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if requirements is None:
        from lnl_toolbox.training.runners import resolve_data_requirements

        requirements = resolve_data_requirements(config, expected_runner="cal")
    data = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=seed,
    )
    train_loader = data.loader(DataRole.TRAIN, stream=21)
    target_map = {
        int(index): int(target)
        for index, target in zip(data.train_indices, data.noisy_targets)
    }
    snapshot_loader = data.loader_for_dataset(
        data.dynamic_dataset(
            data.train_indices,
            views=("weak",),
            targets_by_index=target_map,
            training=False,
        ),
        stream=22,
        shuffle=False,
    )
    test_loader = data.loader(DataRole.TEST, stream=24, shuffle=False)
    validation_loader = (
        data.validation_loader(stream=23, shuffle=False)
        if {
            DataRole.CLEAN_VALIDATION,
            DataRole.NOISY_VALIDATION,
        }.intersection(data.available_roles)
        else None
    )
    classes = data.num_classes
    noisy_prior = torch.as_tensor(np.bincount(data.noisy_targets, minlength=classes) / len(data.noisy_targets), dtype=torch.float32, device=device)
    proxy_path = run_dir / "cal_proxy_artifact.npz"
    payload = read_checkpoint(resume, device) if resume else None
    if payload is None:
        warmup = build_reproduction_model(
            bind_model_input(config["model"], data.input_spec), config["data"], classes
        ).to(device)
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
            for batch in train_loader:
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
        snapshot = collect_posterior_snapshot(warmup, snapshot_loader, device, dataset=data.dataset, split="train")
        losses = []
        loss_indices = []
        warmup.eval()
        with torch.inference_mode():
            for batch in snapshot_loader:
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
    transition_means = _reference_transition_means(
        proxy,
        np.asarray(data.train_indices),
        np.asarray(data.noisy_targets),
        classes,
    ).to(device)
    model = build_reproduction_model(
        bind_model_input(config["model"], data.input_spec), config["data"], classes
    ).to(device)
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
        means = payload["reference_loss_means"].to(device)
        saved_transition_means = payload.get("reference_transition_means")
        if saved_transition_means is None or not torch.equal(
            saved_transition_means.cpu(), transition_means.cpu()
        ):
            raise ValueError("CAL reference transition checkpoint mismatch")
        rows = list(payload.get("metrics", [])); start = int(payload["completed_epoch"]) + 1; restore_rng_state(payload["rng_state"])
    for epoch in range(start, epochs):
        model.train(); total = correct = 0; loss_sum = 0.0
        epoch_loss_sums = torch.zeros_like(means)
        epoch_class_counts = torch.zeros(classes, device=device)
        for batch in train_loader:
            inputs, targets, indices = batch["input"].to(device), batch["target"].to(device), batch["index"].to(device)
            proxy_targets, mask, _ = proxy.lookup(indices); logits = model(inputs)
            confidence_weight = resolve_confidence_weight(
                epoch,
                float(cal_cfg["confidence_weight"]),
                cal_schedule,
            )
            loss, _ = cal_objective(
                logits,
                targets,
                proxy_targets,
                mask,
                noisy_prior,
                proxy_prior,
                means,
                transition_means,
                confidence_weight=confidence_weight,
            )
            detached_all_losses = cal_all_class_losses(logits.detach())
            for proxy_class in range(classes):
                class_mask = mask & proxy_targets.eq(proxy_class)
                epoch_loss_sums[proxy_class] += detached_all_losses[class_mask].sum(dim=0)
                epoch_class_counts[proxy_class] += class_mask.sum()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += targets.numel(); loss_sum += float(loss.detach()) * targets.numel(); correct += int(logits.argmax(1).eq(targets).sum())
        observed_classes = epoch_class_counts > 0
        means[observed_classes] = epoch_loss_sums[observed_classes] / epoch_class_counts[observed_classes, None]
        means = means.detach()
        row_values = {"epoch": epoch + 1, "train_loss": loss_sum / total, "train_accuracy": correct / total, "learning_rate": optimizer.param_groups[0]["lr"], "method": "cal"}
        if validation_loader is not None:
            validation = evaluate_classification(model, validation_loader, criterion, device)
            row_values.update({"validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"]})
        row = standardize_epoch_row(
            row_values, require_validation=validation_loader is not None
        ); rows.append(row)
        if scheduler is not None:
            scheduler.step(resolve_confidence_weight(
                epoch + 1,
                float(cal_cfg["confidence_weight"]),
                cal_schedule,
            ))
        atomic_save({"method": "cal", "config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None if scheduler is None else scheduler.state_dict(), "completed_epoch": epoch, "metrics": rows, "proxy_hash": proxy.artifact_hash, "reference_loss_means": means.cpu(), "reference_transition_means": transition_means.cpu(), "rng_state": capture_rng_state()}, run_dir / "last.pt")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if rows: write_training_curves_svg(rows, run_dir / "training_curves.svg")
    test = evaluate_classification(model, test_loader, criterion, device)
    final_row = {"event": "final", "completed_epochs": epochs, "test_loss": test["loss"], "test_accuracy": test["accuracy"], "selection_split": "none", "test_selection_leakage": False, "method": "cal"}
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in [*rows, final_row]),
        encoding="utf-8",
    )
    return run_dir


__all__ = ["run_cal_experiment"]
