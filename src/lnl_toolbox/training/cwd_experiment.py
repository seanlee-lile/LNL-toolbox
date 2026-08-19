from __future__ import annotations

"""Standalone CWD lifecycle for the CIFAR airplane/automobile benchmark."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.cwd import CWDGlobalObjective
from lnl_toolbox.data import DataRequirements, DataRole
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
from lnl_toolbox.training.data_service import prepare_experiment_data
from lnl_toolbox.training.snapshots import collect_feature_snapshot


def _build_model(config: Mapping[str, Any]):
    name = str(config.get("name", "resnet34")).strip().lower()
    if name in {"resnet34", "cifar_resnet34"}:
        outputs = int(config.get("num_outputs", 2))
        return cifar_resnet34(
            outputs,
            int(config.get("base_width", 64)),
            bias=bool(config.get("bias", True)),
        )
    if name == "tiny_cnn":
        return TinyCNN(2, int(config.get("width", 8)))
    raise ValueError(f"Unsupported CWD model: {name}")


def _classification_from_cwd_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("CWD logits must have shape [B, C]")
    if logits.shape[1] == 1:
        return (logits[:, 0] >= 0.0).long()
    return logits.argmax(1)


@torch.no_grad()
def _evaluate_cwd(model, loader, device, *, scalar_binary: bool) -> dict[str, float]:
    model.eval()
    loss_sum = correct = samples = 0.0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        output = forward_with_features(model, inputs)
        if scalar_binary:
            signed = targets.to(dtype=output.logits.dtype).mul(2.0).sub(1.0)
            margins = output.logits[:, 0]
            loss = (1.0 - signed * margins).square()
        else:
            loss = torch.nn.functional.cross_entropy(output.logits, targets, reduction="none")
        predictions = _classification_from_cwd_logits(output.logits)
        count = int(targets.numel())
        loss_sum += float(loss.sum())
        correct += float(predictions.eq(targets).sum())
        samples += count
    if samples == 0:
        raise ValueError("CWD evaluation loader must not be empty")
    return {"loss": loss_sum / samples, "accuracy": correct / samples}


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
        correct += float(_classification_from_cwd_logits(output.logits).eq(targets).sum())
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
    folds = int(data_config.get("folds", 5))
    fold_index = int(data_config.get("fold_index", 0))
    noise_config = config["noise"]
    rho_positive = float(noise_config["rho_positive"])
    rho_negative = float(noise_config["rho_negative"])
    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.TRAIN_EVAL, DataRole.TEST}),
            manifest_scope="effective_train",
        ),
        run_dir=run_dir, seed=seed, checkpoint_payload=checkpoint,
    )
    if prepared.num_classes != 2 or prepared.manifest is None or prepared.manifest_path is None:
        raise ValueError("CWD requires a noisy binary dataset view")
    manifest, manifest_path = prepared.manifest, prepared.manifest_path
    manifest.metadata.update({"folds": folds, "fold_index": fold_index})
    if checkpoint is not None and checkpoint.get("noise_mapping_hash") != manifest.mapping_hash:
        raise ValueError("CWD resume noise manifest mismatch")
    train_loader = prepared.loader(DataRole.TRAIN)
    snapshot_loader = prepared.loader(DataRole.TRAIN_EVAL, shuffle=False)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False)

    cwd_config = config.get("cwd", {})
    cwd_variant = str(cwd_config.get("variant", "multiclass")).strip().lower()
    model_config = dict(config["model"])
    if cwd_variant == "binary_scalar":
        model_config["num_outputs"] = 1
    model = _build_model(model_config).to(device)
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
    pinv_rcond = config.get("cwd", {}).get("pinv_rcond")
    if pinv_rcond is None and "ridge" in config.get("cwd", {}):
        # Backward-compatible parsing for old smoke configurations.  The
        # reproduction configuration uses the paper's unregularized pinv.
        pinv_rcond = float(config["cwd"]["ridge"])
    estimator = CWDEstimator(transition, pinv_rcond=pinv_rcond)
    criterion = CrossEntropyLoss().to(device)
    epochs = int(config["trainer"]["epochs"])
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(start_epoch, epochs):
        snapshot = collect_feature_snapshot(
            model,
            snapshot_loader,
            device,
            dataset=prepared.dataset,
            split=f"train_fold_{fold_index}",
            feature_extractor=lambda current, inputs: forward_with_features(current, inputs).features,
        )
        artifact = estimator.estimate(snapshot)
        snapshot.save(run_dir / "feature_snapshot.npz")
        artifact.save(run_dir / "statistic_artifact.npz")
        objective = CWDGlobalObjective(
            artifact,
            variant=str(config.get("cwd", {}).get("variant", "multiclass")),
            dynamic_centroid=bool(
                config.get("cwd", {}).get("dynamic_centroid", False)
            ),
        )
        train_loss, train_accuracy = _train_epoch(
            model, optimizer, train_loader, objective, device
        )
        test = _evaluate_cwd(
            model,
            test_loader,
            device,
            scalar_binary=cwd_variant == "binary_scalar",
        )
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
