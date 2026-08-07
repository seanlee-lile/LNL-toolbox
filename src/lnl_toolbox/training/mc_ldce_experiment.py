from __future__ import annotations

"""Dedicated, resumable MC-LDCE lifecycle."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml

from lnl_toolbox.algorithms.mc_ldce import MCLDCEObjective
from lnl_toolbox.algorithms.pcse.volmin import (
    PaperVolMinTransition,
    build_paper_volmin_optimizer,
    paper_volmin_objective,
    validate_paper_volmin_transition,
)
from lnl_toolbox.estimators.mc_ldce import MCLDCEEstimator
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.noise.statistics import StatisticArtifact
from lnl_toolbox.noise.transition import TransitionArtifact
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state, read_checkpoint, restore_rng_state
from lnl_toolbox.training.interfaces import RunContext
from lnl_toolbox.training.experiment import build_optimizer, build_scheduler
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.reproduction_data import build_reproduction_model, prepare_noisy_classification
from lnl_toolbox.training.snapshots import FeatureSnapshot, collect_feature_snapshot


def _directory(config, output_dir, resume) -> Path:
    path = Path(resume).resolve().parent if resume else Path(output_dir or Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _transition(config, snapshot_hash: str) -> TransitionArtifact:
    value = dict(config["transition"])
    if value.get("artifact"):
        return TransitionArtifact.load(value["artifact"])
    if "matrix" not in value:
        raise ValueError("MC-LDCE requires transition.artifact or transition.matrix")
    return TransitionArtifact(value["matrix"], str(value.get("estimator", "known")), snapshot_hash, {"configured": True})


_LIFECYCLE_VERSION = 2


def _lifecycle(config: dict[str, Any]) -> dict[str, Any]:
    value = dict(config.get("mc_ldce", {}))
    version = int(value.get("lifecycle_version", _LIFECYCLE_VERSION))
    feature_mode = str(value.get("feature_mode", "fixed")).lower()
    transition_model = str(value.get("transition_model", "separate")).lower()
    classifier_bias = bool(value.get("classifier_bias", False))
    if version != _LIFECYCLE_VERSION:
        raise ValueError(f"unsupported MC-LDCE lifecycle version: {version}")
    if feature_mode != "fixed":
        raise ValueError("MC-LDCE requires a fixed feature space")
    if transition_model != "separate":
        raise ValueError("MC-LDCE requires a separate transition estimator model")
    if classifier_bias:
        raise ValueError("MC-LDCE paper objective requires classifier_bias=false")
    return {
        "lifecycle_version": version,
        "feature_mode": feature_mode,
        "transition_model": transition_model,
        "classifier_bias": classifier_bias,
    }


def _prepare_fixed_feature_classifier(model: nn.Module) -> None:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise TypeError("MC-LDCE requires a direct linear classifier")
    if classifier.bias is not None:
        replacement = nn.Linear(
            classifier.in_features,
            classifier.out_features,
            bias=False,
            device=classifier.weight.device,
            dtype=classifier.weight.dtype,
        )
        model.classifier = replacement
        classifier = replacement
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)
    freeze = getattr(model, "freeze_feature_extractor", None)
    if callable(freeze):
        freeze()


def _estimate_volmin(config, model, loader, device):
    value = dict(config["transition"])
    parameterization = dict(value["parameterization"])
    transition_model = PaperVolMinTransition(
        int(value["num_classes"]),
        initial_weight=float(parameterization["initial_weight"]),
        seed=int(parameterization["seed"]),
    ).to(device)
    optimizer = build_paper_volmin_optimizer(model, transition_model, value["optimizer"])
    scheduler_config = dict(value.get("scheduler", {}) or {})
    scheduler = None
    if str(scheduler_config.get("name", "none")).lower() == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(item) for item in scheduler_config.get("milestones", [])],
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    elif str(scheduler_config.get("name", "none")).lower() != "none":
        raise ValueError("paper VolMin transition scheduler must be none or multistep")
    latest = {}
    for _ in range(int(value["epochs"])):
        model.train()
        for batch in loader:
            logits = model(batch["input"].to(device)).to(torch.float64)
            objective, latest = paper_volmin_objective(
                logits, batch["target"].to(device), transition_model.matrix(),
                lambda_volume=float(value["lambda_volume"]),
                determinant_tolerance=float(value.get("determinant_tolerance", 1e-8)),
                condition_limit=float(value.get("condition_limit", 1e8)),
            )
            optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
        if scheduler is not None:
            scheduler.step()
    matrix = transition_model.matrix()
    _, diagnostics = validate_paper_volmin_transition(
        matrix,
        determinant_tolerance=float(value.get("determinant_tolerance", 1e-8)),
        condition_limit=float(value.get("condition_limit", 1e8)),
    )
    return matrix.detach().cpu().numpy(), {
        "epochs": int(value["epochs"]),
        "optimizer": dict(value["optimizer"]),
        "scheduler": scheduler_config,
        "parameterization": parameterization,
        **latest,
        "determinant": diagnostics.determinant,
        "condition_number": diagnostics.condition_number,
    }


def run_mc_ldce_experiment(config: dict[str, Any], output_dir=None, resume=None, *, context: RunContext | None = None) -> Path:
    config = deepcopy(config)
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    device = resolve_device(str(config.get("trainer", {}).get("device", "auto")))
    run_dir = _directory(config, output_dir, resume)
    prepared = prepare_noisy_classification(config, run_dir, seed)
    lifecycle = _lifecycle(config)
    model = build_reproduction_model(config["model"], config["data"], prepared.num_classes).to(device)
    epochs = int(config["trainer"]["epochs"])
    criterion = CrossEntropyLoss().to(device)
    start_epoch = 0
    rows = []
    statistic = None
    if resume:
        payload = read_checkpoint(resume, device)
        if payload.get("method") != "mc_ldce" or payload.get("config") != config:
            raise ValueError("MC-LDCE resume identity mismatch")
        if payload.get("lifecycle_version") != _LIFECYCLE_VERSION:
            raise ValueError("MC-LDCE checkpoint predates fixed-feature lifecycle")
        _prepare_fixed_feature_classifier(model)
    elif str(config["transition"].get("estimator", "")).lower() == "paper_volmin":
        estimator_model = build_reproduction_model(
            dict(config["transition"].get("model", {"name": "resnet18"})),
            config["data"],
            prepared.num_classes,
        ).to(device)
        matrix, transition_metadata = _estimate_volmin(
            config, estimator_model, prepared.train_loader, device
        )
        del estimator_model
        _prepare_fixed_feature_classifier(model)
    else:
        warmup_epochs = int(config.get("warmup", {}).get("epochs", 0))
        warmup_optimizer = build_optimizer(model, config["optimizer"])
        for _ in range(warmup_epochs):
            model.train()
            for batch in prepared.train_loader:
                inputs, targets = batch["input"].to(device), batch["target"].to(device)
                loss = criterion(model(inputs), targets).mean()
                warmup_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                warmup_optimizer.step()
        _prepare_fixed_feature_classifier(model)
    optimizer = build_optimizer(model.classifier, config["optimizer"])
    scheduler_config = dict(config.get("scheduler", {}) or {})
    scheduler = (
        None if str(scheduler_config.get("name", "none")).lower() == "linear_after"
        else build_scheduler(optimizer, scheduler_config, epochs)
    )
    if resume:
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None: scheduler.load_state_dict(payload["scheduler"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["completed_epoch"]) + 1
        rows = list(payload.get("metrics", []))
        statistic = StatisticArtifact.load(run_dir / "statistic_artifact.npz")
        if statistic.artifact_hash != payload["statistic_hash"]:
            raise ValueError("MC-LDCE statistic resume mismatch")
    if statistic is None:
        snapshot = collect_feature_snapshot(model, prepared.snapshot_loader, device, dataset=prepared.dataset, split="train", feature_extractor=lambda current, inputs: forward_with_features(current, inputs).features)
        if str(config["transition"].get("estimator", "")).lower() == "paper_volmin":
            transition = TransitionArtifact(
                matrix,
                "paper_volmin",
                snapshot.snapshot_hash,
                transition_metadata,
            )
        else:
            transition = _transition(config, snapshot.snapshot_hash)
        statistic = MCLDCEEstimator(**dict(config.get("statistics", {}))).estimate(snapshot, transition)
        snapshot.save(run_dir / "feature_snapshot.npz"); transition.save(run_dir / "transition_artifact.npz"); statistic.save(run_dir / "statistic_artifact.npz")
    objective = MCLDCEObjective(statistic)
    base_learning_rate = float(config["optimizer"]["lr"])
    decay_start = int(config.get("scheduler", {}).get("decay_start", epochs))
    session = context.session if context is not None and context.state.get("lifecycle_active") else None
    if session is not None:
        session.start_phase("classifier_training", total_units=epochs)
    for epoch in range(start_epoch, epochs):
        if str(config.get("scheduler", {}).get("name", "none")).lower() == "linear_after":
            factor = 1.0 if epoch < decay_start else max(
                0.0, (epochs - epoch) / max(1, epochs - decay_start)
            )
            for group in optimizer.param_groups:
                group["lr"] = base_learning_rate * factor
        model.train(); total = correct = 0; loss_sum = 0.0
        for batch in prepared.train_loader:
            inputs, targets, indices = batch["input"].to(device), batch["target"].to(device), batch["index"].to(device)
            output = forward_with_features(model, inputs)
            loss = objective.compute(model=model, logits=output.logits, features=output.features, noisy_targets=targets, sample_indices=indices, base_loss=criterion, metadata={})
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            count = targets.numel(); total += count; loss_sum += float(loss.detach()) * count; correct += int(output.logits.argmax(1).eq(targets).sum())
        validation = evaluate_classification(model, prepared.validation_loader, criterion, device)
        test = evaluate_classification(model, prepared.test_loader, criterion, device)
        row = standardize_epoch_row({"epoch": epoch + 1, "train_loss": loss_sum / total, "train_accuracy": correct / total, "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"], "test_loss": test["loss"], "test_accuracy": test["accuracy"], "learning_rate": optimizer.param_groups[0]["lr"], "method": "mc_ldce"})
        rows.append(row)
        if session is not None:
            session.log_epoch(
                epoch + 1,
                phase="classifier_training",
                **{key: value for key, value in row.items()
                   if key not in {"event", "epoch", "phase", "seq"}},
            )
        if scheduler is not None: scheduler.step()
        atomic_save({"method": "mc_ldce", "lifecycle_version": lifecycle["lifecycle_version"], "config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None if scheduler is None else scheduler.state_dict(), "completed_epoch": epoch, "metrics": rows, "statistic_hash": statistic.artifact_hash, "rng_state": capture_rng_state()}, run_dir / "last.pt")
    if session is not None:
        session.end_phase("classifier_training", completed_units=max(0, epochs - start_epoch))
        final = {
            "method": "mc_ldce",
            "completed_epochs": epochs,
            "test_loss": rows[-1].get("test_loss") if rows else None,
            "test_accuracy": rows[-1].get("test_accuracy") if rows else None,
            "statistic_hash": statistic.artifact_hash,
        }
        session.emit("final", phase="evaluation", **final)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if session is None:
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if rows: write_training_curves_svg(rows, run_dir / "training_curves.svg")
    return run_dir


__all__ = ["run_mc_ldce_experiment"]
