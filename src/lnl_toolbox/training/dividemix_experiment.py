from __future__ import annotations

"""CIFAR assembly, artifacts, and phase-aware resume for DivideMix."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.dividemix import (
    DivideMixAlgorithm,
    DivideMixConfig,
    DivideMixPhase,
    append_loss_history,
    build_co_divide,
    load_co_divide_artifact,
    save_co_divide_artifact,
)
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.losses.torch_losses import validate_per_sample_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state, read_checkpoint, restore_rng_state
from lnl_toolbox.training.data_service import PreparedData, prepare_experiment_data
from lnl_toolbox.training.experiment import _environment, _resolved_noise_config, build_model, build_optimizer, build_scheduler
from lnl_toolbox.training.noisy_labels import checkpoint_noise_metadata, effective_subset_actual_rate, noise_mode


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _build_peers(model_config, num_classes, seed, offset):
    with torch.random.fork_rng(): torch.manual_seed(seed); model_a = build_model(model_config, num_classes)
    with torch.random.fork_rng(): torch.manual_seed(seed + offset); model_b = build_model(model_config, num_classes)
    if tuple(model_a.state_dict()) != tuple(model_b.state_dict()) or not any(not torch.equal(a, b) for a, b in zip(model_a.state_dict().values(), model_b.state_dict().values()) if torch.is_floating_point(a)):
        raise RuntimeError("DivideMix peers must be architecture-identical and independently initialized")
    return model_a, model_b


@torch.inference_mode()
def _collect_losses(model, loader, device):
    model.eval(); losses, indices = [], []
    for batch in loader:
        inputs, targets = batch["input"].to(device), batch["target"].to(device)
        per_sample = torch.nn.functional.cross_entropy(model(inputs), targets, reduction="none")
        losses.append(validate_per_sample_loss(per_sample, int(targets.numel())).cpu()); indices.append(torch.as_tensor(batch["index"], dtype=torch.long))
    if not losses: raise RuntimeError("DivideMix loss snapshot dataset is empty")
    values, sample_indices = torch.cat(losses), torch.cat(indices)
    if torch.unique(sample_indices).numel() != sample_indices.numel(): raise ValueError("DivideMix loss snapshot indices must be unique")
    return values, sample_indices


@torch.inference_mode()
def _evaluate(algorithm, loader, device):
    algorithm.model_a.eval(); algorithm.model_b.eval(); totals = {"a": 0, "b": 0, "ensemble": 0}; samples = 0
    for batch in loader:
        inputs, targets = batch["input"].to(device), batch["target"].to(device)
        logits_a, logits_b = algorithm.model_a(inputs), algorithm.model_b(inputs)
        totals["a"] += int((logits_a.argmax(1) == targets).sum()); totals["b"] += int((logits_b.argmax(1) == targets).sum()); totals["ensemble"] += int(((logits_a + logits_b).argmax(1) == targets).sum()); samples += int(targets.numel())
    if not samples: raise RuntimeError("DivideMix evaluation split is empty")
    return {f"accuracy_{key}": value / samples for key, value in totals.items()} | {"samples": float(samples)}


def _checkpoint_payload(algorithm, config, noise_metadata, best_epoch, best_metric, best_metrics):
    return {"format_version": 2, "method": "dividemix", "checkpoint_role": "run_state", "config": deepcopy(config), "algorithm": algorithm.state_dict(), "rng_state": capture_rng_state(), "noise": dict(noise_metadata), "best_epoch": int(best_epoch), "best_selection_accuracy": float(best_metric), "component_states": {"dividemix_best": dict(best_metrics)}}


def _save_last(run_dir, algorithm, config, noise_metadata, best_epoch, best_metric, best_metrics):
    atomic_save(_checkpoint_payload(algorithm, config, noise_metadata, best_epoch, best_metric, best_metrics), run_dir / "last.pt")


def _validate_resume(current, saved):
    left, right = deepcopy(current["dividemix"]), deepcopy(saved["dividemix"])
    left["training"].pop("epochs", None); right["training"].pop("epochs", None)
    if left != right: raise ValueError("Resume configuration changed DivideMix identity")
    for key in ("seed", "data", "model", "optimizer", "scheduler", "loader", "evaluation"):
        if current.get(key) != saved.get(key): raise ValueError(f"Resume configuration changed {key}")
    noise_keys = {"name", "rate", "seed", "sampling", "rng", "validation_targets"}
    if {key: current["noise"].get(key) for key in noise_keys if key in current["noise"]} != {key: saved["noise"].get(key) for key in noise_keys if key in saved["noise"]}:
        raise ValueError("Resume configuration changed noise settings")


def _train_peer_epoch(algorithm, peer, artifact, prepared: PreparedData, noisy_by_index, epoch, seed):
    if peer == "a": mask, probabilities = artifact.labeled_for_a, artifact.clean_probability_b
    else: mask, probabilities = artifact.labeled_for_b, artifact.clean_probability_a
    indices = artifact.sample_indices.numpy(); labeled = indices[mask.numpy()]; unlabeled = indices[~mask.numpy()]
    probability_map = {int(index): float(value) for index, value in zip(indices, probabilities)}
    view_names = tuple(f"view_{index}" for index in range(algorithm.config.augmentations))
    labeled_set = prepared.dynamic_dataset(
        labeled, views=view_names, targets_by_index=noisy_by_index,
        overlays={"clean_probability": probability_map},
    )
    unlabeled_set = prepared.dynamic_dataset(
        unlabeled, views=view_names, targets_by_index=noisy_by_index,
    )
    peer_offset = 0 if peer == "a" else 100000
    labeled_loader = prepared.loader_for_dataset(labeled_set, generator_seed=seed + peer_offset + epoch * 2)
    unlabeled_loader = prepared.loader_for_dataset(unlabeled_set, generator_seed=seed + peer_offset + epoch * 2 + 1)
    unlabeled_iterator = iter(unlabeled_loader); totals: dict[str, float] = {}; count = 0
    rng = np.random.default_rng(seed + peer_offset + epoch)
    for batch_index, labeled_batch in enumerate(labeled_loader):
        try: unlabeled_batch = next(unlabeled_iterator)
        except StopIteration: unlabeled_iterator = iter(unlabeled_loader); unlabeled_batch = next(unlabeled_iterator)
        metrics = algorithm.train_peer_step(peer, tuple(value.to(algorithm.device) for value in labeled_batch["views"].values()), tuple(value.to(algorithm.device) for value in unlabeled_batch["views"].values()), labeled_batch["target"], labeled_batch["clean_probability"], epoch=epoch, batch_index=batch_index, num_batches=len(labeled_loader), rng=rng)
        for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / count for key, value in totals.items()} | {"batches": float(count), "labeled_count": float(len(labeled)), "unlabeled_count": float(len(unlabeled))}


def run_dividemix_experiment(config: dict[str, Any], output_dir: str | Path | None = None, resume: str | Path | None = None) -> Path:
    config = deepcopy(config); method = DivideMixConfig.from_mapping(config); seed = int(config.get("seed", 1)); seed_everything(seed)
    device = resolve_device(config.get("trainer", {}).get("device", "auto"))
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    run_dir = Path(resume).resolve().parent if resume else (Path(output_dir).resolve() if output_dir else Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")); run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = read_checkpoint(resume, "cpu") if resume else None
    if checkpoint:
        if checkpoint.get("method") != "dividemix": raise ValueError("checkpoint method is not DivideMix")
        _validate_resume(config, checkpoint["config"])
    data_config = config["data"]
    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.TRAIN_EVAL, DataRole.NOISY_VALIDATION, DataRole.TEST}),
            validation_targets="noisy",
        ),
        run_dir=run_dir, seed=seed, checkpoint_payload=checkpoint,
    )
    dataset_name, num_classes = prepared.dataset, prepared.num_classes
    manifest, manifest_path = prepared.manifest, prepared.manifest_path
    if manifest is None or manifest_path is None: raise ValueError("DivideMix requires a NoiseManifest")
    noisy_by_index = {int(index): int(target) for index, target in zip(manifest.global_indices, manifest.noisy_targets)}
    eval_train_loader = prepared.loader(DataRole.TRAIN_EVAL, shuffle=False)
    validation_loader = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False)
    noise_metadata = checkpoint_noise_metadata(manifest, manifest_path, run_dir, effective_subset_actual_rate(manifest, prepared.train_indices), mode=noise_mode(config), validation_targets="noisy", effective_validation_rate=effective_subset_actual_rate(manifest, prepared.validation_indices)); config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)
    model_a, model_b = _build_peers(config["model"], num_classes, seed, method.peer_seed_offset); optimizer_a, optimizer_b = build_optimizer(model_a, config["optimizer"]), build_optimizer(model_b, config["optimizer"])
    total_epochs = method.warmup_epochs + method.training_epochs; scheduler_a, scheduler_b = build_scheduler(optimizer_a, config.get("scheduler"), total_epochs), build_scheduler(optimizer_b, config.get("scheduler"), total_epochs)
    algorithm = DivideMixAlgorithm(model_a=model_a, model_b=model_b, optimizer_a=optimizer_a, optimizer_b=optimizer_b, scheduler_a=scheduler_a, scheduler_b=scheduler_b, config=method, device=device)
    best_epoch, best_metric, best_metrics = -1, float("-inf"), {"accuracy_a": float("-inf"), "accuracy_b": float("-inf"), "accuracy_ensemble": float("-inf")}
    if checkpoint:
        algorithm.load_state_dict(checkpoint["algorithm"]); restore_rng_state(checkpoint["rng_state"]); best_epoch = int(checkpoint["best_epoch"]); best_metric = float(checkpoint["best_selection_accuracy"]); best_metrics = dict(checkpoint["component_states"]["dividemix_best"])
        if algorithm.state.phase == DivideMixPhase.COMPLETED:
            if method.training_epochs <= algorithm.state.main_completed_epochs: return run_dir
            algorithm.state.phase = DivideMixPhase.CO_DIVIDE_FITTING
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8"); (run_dir / "environment.json").write_text(json.dumps(_environment(seed, device), indent=2), encoding="utf-8"); (run_dir / "noise_summary.json").write_text(json.dumps(noise_metadata, indent=2), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        while algorithm.state.warmup_completed_epochs < method.warmup_epochs:
            epoch = algorithm.state.warmup_completed_epochs; sums = {peer: {"objective": 0.0, "confidence_penalty": 0.0} for peer in ("a", "b")}; batches = 0
            warm_set = prepared.dynamic_dataset(prepared.train_indices, views=("view_0",), targets_by_index=noisy_by_index)
            for batch in prepared.loader_for_dataset(warm_set, generator_seed=seed + epoch):
                for peer in ("a", "b"):
                    result = algorithm.warmup_step(peer, batch["views"]["view_0"], batch["target"], asymmetric=str(config["noise"].get("name", "")).lower() == "asymmetric")
                    for key in sums[peer]: sums[peer][key] += result[key]
                batches += 1
            algorithm.step_schedulers(); algorithm.state.warmup_completed_epochs += 1
            if algorithm.state.warmup_completed_epochs == method.warmup_epochs: algorithm.state.transition(DivideMixPhase.CO_DIVIDE_FITTING)
            row = {"event": "warmup", "epoch": epoch + 1, "warmup_objective_a": sums["a"]["objective"] / batches, "warmup_objective_b": sums["b"]["objective"] / batches, "confidence_penalty_a": sums["a"]["confidence_penalty"] / batches, "confidence_penalty_b": sums["b"]["confidence_penalty"] / batches}
            metrics_file.write(json.dumps(row) + "\n"); metrics_file.flush(); _save_last(run_dir, algorithm, config, noise_metadata, best_epoch, best_metric, best_metrics)
        while algorithm.state.main_completed_epochs < method.training_epochs:
            epoch = algorithm.state.main_completed_epochs
            artifact_path = run_dir / f"dividemix_epoch_{epoch + 1:04d}.npz"
            if algorithm.state.phase in {DivideMixPhase.CO_DIVIDE_FITTING, DivideMixPhase.EPOCH_READY}:
                if algorithm.state.phase == DivideMixPhase.EPOCH_READY: algorithm.state.transition(DivideMixPhase.CO_DIVIDE_FITTING)
                losses_a, indices_a = _collect_losses(algorithm.model_a, eval_train_loader, device); losses_b, indices_b = _collect_losses(algorithm.model_b, eval_train_loader, device)
                if not torch.equal(indices_a, indices_b): raise ValueError("DivideMix peer loss snapshots are not stable-index aligned")
                append_loss_history(algorithm.state.loss_history_a, indices_a, losses_a, method.gmm.history_window_epochs); append_loss_history(algorithm.state.loss_history_b, indices_b, losses_b, method.gmm.history_window_epochs)
                co_divide = build_co_divide(indices_a, algorithm.state.loss_history_a, algorithm.state.loss_history_b, method, float(config["noise"]["rate"]))
                artifact_hash = save_co_divide_artifact(artifact_path, co_divide, epoch=epoch, metadata={"config_hash": method.identity_hash, "manifest_mapping_hash": noise_metadata["mapping_hash"], "model_hash_a": _state_hash(algorithm.model_a), "model_hash_b": _state_hash(algorithm.model_b), "threshold": method.gmm.threshold, **co_divide.metrics})
                algorithm.state.current_artifact, algorithm.state.current_artifact_hash = artifact_path.name, artifact_hash; algorithm.state.transition(DivideMixPhase.CO_DIVIDE_READY); _save_last(run_dir, algorithm, config, noise_metadata, best_epoch, best_metric, best_metrics)
            artifact, metadata, artifact_hash = load_co_divide_artifact(run_dir / str(algorithm.state.current_artifact))
            if artifact_hash != algorithm.state.current_artifact_hash or metadata["config_hash"] != method.identity_hash or metadata["manifest_mapping_hash"] != noise_metadata["mapping_hash"]: raise ValueError("DivideMix co-divide artifact provenance mismatch")
            if algorithm.state.phase in {DivideMixPhase.CO_DIVIDE_READY, DivideMixPhase.TRAIN_NETWORK_A}:
                if metadata["model_hash_a"] != _state_hash(algorithm.model_a) or metadata["model_hash_b"] != _state_hash(algorithm.model_b):
                    raise ValueError("DivideMix ready artifact does not match its epoch-start peer models")
            if algorithm.state.phase in {DivideMixPhase.NETWORK_A_READY, DivideMixPhase.TRAIN_NETWORK_B} and metadata["model_hash_b"] != _state_hash(algorithm.model_b):
                raise ValueError("DivideMix network-A-ready artifact does not match the unchanged peer B")
            metrics_a = {}
            if algorithm.state.phase in {DivideMixPhase.CO_DIVIDE_READY, DivideMixPhase.TRAIN_NETWORK_A}:
                if algorithm.state.phase == DivideMixPhase.CO_DIVIDE_READY: algorithm.state.transition(DivideMixPhase.TRAIN_NETWORK_A)
                metrics_a = _train_peer_epoch(algorithm, "a", artifact, prepared, noisy_by_index, epoch, seed); algorithm.state.transition(DivideMixPhase.NETWORK_A_READY); _save_last(run_dir, algorithm, config, noise_metadata, best_epoch, best_metric, best_metrics)
            if algorithm.state.phase == DivideMixPhase.NETWORK_A_READY: algorithm.state.transition(DivideMixPhase.TRAIN_NETWORK_B)
            metrics_b = _train_peer_epoch(algorithm, "b", artifact, prepared, noisy_by_index, epoch, seed); algorithm.state.transition(DivideMixPhase.EPOCH_READY); algorithm.step_schedulers(); algorithm.state.main_completed_epochs += 1
            validation = _evaluate(algorithm, validation_loader, device); improved = validation["accuracy_ensemble"] > best_metric
            if improved: best_epoch, best_metric, best_metrics = epoch, validation["accuracy_ensemble"], validation
            row = {"event": "epoch", "epoch": epoch + 1, "global_step_a": algorithm.state.optimizer_steps_a, "global_step_b": algorithm.state.optimizer_steps_b, "validation_accuracy_a": validation["accuracy_a"], "validation_accuracy_b": validation["accuracy_b"], "validation_accuracy_ensemble": validation["accuracy_ensemble"], "artifact_hash": artifact_hash, **{f"train_a_{k}": v for k, v in metrics_a.items()}, **{f"train_b_{k}": v for k, v in metrics_b.items()}}
            metrics_file.write(json.dumps(row) + "\n"); metrics_file.flush(); _save_last(run_dir, algorithm, config, noise_metadata, best_epoch, best_metric, best_metrics)
            if improved: atomic_save(_checkpoint_payload(algorithm, config, noise_metadata, best_epoch, best_metric, best_metrics) | {"checkpoint_role": "paired_best"}, run_dir / "best.pt")
        algorithm.state.transition(DivideMixPhase.COMPLETED); _save_last(run_dir, algorithm, config, noise_metadata, best_epoch, best_metric, best_metrics)
        best = read_checkpoint(run_dir / "best.pt", "cpu"); algorithm.model_a.load_state_dict(best["algorithm"]["model"]["a"]); algorithm.model_b.load_state_dict(best["algorithm"]["model"]["b"]); test = _evaluate(algorithm, test_loader, device)
        final = {"event": "final", "completed_warmup_epochs": algorithm.state.warmup_completed_epochs, "completed_epochs": algorithm.state.main_completed_epochs, "global_step_a": algorithm.state.optimizer_steps_a, "global_step_b": algorithm.state.optimizer_steps_b, "best_epoch": best_epoch + 1, "best_validation_accuracy_ensemble": best_metric, "test_accuracy_a": test["accuracy_a"], "test_accuracy_b": test["accuracy_b"], "test_accuracy_ensemble": test["accuracy_ensemble"], "ensemble": method.ensemble, "co_divide_artifact_hash": algorithm.state.current_artifact_hash, "noise": noise_metadata}
        if device.type == "cuda": final["max_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        metrics_file.write(json.dumps(final) + "\n")
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return run_dir


__all__ = ["run_dividemix_experiment"]
