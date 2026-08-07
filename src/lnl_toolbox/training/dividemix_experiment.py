from __future__ import annotations

"""CIFAR assembly, artifacts, and phase-aware resume for DivideMix."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
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
from lnl_toolbox.data import NoisyTargetDataset
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.torch_cifar import TorchCifarDataset, build_cifar_transform, train_validation_split
from lnl_toolbox.losses.torch_losses import validate_per_sample_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state, read_checkpoint, restore_rng_state
from lnl_toolbox.training.experiment import _environment, _loader, _resolved_noise_config, _seed_worker, _subset, build_model, build_optimizer, build_scheduler
from lnl_toolbox.training.noisy_labels import checkpoint_noise_metadata, effective_subset_actual_rate, noise_mode, prepare_noise_manifest


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class _MultiViewDataset(Dataset[dict[str, Any]]):
    def __init__(self, data, indices, targets_by_index, transform, views: int, probabilities=None):
        self.data = data; self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or self.indices.size == 0 or np.unique(self.indices).size != self.indices.size:
            raise ValueError("DivideMix view dataset indices must be non-empty and unique")
        self.targets = {int(k): int(v) for k, v in targets_by_index.items()}
        self.probabilities = None if probabilities is None else {int(k): float(v) for k, v in probabilities.items()}
        self.transform, self.views = transform, int(views)

    def __len__(self): return int(self.indices.size)

    def __getitem__(self, item):
        index = int(self.indices[item]); image = Image.fromarray(self.data.images[index], mode="RGB")
        result = {"views": tuple(self.transform(image) for _ in range(self.views)), "target": self.targets[index], "index": index}
        if self.probabilities is not None: result["clean_probability"] = self.probabilities[index]
        return result


def _epoch_loader(dataset, loader_config, seed, *, shuffle=True):
    workers = int(loader_config.get("num_workers", 0))
    return DataLoader(dataset, batch_size=int(loader_config["batch_size"]), shuffle=shuffle, num_workers=workers, pin_memory=bool(loader_config.get("pin_memory", True)), persistent_workers=False, worker_init_fn=_seed_worker if workers else None, generator=torch.Generator().manual_seed(seed))


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


def _train_peer_epoch(algorithm, peer, artifact, train_data, noisy_by_index, transform, loader_config, epoch, seed):
    if peer == "a": mask, probabilities = artifact.labeled_for_a, artifact.clean_probability_b
    else: mask, probabilities = artifact.labeled_for_b, artifact.clean_probability_a
    indices = artifact.sample_indices.numpy(); labeled = indices[mask.numpy()]; unlabeled = indices[~mask.numpy()]
    probability_map = {int(index): float(value) for index, value in zip(indices, probabilities)}
    labeled_set = _MultiViewDataset(train_data, labeled, noisy_by_index, transform, algorithm.config.augmentations, probability_map)
    unlabeled_set = _MultiViewDataset(train_data, unlabeled, noisy_by_index, transform, algorithm.config.augmentations)
    peer_offset = 0 if peer == "a" else 100000
    labeled_loader = _epoch_loader(labeled_set, loader_config, seed + peer_offset + epoch * 2)
    unlabeled_loader = _epoch_loader(unlabeled_set, loader_config, seed + peer_offset + epoch * 2 + 1)
    unlabeled_iterator = iter(unlabeled_loader); totals: dict[str, float] = {}; count = 0
    rng = np.random.default_rng(seed + peer_offset + epoch)
    for batch_index, labeled_batch in enumerate(labeled_loader):
        try: unlabeled_batch = next(unlabeled_iterator)
        except StopIteration: unlabeled_iterator = iter(unlabeled_loader); unlabeled_batch = next(unlabeled_iterator)
        metrics = algorithm.train_peer_step(peer, tuple(value.to(algorithm.device) for value in labeled_batch["views"]), tuple(value.to(algorithm.device) for value in unlabeled_batch["views"]), labeled_batch["target"], labeled_batch["clean_probability"], epoch=epoch, batch_index=batch_index, num_batches=len(labeled_loader), rng=rng)
        for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / count for key, value in totals.items()} | {"batches": float(count), "labeled_count": float(len(labeled)), "unlabeled_count": float(len(unlabeled))}


def run_dividemix_experiment(config: dict[str, Any], output_dir: str | Path | None = None, resume: str | Path | None = None, *, context=None) -> Path:
    config = deepcopy(config); method = DivideMixConfig.from_mapping(config); seed = int(config.get("seed", 1)); seed_everything(seed)
    device = resolve_device(config.get("trainer", {}).get("device", "auto"))
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    run_dir = Path(resume).resolve().parent if resume else (Path(output_dir).resolve() if output_dir else Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")); run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = read_checkpoint(resume, "cpu") if resume else None
    if checkpoint:
        if checkpoint.get("method") != "dividemix": raise ValueError("checkpoint method is not DivideMix")
        _validate_resume(config, checkpoint["config"])
    data_config = config["data"]; dataset_name = str(data_config.get("name", "cifar10")).lower()
    if dataset_name not in {"cifar10", "cifar100"}: raise ValueError("DivideMix supports CIFAR-10 and CIFAR-100")
    load = load_cifar10 if dataset_name == "cifar10" else load_cifar100; num_classes = 10 if dataset_name == "cifar10" else 100
    train_data, test_data = load(data_config.get("root"), "train"), load(data_config.get("root"), "test")
    train_indices, validation_indices = train_validation_split(train_data.labels, int(data_config["validation_size"]), seed)
    manifest_indices = np.sort(np.concatenate((train_indices, validation_indices)))
    manifest, manifest_path = prepare_noise_manifest(config, dataset=dataset_name, clean_targets=train_data.labels[manifest_indices], global_indices=manifest_indices, num_classes=num_classes, run_dir=run_dir, checkpoint_payload=checkpoint, dataset_targets=train_data.labels)
    if manifest is None or manifest_path is None: raise ValueError("DivideMix requires a NoiseManifest")
    train_indices = _subset(train_indices, train_data.labels, data_config.get("max_train_samples"), seed + 1); validation_indices = _subset(validation_indices, train_data.labels, data_config.get("max_validation_samples"), seed + 2); test_indices = _subset(np.arange(len(test_data)), test_data.labels, data_config.get("max_test_samples"), seed + 3)
    noisy_by_index = {int(index): int(target) for index, target in zip(manifest.global_indices, manifest.noisy_targets)}
    train_transform = build_cifar_transform(True, bool(data_config.get("augment", True)), normalization_mean=data_config.get("normalization_mean"), normalization_std=data_config.get("normalization_std"))
    eval_transform = build_cifar_transform(False, normalization_mean=data_config.get("normalization_mean"), normalization_std=data_config.get("normalization_std"))
    eval_train_set = NoisyTargetDataset(TorchCifarDataset(train_data, train_indices, transform=eval_transform), manifest.global_indices, manifest.noisy_targets)
    validation_set = NoisyTargetDataset(TorchCifarDataset(train_data, validation_indices, transform=eval_transform), manifest.global_indices, manifest.noisy_targets)
    test_set = TorchCifarDataset(test_data, test_indices, transform=eval_transform)
    eval_train_loader = _epoch_loader(eval_train_set, config["loader"], seed, shuffle=False); validation_loader = _loader(validation_set, config["loader"], shuffle=False, seed=seed); test_loader = _loader(test_set, config["loader"], shuffle=False, seed=seed)
    noise_metadata = checkpoint_noise_metadata(manifest, manifest_path, run_dir, effective_subset_actual_rate(manifest, train_indices), mode=noise_mode(config), validation_targets="noisy", effective_validation_rate=effective_subset_actual_rate(manifest, validation_indices)); config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)
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
            warm_set = _MultiViewDataset(train_data, train_indices, noisy_by_index, train_transform, 1)
            for batch in _epoch_loader(warm_set, config["loader"], seed + epoch):
                for peer in ("a", "b"):
                    result = algorithm.warmup_step(peer, batch["views"][0], batch["target"], asymmetric=str(config["noise"].get("name", "")).lower() == "asymmetric")
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
                metrics_a = _train_peer_epoch(algorithm, "a", artifact, train_data, noisy_by_index, train_transform, config["loader"], epoch, seed); algorithm.state.transition(DivideMixPhase.NETWORK_A_READY); _save_last(run_dir, algorithm, config, noise_metadata, best_epoch, best_metric, best_metrics)
            if algorithm.state.phase == DivideMixPhase.NETWORK_A_READY: algorithm.state.transition(DivideMixPhase.TRAIN_NETWORK_B)
            metrics_b = _train_peer_epoch(algorithm, "b", artifact, train_data, noisy_by_index, train_transform, config["loader"], epoch, seed); algorithm.state.transition(DivideMixPhase.EPOCH_READY); algorithm.step_schedulers(); algorithm.state.main_completed_epochs += 1
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
