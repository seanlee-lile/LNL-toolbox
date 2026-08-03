from __future__ import annotations

"""Unified CIFAR runner for the standalone VolMinNet method."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.algorithms.volminnet import (
    VolMinNetAlgorithm,
    VolMinNetConfig,
    VolMinTransition,
)
from lnl_toolbox.algorithms.volminnet.artifacts import (
    VolMinTransitionArtifact,
    persist_transition_atomically,
)
from lnl_toolbox.data import NoisyTargetDataset
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.torch_cifar import (
    TorchCifarDataset,
    build_cifar_transform,
    train_validation_split,
)
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.experiment import (
    _environment,
    _loader,
    _resolved_noise_config,
    _seed_worker,
    _subset,
    build_model,
    build_optimizer,
    build_scheduler,
)
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    effective_subset_actual_rate,
    noise_mode,
    prepare_noise_manifest,
)


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _resume_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    noise = dict(config["noise"])
    allowed_noise = {
        "name", "rate", "seed", "sampling", "rng", "manifest",
        "manifest_sha256", "manifest_filename", "validation_targets",
    }
    trainer = dict(config["trainer"])
    trainer.pop("epochs", None)
    return {
        "method": config.get("method"),
        "execution": config.get("execution"),
        "seed": config.get("seed"),
        "data": config.get("data"),
        "noise": {key: noise[key] for key in sorted(allowed_noise & set(noise))},
        "volminnet": config.get("volminnet"),
        "loader": config.get("loader"),
        "trainer": trainer,
    }


def _cpu_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item.detach().cpu().clone() if torch.is_tensor(item) else deepcopy(item)
        for key, item in value.items()
    }


def _epoch_loader(dataset: Any, config: Mapping[str, Any], seed: int) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", True)),
        persistent_workers=False,
        worker_init_fn=_seed_worker if workers else None,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def _evaluate_noisy(
    model: torch.nn.Module,
    transition: VolMinTransition,
    loader: Any,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    transition.eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        logits = model(inputs)
        matrix = transition.matrix(dtype=logits.dtype)
        noisy_log_probability = torch.logsumexp(
            F.log_softmax(logits, dim=1)[:, :, None] + torch.log(matrix)[None, :, :],
            dim=1,
        )
        loss_sum += float(F.nll_loss(noisy_log_probability, targets, reduction="sum").item())
        correct += int((noisy_log_probability.argmax(1) == targets).sum().item())
        samples += int(targets.numel())
    if samples == 0:
        raise ValueError("VolMinNet noisy-validation loader is empty")
    return {"loss": loss_sum / samples, "accuracy": correct / samples, "samples": float(samples)}


@torch.inference_mode()
def _evaluate_clean(model: torch.nn.Module, loader: Any, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        logits = model(inputs)
        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        correct += int((logits.argmax(1) == targets).sum().item())
        samples += int(targets.numel())
    if samples == 0:
        raise ValueError("VolMinNet clean-test loader is empty")
    return {"loss": loss_sum / samples, "accuracy": correct / samples, "samples": float(samples)}


def _artifact(
    algorithm: VolMinNetAlgorithm,
    *,
    role: str,
    epoch: int,
    noise_metadata: Mapping[str, Any],
    true_transition: np.ndarray | None,
) -> VolMinTransitionArtifact:
    matrix = algorithm.transition.matrix().detach().cpu().numpy()
    diagnostics = algorithm.transition.diagnostics()
    metadata: dict[str, Any] = {
        "method": "volminnet",
        "role": role,
        "convention": algorithm.transition.convention,
        "parameterization": algorithm.transition.parameterization,
        "normalization_axis": algorithm.transition.normalization_axis,
        "initialization": algorithm.transition.initialization,
        "initial_raw_value": algorithm.transition.initial_raw_value,
        "epoch": int(epoch),
        "global_step": algorithm.state.global_step,
        "manifest_sha256": noise_metadata["manifest_sha256"],
        "mapping_hash": noise_metadata["mapping_hash"],
        **diagnostics,
    }
    if true_transition is not None:
        difference = matrix - np.asarray(true_transition, dtype=np.float64)
        metadata.update({
            "diagnostic_only_true_transition": True,
            "transition_frobenius_error": float(np.linalg.norm(difference)),
            "transition_mean_absolute_error": float(np.abs(difference).mean()),
            "transition_max_absolute_error": float(np.abs(difference).max()),
        })
    return VolMinTransitionArtifact(
        algorithm.transition.off_diagonal_logits.detach().cpu().numpy(), matrix, metadata
    )


def _validate_resume_config(current: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    if VolMinNetConfig.from_mapping(current) != VolMinNetConfig.from_mapping(saved):
        raise ValueError("Resume configuration changed VolMinNet method settings")
    current_identity = _resume_identity(current)
    saved_identity = _resume_identity(saved)
    if current_identity != saved_identity:
        raise ValueError("Resume configuration changed VolMinNet run identity")


def _checkpoint_payload(
    algorithm: VolMinNetAlgorithm,
    *,
    role: str,
    config: Mapping[str, Any],
    noise_metadata: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    best_pair: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": "volminnet",
        "checkpoint_role": role,
        "config": dict(config),
        "config_identity_hash": _stable_hash(_resume_identity(config)),
        "algorithm": algorithm.state_dict(),
        "best_pair": dict(best_pair),
        "noise": dict(noise_metadata),
        "artifact_hashes": dict(artifact_hashes),
        "rng_state": capture_rng_state(),
    }


def _validate_artifacts(run_dir: Path, hashes: Mapping[str, Any]) -> None:
    for name in ("initial", "best", "last"):
        expected = hashes.get(name)
        if expected is None and name == "best":
            continue
        path = run_dir / f"transition_{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"VolMinNet transition_{name}.npz is missing")
        artifact = VolMinTransitionArtifact.load(path)
        if artifact.artifact_hash != expected:
            raise ValueError(f"VolMinNet transition_{name} artifact hash mismatch")


def run_volminnet_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    config = deepcopy(config)
    method_config = VolMinNetConfig.from_mapping(config)
    seed = int(config.get("seed", 1))
    epochs = int(config["trainer"]["epochs"])
    seed_everything(seed)
    device = resolve_device(str(config["trainer"].get("device", "auto")))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_dir = (
        Path(resume).resolve().parent
        if resume is not None
        else Path(output_dir).resolve()
        if output_dir is not None
        else (Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_payload = None
    if resume is not None:
        resume_payload = read_checkpoint(resume, "cpu")
        if resume_payload.get("method") != "volminnet" or resume_payload.get("checkpoint_role") != "run_state":
            raise ValueError("Only a VolMinNet last.pt checkpoint may be resumed")
        saved_config = resume_payload.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("VolMinNet checkpoint resolved config is missing")
        _validate_resume_config(config, saved_config)
        if resume_payload.get("config_identity_hash") != _stable_hash(_resume_identity(config)):
            raise ValueError("VolMinNet checkpoint config identity hash mismatch")

    data_config = config["data"]
    dataset_name = str(data_config["name"]).lower()
    loader_fn = load_cifar10 if dataset_name == "cifar10" else load_cifar100
    num_classes = 10 if dataset_name == "cifar10" else 100
    train_data = loader_fn(data_config.get("root"), "train")
    test_data = loader_fn(data_config.get("root"), "test")
    train_indices, validation_indices = train_validation_split(
        train_data.labels,
        int(data_config["validation_size"]),
        seed,
        strategy=str(data_config.get("validation_split", {}).get("strategy", "stratified")),
        rng=str(data_config.get("validation_split", {}).get("rng", "default_rng")),
    )
    manifest_indices = np.sort(np.concatenate((train_indices, validation_indices)))
    manifest, manifest_path = prepare_noise_manifest(
        config,
        dataset=dataset_name,
        clean_targets=train_data.labels[manifest_indices],
        global_indices=manifest_indices,
        num_classes=num_classes,
        run_dir=run_dir,
        checkpoint_payload=resume_payload,
        dataset_targets=train_data.labels,
    )
    if manifest is None or manifest_path is None:
        raise ValueError("VolMinNet requires noisy train and validation labels")
    train_indices = _subset(
        train_indices, train_data.labels, data_config.get("max_train_samples"), seed + 1
    )
    validation_indices = _subset(
        validation_indices, train_data.labels, data_config.get("max_validation_samples"), seed + 2
    )
    test_indices = _subset(
        np.arange(len(test_data)), test_data.labels, data_config.get("max_test_samples"), seed + 3
    )
    transform_train = build_cifar_transform(True, bool(data_config.get("augment", True)))
    transform_eval = build_cifar_transform(False)
    train_set = NoisyTargetDataset(
        TorchCifarDataset(train_data, train_indices, transform=transform_train),
        manifest.global_indices,
        manifest.noisy_targets,
    )
    validation_set = NoisyTargetDataset(
        TorchCifarDataset(train_data, validation_indices, transform=transform_eval),
        manifest.global_indices,
        manifest.noisy_targets,
    )
    test_set = TorchCifarDataset(test_data, test_indices, transform=transform_eval)
    validation_loader = _loader(validation_set, config["loader"], shuffle=False, seed=seed + 20)
    test_loader = _loader(test_set, config["loader"], shuffle=False, seed=seed + 21)
    effective_rate = effective_subset_actual_rate(manifest, train_indices)
    effective_validation_rate = effective_subset_actual_rate(manifest, validation_indices)
    noise_metadata = checkpoint_noise_metadata(
        manifest,
        manifest_path,
        run_dir,
        effective_rate,
        mode=noise_mode(config),
        validation_targets="noisy",
        effective_validation_rate=effective_validation_rate,
    )
    if resume_payload is not None and dict(resume_payload["noise"]) != noise_metadata:
        raise ValueError("VolMinNet resume noise provenance changed")
    config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)

    model_config = config["volminnet"]["model"]
    model = build_model(model_config, num_classes)
    transition = VolMinTransition(num_classes)
    classifier_optimizer = build_optimizer(model, method_config.classifier_optimizer)
    transition_optimizer = build_optimizer(transition, method_config.transition_optimizer)
    classifier_scheduler = build_scheduler(
        classifier_optimizer, method_config.classifier_scheduler, epochs
    )
    transition_scheduler = build_scheduler(
        transition_optimizer, method_config.transition_scheduler, epochs
    )
    algorithm = VolMinNetAlgorithm(
        model=model,
        transition=transition,
        classifier_optimizer=classifier_optimizer,
        transition_optimizer=transition_optimizer,
        classifier_scheduler=classifier_scheduler,
        transition_scheduler=transition_scheduler,
        lambda_volume=method_config.lambda_volume,
        device=device,
    )
    artifact_hashes: dict[str, str] = {}
    best_pair: dict[str, Any] = {}
    if resume_payload is not None:
        algorithm.load_state_dict(resume_payload["algorithm"])
        restore_rng_state(resume_payload["rng_state"])
        artifact_hashes = dict(resume_payload.get("artifact_hashes", {}))
        best_pair = dict(resume_payload.get("best_pair", {}))
        _validate_artifacts(run_dir, artifact_hashes)
        if algorithm.state.completed and algorithm.state.completed_epochs >= epochs:
            return run_dir
    else:
        initial = persist_transition_atomically(
            _artifact(
                algorithm,
                role="initial",
                epoch=0,
                noise_metadata=noise_metadata,
                true_transition=manifest.transition_matrix,
            ),
            run_dir / "transition_initial.npz",
        )
        artifact_hashes["initial"] = initial.artifact_hash

    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(seed, device), indent=2), encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(algorithm.state.completed_epochs, epochs):
        sums = {"objective": 0.0, "classification_loss": 0.0, "volume_logdet": 0.0}
        samples = 0.0
        train_loader = _epoch_loader(train_set, config["loader"], seed + 1000 + epoch)
        for batch in train_loader:
            metrics = algorithm.train_batch(batch)
            count = metrics["samples"]
            samples += count
            for name in sums:
                sums[name] += metrics[name] * count
        if samples == 0:
            raise ValueError("VolMinNet training loader is empty")
        validation = _evaluate_noisy(model, transition, validation_loader, device)
        algorithm.state.completed_epochs = epoch + 1
        improved = validation["loss"] < algorithm.state.best_validation_loss
        if improved:
            algorithm.state.best_epoch = epoch + 1
            algorithm.state.best_validation_loss = validation["loss"]
            algorithm.state.best_validation_accuracy = validation["accuracy"]
            best_pair = {
                "model": _cpu_state(model.state_dict()),
                "transition": _cpu_state(transition.state_dict()),
                "epoch": epoch + 1,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
            }
        current_last = persist_transition_atomically(
            _artifact(
                algorithm,
                role="last",
                epoch=epoch + 1,
                noise_metadata=noise_metadata,
                true_transition=manifest.transition_matrix,
            ),
            run_dir / "transition_last.npz",
        )
        artifact_hashes["last"] = current_last.artifact_hash
        if improved:
            current_best = persist_transition_atomically(
                _artifact(
                    algorithm,
                    role="best",
                    epoch=epoch + 1,
                    noise_metadata=noise_metadata,
                    true_transition=manifest.transition_matrix,
                ),
                run_dir / "transition_best.npz",
            )
            artifact_hashes["best"] = current_best.artifact_hash
        row = {
            "event": "epoch",
            "epoch": epoch + 1,
            "global_step": algorithm.state.global_step,
            "train_objective": sums["objective"] / samples,
            "train_noisy_nll": sums["classification_loss"] / samples,
            "volume_logdet": sums["volume_logdet"] / samples,
            "noisy_validation_loss": validation["loss"],
            "noisy_validation_accuracy": validation["accuracy"],
            "learning_rate_classifier": float(classifier_optimizer.param_groups[0]["lr"]),
            "learning_rate_transition": float(transition_optimizer.param_groups[0]["lr"]),
            "best_epoch": algorithm.state.best_epoch,
            "best_noisy_validation_loss": algorithm.state.best_validation_loss,
            **transition.diagnostics(),
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        algorithm.step_schedulers()
        payload = _checkpoint_payload(
            algorithm,
            role="run_state",
            config=config,
            noise_metadata=noise_metadata,
            artifact_hashes=artifact_hashes,
            best_pair=best_pair,
        )
        atomic_save(payload, run_dir / "last.pt")
        if improved:
            best_payload = dict(payload)
            best_payload["checkpoint_role"] = "best_evaluation"
            atomic_save(best_payload, run_dir / "best.pt")
        print(json.dumps(row), flush=True)

    if not best_pair:
        raise ValueError("VolMinNet best classifier/transition pair is missing")
    current_model = _cpu_state(model.state_dict())
    current_transition = _cpu_state(transition.state_dict())
    model.load_state_dict(best_pair["model"])
    transition.load_state_dict(best_pair["transition"])
    test = _evaluate_clean(model, test_loader, device)
    best_artifact = VolMinTransitionArtifact.load(run_dir / "transition_best.npz")
    model.load_state_dict(current_model)
    transition.load_state_dict(current_transition)
    algorithm.state.completed = True
    final: dict[str, Any] = {
        "event": "final",
        "method": "volminnet",
        "completed_epochs": algorithm.state.completed_epochs,
        "global_step": algorithm.state.global_step,
        "best_epoch": algorithm.state.best_epoch,
        "best_noisy_validation_loss": algorithm.state.best_validation_loss,
        "best_noisy_validation_accuracy": algorithm.state.best_validation_accuracy,
        "best_checkpoint_clean_test_loss": test["loss"],
        "best_checkpoint_clean_test_accuracy": test["accuracy"],
        "learned_transition": best_artifact.matrix.tolist(),
        "transition_best_hash": best_artifact.artifact_hash,
        "transition_diagnostics": dict(best_artifact.metadata),
        "classifier_optimizer_steps": algorithm.state.classifier_optimizer_steps,
        "transition_optimizer_steps": algorithm.state.transition_optimizer_steps,
        "noise": noise_metadata,
    }
    if device.type == "cuda":
        final["max_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(final, sort_keys=True) + "\n")
    atomic_save(
        _checkpoint_payload(
            algorithm,
            role="run_state",
            config=config,
            noise_metadata=noise_metadata,
            artifact_hashes=artifact_hashes,
            best_pair=best_pair,
        ),
        run_dir / "last.pt",
    )
    print(json.dumps(final), flush=True)
    return run_dir
