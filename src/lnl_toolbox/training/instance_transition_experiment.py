from __future__ import annotations

"""Reusable staged CIFAR runner for instance-transition estimators."""

from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.algorithms.instance_transition import InstanceTransitionClassificationAlgorithm
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.core.hyperparameters import resolve_parameter_sampling
from lnl_toolbox.data import NoisyTargetDataset
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.torch_cifar import TorchCifarDataset, build_cifar_transform, stratified_split
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.noise import NoiseManifest, PartTransitionArtifact, generate_pdl_idn
from lnl_toolbox.plugins.builtin import (
    build_builtin_instance_transition_algorithm,
    build_builtin_instance_transition_estimator,
    build_builtin_loss,
)
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from lnl_toolbox.training.experiment import build_model, build_optimizer, build_scheduler
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.snapshots import (
    collect_feature_snapshot,
    collect_posterior_snapshot,
    pretrain_noisy_classifier,
)


def _loader(dataset, config: Mapping[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)

    def seed_worker(_worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=shuffle,
        num_workers=workers, pin_memory=bool(config.get("pin_memory", True)),
        drop_last=bool(config.get("drop_last", False)) if shuffle else False,
        persistent_workers=workers > 0, worker_init_fn=seed_worker if workers else None,
        generator=generator)


def _run_directory(config: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    path = Path(output_dir) if output_dir is not None else Path(
        config.get("output_root", "artifacts/runs")
    ) / datetime.now().strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _transform(training: bool, config: Mapping[str, Any]):
    return build_cifar_transform(
        training, bool(config.get("augment", False)) if training else False,
        preprocessing=str(config.get("preprocessing", "standard")),
    )


def _write_environment(path: Path, seed: int, device: torch.device) -> None:
    path.write_text(json.dumps({
        "python_executable": __import__("sys").executable,
        "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "device": str(device), "seed": seed,
    }, indent=2), encoding="utf-8")


def _prepare_pdl_manifest(
    *, config: Mapping[str, Any], data: Any, indices: np.ndarray,
    num_classes: int, run_dir: Path, resume: bool,
) -> NoiseManifest:
    path = run_dir / str(config.get("manifest_filename", "noise_manifest.npz"))
    if resume:
        manifest = NoiseManifest.load(path)
        manifest.validate_for(data.labels, data.dataset, num_classes, required_indices=indices)
        return manifest
    if str(config.get("name", "")).lower() != "pdl":
        raise ValueError("instance-transition runner currently requires noise.name=pdl")
    manifest = generate_pdl_idn(
        data.images[indices].astype(np.float64) / 255.0,
        data.labels[indices], num_classes, float(config["rate"]),
        int(config["seed"]), data.dataset,
    )
    manifest.global_indices = indices.copy()
    manifest.split = "train"
    manifest.num_classes = num_classes
    manifest.save(path)
    return manifest


def run_instance_transition_experiment(
    raw_config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Run warm-up → snapshots → instance estimator → corrected training."""

    config, parameter_record = resolve_parameter_sampling(raw_config)
    config = dict(config)
    seed = int(config.get("seed", 1))
    epochs = int(config["trainer"]["epochs"])
    if epochs <= 0:
        raise ValueError("trainer.epochs must be positive")
    seed_everything(seed)
    device = resolve_device(str(config["trainer"].get("device", "auto")))
    run_dir = Path(resume).resolve().parent if resume else _run_directory(config, output_dir)
    checkpoint_payload = None if resume is None else read_checkpoint(resume, device)
    if checkpoint_payload is not None and dict(checkpoint_payload.get("config") or {}) != config:
        raise ValueError("Resume configuration changed")

    data_config = dict(config["data"])
    dataset_name = str(data_config["name"]).lower()
    if dataset_name == "cifar10":
        train_data, test_data, num_classes = (
            load_cifar10(data_config["root"], "train"),
            load_cifar10(data_config["root"], "test"), 10,
        )
    elif dataset_name == "cifar100":
        train_data, test_data, num_classes = (
            load_cifar100(data_config["root"], "train"),
            load_cifar100(data_config["root"], "test"), 100,
        )
    else:
        raise ValueError("instance-transition runner supports CIFAR-10/100")
    all_indices = np.arange(len(train_data), dtype=np.int64)
    maximum = data_config.get("max_train_samples")
    if maximum is not None and int(maximum) < all_indices.size:
        _, selected = stratified_split(train_data.labels, int(maximum), seed + 1)
        all_indices = selected
    manifest = _prepare_pdl_manifest(config=config["noise"], data=train_data,
        indices=all_indices, num_classes=num_classes, run_dir=run_dir, resume=resume is not None)

    noisy_validation_size = int(config["warmup"]["noisy_validation_size"])
    train_positions, validation_positions = stratified_split(
        manifest.noisy_targets, noisy_validation_size, seed + 2
    )
    train_indices = manifest.global_indices[train_positions]
    validation_indices = manifest.global_indices[validation_positions]
    train_targets = manifest.noisy_targets[train_positions]
    validation_targets = manifest.noisy_targets[validation_positions]
    train_set = NoisyTargetDataset(TorchCifarDataset(train_data, train_indices,
        transform=_transform(True, data_config)), train_indices, train_targets)
    noisy_validation_set = NoisyTargetDataset(TorchCifarDataset(train_data, validation_indices,
        transform=_transform(False, data_config)), validation_indices, validation_targets)
    union_set = NoisyTargetDataset(TorchCifarDataset(train_data, manifest.global_indices,
        transform=_transform(False, data_config)), manifest.global_indices, manifest.noisy_targets)
    max_test = data_config.get("max_test_samples")
    test_indices = np.arange(len(test_data), dtype=np.int64)
    if max_test is not None and int(max_test) < test_indices.size:
        _, test_indices = stratified_split(test_data.labels, int(max_test), seed + 3)
    test_set = TorchCifarDataset(test_data, test_indices, transform=_transform(False, data_config))
    loader_config = dict(config["loader"])
    train_loader = _loader(train_set, loader_config, shuffle=True, seed=seed)
    noisy_validation_loader = _loader(noisy_validation_set, loader_config, shuffle=False, seed=seed)
    union_loader = _loader(union_set, loader_config, shuffle=False, seed=seed)
    test_loader = _loader(test_set, loader_config, shuffle=False, seed=seed)

    artifact_path = run_dir / "instance_transition_artifact.npz"
    if resume is not None:
        artifact = PartTransitionArtifact.load(artifact_path)
    else:
        warmup_model = build_model(config["model"], num_classes)
        warmup_optimizer = build_optimizer(warmup_model, config["warmup"]["optimizer"])
        warmup_scheduler = build_scheduler(
            warmup_optimizer, config["warmup"].get("scheduler"),
            int(config["warmup"]["epochs"]),
        )
        warmup_loss = build_builtin_loss(config["warmup"].get("loss", {"name": "ce"}))
        warmup_epochs = int(config["warmup"]["epochs"])
        best_state = None
        best_accuracy = float("-inf")
        for _ in range(warmup_epochs):
            pretrain_noisy_classifier(warmup_model, warmup_optimizer, train_loader,
                device, epochs=1, criterion=warmup_loss)
            score = evaluate_classification(warmup_model, noisy_validation_loader, warmup_loss, device)
            if score["accuracy"] > best_accuracy:
                best_accuracy = score["accuracy"]
                best_state = {key: value.detach().cpu().clone() for key, value in warmup_model.state_dict().items()}
            if warmup_scheduler is not None:
                warmup_scheduler.step()
        if best_state is not None:
            warmup_model.load_state_dict(best_state)
        torch.save({"model": warmup_model.state_dict(), "best_noisy_validation_accuracy": best_accuracy},
            run_dir / "warmup_model.pt")
        posterior = collect_posterior_snapshot(warmup_model, union_loader, device,
            dataset=dataset_name, split="train")
        features = collect_feature_snapshot(warmup_model, union_loader, device,
            dataset=dataset_name, split="train",
            feature_extractor=lambda model, inputs: forward_with_features(model, inputs).features)
        posterior.save(run_dir / "posterior_snapshot.npz")
        features.save(run_dir / "feature_snapshot.npz")
        estimator = build_builtin_instance_transition_estimator(config["instance_transition"])
        artifact = estimator.estimate(features, posterior)
        artifact.save(artifact_path)

    model = build_model(config["model"], num_classes)
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(optimizer, config.get("scheduler"), epochs)
    criterion = build_builtin_loss(config.get("loss", {"name": "ce"})).to(device)
    algorithm: InstanceTransitionClassificationAlgorithm = build_builtin_instance_transition_algorithm(
        config["algorithm"], model=model, optimizer=optimizer, loss=criterion,
        transition=artifact, device=device,
    )
    algorithm.setup(ExperimentContext(run_dir, config, seed))
    state = RunState(phase="corrected_train")
    completed_epoch, best_epoch, best_accuracy = -1, -1, float("-inf")
    if resume is not None:
        state, completed_epoch, checkpoint_payload = load_checkpoint(
            resume, algorithm, device, scheduler=scheduler
        )
        best_epoch = int(checkpoint_payload["best_epoch"])
        best_accuracy = float(checkpoint_payload["best_selection_accuracy"])

    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if parameter_record is not None:
        (run_dir / "parameter_record.json").write_text(
            json.dumps(parameter_record.to_dict(), indent=2), encoding="utf-8")
    _write_environment(run_dir / "environment.json", seed, device)
    (run_dir / "pipeline_state.json").write_text(json.dumps({
        "phase": "corrected_train", "artifact_hash": artifact.artifact_hash,
        "feature_snapshot_hash": artifact.feature_snapshot_hash,
        "posterior_snapshot_hash": artifact.posterior_snapshot_hash,
        "train_indices": train_indices.tolist(), "noisy_validation_indices": validation_indices.tolist(),
    }, indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.jsonl"
    rows = []
    if metrics_path.is_file():
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("event") == "epoch"]
    algorithm.on_run_start(state)
    with metrics_path.open("a", encoding="utf-8") as handle:
        for epoch in range(completed_epoch + 1, epochs):
            state.cycle = epoch
            algorithm.on_cycle_start(state)
            loss_sum = accuracy_sum = samples = 0.0
            for raw_batch in train_loader:
                result = algorithm.step(Batch(raw_batch), state)
                count = result.metrics["samples"]
                samples += count
                loss_sum += result.metrics["loss"] * count
                accuracy_sum += result.metrics["accuracy"] * count
            algorithm.on_cycle_end(state)
            selection = evaluate_classification(model, noisy_validation_loader, criterion, device)
            row = standardize_epoch_row({"event": "epoch", "epoch": epoch + 1,
                "global_step": state.step, "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": loss_sum / samples, "train_accuracy": accuracy_sum / samples,
                "selection_split": "noisy_validation", "selection_loss": selection["loss"],
                "selection_accuracy": selection["accuracy"]})
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if scheduler is not None:
                scheduler.step()
            if selection["accuracy"] > best_accuracy:
                best_accuracy, best_epoch = selection["accuracy"], epoch + 1
                save_checkpoint(run_dir / "best.pt", algorithm, state, epoch, config,
                    scheduler=scheduler, best_epoch=best_epoch,
                    best_selection_accuracy=best_accuracy, selection_split="noisy_validation",
                    pipeline={"phase": "corrected_train", "artifact_hash": artifact.artifact_hash})
            save_checkpoint(run_dir / "last.pt", algorithm, state, epoch, config,
                scheduler=scheduler, best_epoch=best_epoch,
                best_selection_accuracy=best_accuracy, selection_split="noisy_validation",
                pipeline={"phase": "corrected_train", "artifact_hash": artifact.artifact_hash})
            if bool(config["trainer"].get("progress", {}).get("curves", True)):
                write_training_curves_svg(rows, run_dir / "training_curves.svg")
        algorithm.on_run_end(state)
        test = evaluate_classification(model, test_loader, criterion, device)
        final = {"event": "final", "completed_epochs": epochs, "global_step": state.step,
            "best_epoch": best_epoch, "best_selection_accuracy": best_accuracy,
            "test_loss": test["loss"], "test_accuracy": test["accuracy"],
            "artifact_hash": artifact.artifact_hash}
        handle.write(json.dumps(final) + "\n")
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return run_dir


__all__ = ["run_instance_transition_experiment"]
