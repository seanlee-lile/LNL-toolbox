from __future__ import annotations

"""CIFAR assembly and lifecycle for the complete Co-teaching method."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.algorithms.coteaching import CoTeachingAlgorithm, CoTeachingConfig
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.data import NoisyTargetDataset
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.torch_cifar import (
    TorchCifarDataset,
    build_cifar_transform,
    cifar_pixel_mean,
    train_validation_split,
)
from lnl_toolbox.losses.torch_losses import validate_per_sample_loss
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import (
    load_checkpoint,
    read_checkpoint,
    save_checkpoint,
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
from lnl_toolbox.training.interfaces import RunContext
from lnl_toolbox.training.reporting import RunSession


def _build_peer_models(
    model_config: Mapping[str, Any],
    num_classes: int,
    seed: int,
    peer_seed_offset: int,
):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model_a = build_model(model_config, num_classes)
    with torch.random.fork_rng():
        torch.manual_seed(seed + peer_seed_offset)
        model_b = build_model(model_config, num_classes)
    if tuple(model_a.state_dict()) != tuple(model_b.state_dict()):
        raise RuntimeError("Co-teaching peer architectures do not match")
    if not any(
        not torch.equal(left, right)
        for left, right in zip(model_a.state_dict().values(), model_b.state_dict().values())
        if torch.is_floating_point(left)
    ):
        raise RuntimeError("Co-teaching peers must have different initial parameters")
    return model_a, model_b


def _train_loader_for_epoch(dataset, config: Mapping[str, Any], seed: int) -> DataLoader:
    """Build an epoch-seeded loader so epoch-boundary resume is deterministic."""

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
def _evaluate_peers(
    algorithm: CoTeachingAlgorithm,
    loader,
    criterion,
    device: torch.device,
) -> dict[str, float]:
    algorithm.model_a.eval()
    algorithm.model_b.eval()
    loss_a = 0.0
    loss_b = 0.0
    correct_a = 0
    correct_b = 0
    correct_ensemble = 0
    samples = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        logits_a = algorithm.model_a(inputs)
        logits_b = algorithm.model_b(inputs)
        count = int(targets.numel())
        per_a = validate_per_sample_loss(criterion(logits_a, targets), count)
        per_b = validate_per_sample_loss(criterion(logits_b, targets), count)
        ensemble = (
            torch.softmax(logits_a, dim=1) + torch.softmax(logits_b, dim=1)
        ) / 2.0
        loss_a += float(per_a.sum().item())
        loss_b += float(per_b.sum().item())
        correct_a += int((logits_a.argmax(1) == targets).sum().item())
        correct_b += int((logits_b.argmax(1) == targets).sum().item())
        correct_ensemble += int((ensemble.argmax(1) == targets).sum().item())
        samples += count
    if samples == 0:
        raise RuntimeError("Co-teaching evaluation split is empty")
    accuracy_a = correct_a / samples
    accuracy_b = correct_b / samples
    return {
        "loss_a": loss_a / samples,
        "loss_b": loss_b / samples,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "mean_peer_accuracy": (accuracy_a + accuracy_b) / 2.0,
        "ensemble_accuracy": correct_ensemble / samples,
        "samples": float(samples),
    }


def _resume_noise_spec(noise: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "name",
        "rate",
        "seed",
        "sampling",
        "rng",
        "manifest",
        "manifest_sha256",
        "manifest_filename",
        "validation_targets",
    }
    return {key: noise[key] for key in keys if key in noise}


def _validate_resume_config(
    current: Mapping[str, Any], saved: Mapping[str, Any]
) -> None:
    if CoTeachingConfig.from_mapping(current) != CoTeachingConfig.from_mapping(saved):
        raise ValueError("Resume configuration changed Co-teaching settings")
    for key in ("seed", "data", "model", "optimizer", "scheduler", "loader"):
        if current.get(key) != saved.get(key):
            raise ValueError(f"Resume configuration changed {key}")
    current_trainer = dict(current.get("trainer", {}))
    saved_trainer = dict(saved.get("trainer", {}))
    current_trainer.pop("epochs", None)
    saved_trainer.pop("epochs", None)
    if current_trainer != saved_trainer:
        raise ValueError("Resume configuration changed trainer settings")
    if _resume_noise_spec(current["noise"]) != _resume_noise_spec(saved["noise"]):
        raise ValueError("Resume configuration changed noise settings")


def _best_component_state(metrics: Mapping[str, float]) -> dict[str, Any]:
    return {
        "accuracy_a": float(metrics["accuracy_a"]),
        "accuracy_b": float(metrics["accuracy_b"]),
        "mean_peer_accuracy": float(metrics["mean_peer_accuracy"]),
        "ensemble_accuracy": float(metrics["ensemble_accuracy"]),
    }


def run_coteaching_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *,
    context: RunContext | None = None,
) -> Path:
    """Run Co-teaching with exact peer cross-update and epoch-boundary resume."""

    config = deepcopy(config)
    method_config = CoTeachingConfig.from_mapping(config)
    if dict(config.get("loss", {"name": "ce"})) != {"name": "ce"}:
        raise ValueError("Co-teaching first version requires per-sample CE loss")
    seed = int(config.get("seed", 1))
    epochs = int(config["trainer"]["epochs"])
    if epochs <= 0:
        raise ValueError("trainer.epochs must be positive")
    seed_everything(seed)
    device = resolve_device(config["trainer"].get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if resume is not None:
        run_dir = Path(resume).resolve().parent
    elif output_dir is not None:
        run_dir = Path(output_dir).resolve()
    else:
        run_dir = Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    session = context.session if context is not None else RunSession(
        run_dir,
        config=config,
        runner="coteaching",
        method=str(config.get("method", "coteaching")),
        resumed=resume is not None,
    )
    lifecycle_active = context is None or bool(
        context.state.get("lifecycle_active", False)
    )
    if lifecycle_active:
        if not (context is not None and context.state.get("resume_lifecycle")):
            session.start_run()
            session.start_phase("train", total_units=epochs)

    checkpoint_payload = None
    if resume is not None:
        checkpoint_payload = read_checkpoint(resume, "cpu")
        saved_config = checkpoint_payload.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("Co-teaching checkpoint is missing resolved config")
        _validate_resume_config(config, saved_config)

    data_config = config["data"]
    dataset_name = str(data_config.get("name", "cifar10")).lower()
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError("Co-teaching first version supports CIFAR-10 and CIFAR-100")
    loader_fn = load_cifar10 if dataset_name == "cifar10" else load_cifar100
    num_classes = 10 if dataset_name == "cifar10" else 100
    train_data = loader_fn(data_config.get("root"), "train")
    test_data = loader_fn(data_config.get("root"), "test")
    validation_size = int(data_config["validation_size"])
    if validation_size <= 0:
        raise ValueError("Co-teaching requires a non-empty noisy validation split")
    split_config = data_config.get("validation_split", {}) or {}
    full_train_indices, validation_indices = train_validation_split(
        train_data.labels,
        validation_size,
        seed,
        strategy=str(split_config.get("strategy", "stratified")),
        rng=str(split_config.get("rng", "default_rng")),
    )
    if str(config["noise"].get("validation_targets", "")).lower() != "noisy":
        raise ValueError("Co-teaching checkpoint selection requires noisy validation targets")
    manifest_indices = np.sort(np.concatenate((full_train_indices, validation_indices)))
    manifest, manifest_path = prepare_noise_manifest(
        config,
        dataset=dataset_name,
        clean_targets=train_data.labels[manifest_indices],
        global_indices=manifest_indices,
        num_classes=num_classes,
        run_dir=run_dir,
        checkpoint_payload=checkpoint_payload,
        dataset_targets=train_data.labels,
    )
    if manifest is None or manifest_path is None:
        raise ValueError("Co-teaching requires a noisy-label manifest")

    train_indices = _subset(
        full_train_indices, train_data.labels, data_config.get("max_train_samples"), seed + 1
    )
    validation_indices = _subset(
        validation_indices,
        train_data.labels,
        data_config.get("max_validation_samples"),
        seed + 2,
    )
    test_indices = _subset(
        np.arange(len(test_data)),
        test_data.labels,
        data_config.get("max_test_samples"),
        seed + 3,
    )
    preprocessing = str(data_config.get("preprocessing", "standard")).lower()
    pixel_mean = cifar_pixel_mean(train_data.images) if preprocessing == "gce2018" else None
    transform_options = {"preprocessing": preprocessing, "pixel_mean": pixel_mean}
    clean_train_set = TorchCifarDataset(
        train_data,
        train_indices,
        transform=build_cifar_transform(
            True, bool(data_config.get("augment", True)), **transform_options
        ),
    )
    train_set = NoisyTargetDataset(
        clean_train_set, manifest.global_indices, manifest.noisy_targets
    )
    clean_validation_set = TorchCifarDataset(
        train_data,
        validation_indices,
        transform=build_cifar_transform(False, **transform_options),
    )
    validation_set = NoisyTargetDataset(
        clean_validation_set, manifest.global_indices, manifest.noisy_targets
    )
    test_set = TorchCifarDataset(
        test_data,
        test_indices,
        transform=build_cifar_transform(False, **transform_options),
    )
    validation_loader = _loader(validation_set, config["loader"], shuffle=False, seed=seed)
    test_loader = _loader(test_set, config["loader"], shuffle=False, seed=seed)

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
    config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)

    model_a, model_b = _build_peer_models(
        config["model"], num_classes, seed, method_config.peer_seed_offset
    )
    optimizer_a = build_optimizer(model_a, config["optimizer"])
    optimizer_b = build_optimizer(model_b, config["optimizer"])
    scheduler_a = build_scheduler(optimizer_a, config.get("scheduler"), epochs)
    scheduler_b = build_scheduler(optimizer_b, config.get("scheduler"), epochs)
    criterion = build_builtin_loss({"name": "ce"}).to(device)
    algorithm = CoTeachingAlgorithm(
        model_a=model_a,
        model_b=model_b,
        optimizer_a=optimizer_a,
        optimizer_b=optimizer_b,
        scheduler_a=scheduler_a,
        scheduler_b=scheduler_b,
        loss=criterion,
        device=device,
        method_config=method_config,
    )
    algorithm.setup(ExperimentContext(run_dir, config, seed))
    state = RunState(phase="train")
    completed_epoch = -1
    best_epoch = -1
    best_primary = float("-inf")
    best_metrics = {
        "accuracy_a": float("-inf"),
        "accuracy_b": float("-inf"),
        "mean_peer_accuracy": float("-inf"),
        "ensemble_accuracy": float("-inf"),
    }
    if resume is not None:
        state, completed_epoch, checkpoint_payload = load_checkpoint(
            resume, algorithm, device, scheduler=None
        )
        best_epoch = int(checkpoint_payload["best_epoch"])
        best_primary = float(checkpoint_payload["best_selection_accuracy"])
        component_states = checkpoint_payload.get("component_states", {})
        if not isinstance(component_states, Mapping) or "coteaching_best" not in component_states:
            raise ValueError("Co-teaching checkpoint is missing peer best metrics")
        best_metrics = dict(component_states["coteaching_best"])
        if completed_epoch + 1 >= epochs:
            return run_dir

    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(seed, device), indent=2), encoding="utf-8"
    )
    (run_dir / "noise_summary.json").write_text(
        json.dumps(noise_metadata, indent=2), encoding="utf-8"
    )

    corruption_by_index = {
        int(index): bool(corrupted)
        for index, corrupted in zip(manifest.global_indices, manifest.corruption_mask)
    }
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(completed_epoch + 1, epochs):
            state.cycle = epoch
            algorithm.on_cycle_start(state)
            sums: dict[str, float] = {}
            selected_total_a = 0.0
            selected_total_b = 0.0
            samples = 0.0
            clean_selected_a = 0
            clean_selected_b = 0
            train_loader = _train_loader_for_epoch(
                train_set, config["loader"], seed + epoch
            )
            for raw_batch in train_loader:
                result = algorithm.step(Batch(raw_batch), state)
                count = result.metrics["samples"]
                selected_a = result.metrics["selected_by_a_count"]
                selected_b = result.metrics["selected_by_b_count"]
                samples += count
                selected_total_a += selected_a
                selected_total_b += selected_b
                for key in (
                    "all_sample_loss_a",
                    "all_sample_loss_b",
                    "selection_overlap_rate",
                    "prediction_agreement_rate",
                    "accuracy_a",
                    "accuracy_b",
                    "accuracy_ensemble",
                ):
                    sums[key] = sums.get(key, 0.0) + result.metrics[key] * count
                sums["loss_a_on_selected_by_b"] = sums.get(
                    "loss_a_on_selected_by_b", 0.0
                ) + result.metrics["loss_a_on_selected_by_b"] * selected_b
                sums["loss_b_on_selected_by_a"] = sums.get(
                    "loss_b_on_selected_by_a", 0.0
                ) + result.metrics["loss_b_on_selected_by_a"] * selected_a
                clean_selected_a += sum(
                    not corruption_by_index[int(index)]
                    for index in result.metadata["selected_by_a_indices"].tolist()
                )
                clean_selected_b += sum(
                    not corruption_by_index[int(index)]
                    for index in result.metadata["selected_by_b_indices"].tolist()
                )
            algorithm.on_cycle_end(state)
            validation = _evaluate_peers(algorithm, validation_loader, criterion, device)
            learning_rate_a = float(optimizer_a.param_groups[0]["lr"])
            learning_rate_b = float(optimizer_b.param_groups[0]["lr"])
            algorithm.step_schedulers()
            row = {
                "event": "epoch",
                "epoch": epoch + 1,
                "global_step": state.step,
                "remember_rate": method_config.rate_at(epoch),
                "learning_rate_a": learning_rate_a,
                "learning_rate_b": learning_rate_b,
                "train_loss_a_on_selected_by_b": sums["loss_a_on_selected_by_b"] / selected_total_b,
                "train_loss_b_on_selected_by_a": sums["loss_b_on_selected_by_a"] / selected_total_a,
                "train_all_sample_loss_a": sums["all_sample_loss_a"] / samples,
                "train_all_sample_loss_b": sums["all_sample_loss_b"] / samples,
                "selected_by_a_count": selected_total_a,
                "selected_by_b_count": selected_total_b,
                "selected_clean_precision_a": clean_selected_a / selected_total_a,
                "selected_clean_precision_b": clean_selected_b / selected_total_b,
                "selection_overlap_rate": sums["selection_overlap_rate"] / samples,
                "prediction_agreement_rate": sums["prediction_agreement_rate"] / samples,
                "train_accuracy_a": sums["accuracy_a"] / samples,
                "train_accuracy_b": sums["accuracy_b"] / samples,
                "train_accuracy_ensemble": sums["accuracy_ensemble"] / samples,
                "validation_accuracy_a": validation["accuracy_a"],
                "validation_accuracy_b": validation["accuracy_b"],
                "mean_peer_accuracy": validation["mean_peer_accuracy"],
                "validation_accuracy_ensemble": validation["ensemble_accuracy"],
                "optimizer_steps_a": algorithm.private_state.optimizer_steps_a,
                "optimizer_steps_b": algorithm.private_state.optimizer_steps_b,
            }
            improved = validation["mean_peer_accuracy"] > best_primary
            if improved:
                best_primary = validation["mean_peer_accuracy"]
                best_epoch = epoch
                best_metrics = _best_component_state(validation)
            state.metrics = {
                key: float(value)
                for key, value in row.items()
                if isinstance(value, float)
            }
            checkpoint_kwargs = {
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_primary,
                "selection_split": "validation",
                "best_selection_accuracy": best_primary,
                "noise": noise_metadata,
                "component_states": {"coteaching_best": best_metrics},
            }
            save_checkpoint(
                run_dir / "last.pt",
                algorithm,
                state,
                epoch,
                config,
                **checkpoint_kwargs,
            )
            if improved:
                save_checkpoint(
                    run_dir / "best.pt",
                    algorithm,
                    state,
                    epoch,
                    config,
                    **checkpoint_kwargs,
                )
            event_metrics = dict(row)
            event_metrics.pop("event", None)
            event_metrics.pop("epoch", None)
            event_metrics.pop("global_step", None)
            if lifecycle_active:
                session.log_epoch(epoch + 1, phase="train", **event_metrics)
            print(json.dumps(row), flush=True)

        best_payload = read_checkpoint(run_dir / "best.pt", device)
        models = best_payload["model"]
        algorithm.model_a.load_state_dict(models["a"])
        algorithm.model_b.load_state_dict(models["b"])
        test = _evaluate_peers(algorithm, test_loader, criterion, device)
        final = {
            "event": "final",
            "completed_epochs": epochs,
            "global_step": state.step,
            "best_epoch": best_epoch + 1,
            "best_validation_accuracy_a": best_metrics["accuracy_a"],
            "best_validation_accuracy_b": best_metrics["accuracy_b"],
            "best_mean_peer_accuracy": best_metrics["mean_peer_accuracy"],
            "best_validation_accuracy_ensemble": best_metrics["ensemble_accuracy"],
            "test_accuracy_a": test["accuracy_a"],
            "test_accuracy_b": test["accuracy_b"],
            "test_mean_peer_accuracy": test["mean_peer_accuracy"],
            "test_accuracy_ensemble": test["ensemble_accuracy"],
            "optimizer_steps_a": algorithm.private_state.optimizer_steps_a,
            "optimizer_steps_b": algorithm.private_state.optimizer_steps_b,
            "noise": noise_metadata,
        }
        if device.type == "cuda":
            final["max_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        if lifecycle_active:
            phase_metrics = {
                key: value
                for key, value in row.items()
                if key not in {"event", "epoch", "global_step"}
            }
            session.emit(
                "phase_end",
                phase="train",
                completed_units=epochs,
                **phase_metrics,
            )
            final_metrics = dict(final)
            final_metrics.pop("event", None)
            session.emit("final", phase="evaluation", **final_metrics)
        print(json.dumps(final), flush=True)
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    algorithm.on_run_end(state)
    algorithm.close()
    return run_dir
