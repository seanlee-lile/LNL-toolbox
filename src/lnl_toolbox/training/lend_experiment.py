from __future__ import annotations

"""CIFAR assembly and strict epoch-boundary lifecycle for LEND."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from lnl_toolbox.algorithms.lend import LENDAlgorithm, LENDConfig
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.losses.torch_losses import validate_per_sample_loss
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from lnl_toolbox.training.experiment import (
    _environment, _resolved_noise_config,
    build_model, build_optimizer, build_scheduler,
)
from lnl_toolbox.training.data_service import prepare_experiment_data
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata, effective_subset_actual_rate, noise_mode,
)


@torch.inference_mode()
def _evaluate(model, loader, criterion, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    for raw in loader:
        inputs = raw["input"].to(device, non_blocking=True)
        targets = raw["target"].to(device, non_blocking=True)
        logits = model(inputs)
        count = int(targets.numel())
        per_sample = validate_per_sample_loss(criterion(logits, targets), count)
        loss_sum += float(per_sample.sum().item())
        correct += int((logits.argmax(1) == targets).sum().item())
        samples += count
    if samples == 0:
        raise RuntimeError("LEND evaluation split is empty")
    return {"loss": loss_sum / samples, "accuracy": correct / samples,
            "samples": float(samples)}


def _resume_noise_spec(noise: Mapping[str, Any]) -> dict[str, Any]:
    keys = {"name", "rate", "seed", "sampling", "rng", "manifest",
            "manifest_sha256", "manifest_filename", "validation_targets"}
    return {key: noise[key] for key in keys if key in noise}


def _validate_resume_config(current: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    if LENDConfig.from_mapping(current).identity() != LENDConfig.from_mapping(saved).identity():
        raise ValueError("Resume configuration changed LEND settings")
    for key in ("seed", "data", "model", "loss", "optimizer", "scheduler", "loader", "evaluation"):
        if current.get(key) != saved.get(key):
            raise ValueError(f"Resume configuration changed {key}")
    current_trainer = dict(current.get("trainer", {}))
    saved_trainer = dict(saved.get("trainer", {}))
    if current_trainer != saved_trainer:
        raise ValueError("Resume configuration changed trainer settings")
    if _resume_noise_spec(current["noise"]) != _resume_noise_spec(saved["noise"]):
        raise ValueError("Resume configuration changed noise settings")
    if LENDConfig.from_mapping(current).epochs < LENDConfig.from_mapping(saved).epochs:
        raise ValueError("LEND resume cannot reduce the epoch target")


def run_lend_experiment(config: dict[str, Any], output_dir: str | Path | None = None,
                        resume: str | Path | None = None) -> Path:
    """Run the paper-oriented, online LEND workflow."""

    config = deepcopy(config)
    method_config = LENDConfig.from_mapping(config)
    seed = int(config.get("seed", 1))
    epochs = method_config.epochs
    seed_everything(seed)
    device = resolve_device(config.get("trainer", {}).get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if resume is not None:
        run_dir = Path(resume).resolve().parent
    elif output_dir is not None:
        run_dir = Path(output_dir).resolve()
    else:
        run_dir = Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = None
    if resume is not None:
        checkpoint_payload = read_checkpoint(resume, "cpu")
        saved_config = checkpoint_payload.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("LEND checkpoint is missing resolved config")
        _validate_resume_config(config, saved_config)

    data_config = config["data"]
    validation_size = int(data_config.get("validation_size", 0))
    if validation_size <= 0:
        raise ValueError("LEND requires a non-empty noisy validation split")
    if str(config["noise"].get("validation_targets", "")).lower() != "noisy":
        raise ValueError("LEND best-checkpoint selection requires noisy validation targets")
    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
            validation_targets="noisy",
        ),
        run_dir=run_dir, seed=seed, checkpoint_payload=checkpoint_payload,
    )
    dataset_name, num_classes = prepared.dataset, prepared.num_classes
    manifest, manifest_path = prepared.manifest, prepared.manifest_path
    if manifest is None or manifest_path is None:
        raise ValueError("LEND requires a noisy-label manifest")
    batch_size = int(config["loader"]["batch_size"])
    drop_last = bool(config["loader"].get("drop_last", False))
    remainder = len(prepared.train_indices) % batch_size
    if len(prepared.train_indices) <= method_config.k or (not drop_last and 0 < remainder <= method_config.k):
        raise ValueError("LEND final partial training batch must satisfy B > k")
    validation_loader = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False)
    effective_rate = effective_subset_actual_rate(manifest, prepared.train_indices)
    effective_validation_rate = effective_subset_actual_rate(manifest, prepared.validation_indices)
    noise_metadata = checkpoint_noise_metadata(
        manifest, manifest_path, run_dir, effective_rate, mode=noise_mode(config),
        validation_targets="noisy", effective_validation_rate=effective_validation_rate,
    )
    config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)

    model = build_model(config["model"], num_classes)
    if not callable(getattr(model, "forward_with_features", None)):
        raise ValueError("LEND model must support forward_with_features()")
    # Runtime shape proof before any training or checkpoint mutation.
    probe_loader = prepared.loader(DataRole.TRAIN, shuffle=False,
                                   batch_size=min(batch_size, len(prepared.train_indices)))
    probe = next(iter(probe_loader))
    model.to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        probe_output = forward_with_features(model, probe["input"].to(device))
    model.train(was_training)
    if probe_output.features.ndim != 2 or probe_output.features.shape[0] != probe["input"].shape[0]:
        raise ValueError("LEND model embedding must have shape [B,D]")
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(optimizer, config.get("scheduler"), epochs)
    criterion = build_builtin_loss({"name": "ce"}).to(device)
    algorithm = LENDAlgorithm(
        model=model, optimizer=optimizer, loss=criterion, device=device,
        method_config=method_config,
        canonical_global_indices=torch.as_tensor(prepared.train_indices, dtype=torch.int64),
        num_classes=num_classes,
    )
    algorithm.setup(ExperimentContext(run_dir, config, seed))
    state = RunState(phase="training")
    completed_epoch = best_epoch = -1
    best_accuracy = float("-inf")
    if resume is not None:
        state, completed_epoch, checkpoint_payload = load_checkpoint(
            resume, algorithm, device, scheduler=scheduler
        )
        history = algorithm.private_state.history
        if bool(history.initialized.any()) and int(history.last_updated_epoch.max()) > completed_epoch:
            raise ValueError("LEND history contains updates beyond the completed epoch")
        best_epoch = int(checkpoint_payload["best_epoch"])
        best_accuracy = float(checkpoint_payload["best_validation_accuracy"])
        if completed_epoch + 1 >= epochs:
            return run_dir

    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(_environment(seed, device), indent=2), encoding="utf-8")
    (run_dir / "noise_summary.json").write_text(json.dumps(noise_metadata, indent=2), encoding="utf-8")
    corruption = {int(index): bool(value) for index, value in zip(manifest.global_indices, manifest.corruption_mask)}
    clean_target = {int(index): int(value) for index, value in zip(manifest.global_indices, manifest.clean_targets)}
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(completed_epoch + 1, epochs):
            state.cycle = epoch
            algorithm.on_cycle_start(state)
            totals: dict[str, float] = {}
            samples = selected_total = oracle_clean_selected = 0.0
            oracle_clean_observed = oracle_diluted_correct = 0.0
            for raw_batch in prepared.loader(DataRole.TRAIN, epoch=epoch):
                result = algorithm.step(Batch(raw_batch), state)
                count = result.metrics["samples"]
                samples += count
                selected = result.metrics["selected_samples"]
                selected_total += selected
                oracle_clean_selected += sum(
                    not corruption[int(index)] for index in result.metadata["selected_indices"].tolist()
                )
                batch_indices = result.metadata["batch_indices"].tolist()
                oracle_clean_observed += sum(not corruption[int(index)] for index in batch_indices)
                history_predictions = result.metadata["history_values"].argmax(dim=1).tolist()
                oracle_diluted_correct += sum(
                    prediction == clean_target[int(index)]
                    for prediction, index in zip(history_predictions, batch_indices)
                )
                for key, value in result.metrics.items():
                    if key in {"samples", "selected_samples", "optimizer_steps", "empty_selection_batches"}:
                        continue
                    if key == "selected_train_loss_sum":
                        totals[key] = totals.get(key, 0.0) + value
                    else:
                        totals[key] = totals.get(key, 0.0) + value * count
            algorithm.on_cycle_end(state)
            validation = _evaluate(model, validation_loader, criterion, device)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step()
            improved = validation["accuracy"] > best_accuracy
            if improved:
                best_accuracy, best_epoch = validation["accuracy"], epoch
            row = {
                "event": "epoch", "epoch": epoch + 1,
                "global_batch_step": state.step,
                "optimizer_steps": algorithm.private_state.optimizer_steps,
                "learning_rate": learning_rate,
                "selected_samples": selected_total,
                "selected_ratio": selected_total / samples,
                "empty_selection_batches": algorithm.private_state.empty_selection_batches,
                "oracle_selection_precision": (
                    oracle_clean_selected / selected_total if selected_total else 0.0
                ),
                "oracle_selection_recall": (
                    oracle_clean_selected / oracle_clean_observed
                    if oracle_clean_observed else 0.0
                ),
                "oracle_diluted_label_accuracy": oracle_diluted_correct / samples,
                "validation_accuracy": validation["accuracy"],
                "best_validation_accuracy": best_accuracy,
            }
            for key, value in totals.items():
                row["train_" + key] = (
                    value if key == "selected_train_loss_sum" else value / samples
                )
            kwargs = {
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "selection_split": "validation",
                "best_selection_accuracy": best_accuracy,
                "noise": noise_metadata,
            }
            save_checkpoint(run_dir / "last.pt", algorithm, state, epoch, config,
                            scheduler=scheduler, **kwargs)
            if improved:
                save_checkpoint(run_dir / "best.pt", algorithm, state, epoch, config,
                                scheduler=scheduler, **kwargs)
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            print(json.dumps(row), flush=True)
        final_model_state = deepcopy(model.state_dict())
        best_payload = read_checkpoint(run_dir / "best.pt", device)
        model.load_state_dict(best_payload["model"])
        test = _evaluate(model, test_loader, criterion, device)
        model.load_state_dict(final_model_state)
        algorithm.on_run_end(state)
        final = {
            "event": "final", "completed_epochs": epochs,
            "global_batch_step": state.step,
            "optimizer_steps": algorithm.private_state.optimizer_steps,
            "best_epoch": best_epoch + 1,
            "best_validation_accuracy": best_accuracy,
            "clean_test_accuracy": test["accuracy"],
            "fidelity": "paper_oriented",
            "noise": noise_metadata,
        }
        if device.type == "cuda":
            final["max_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        metrics_file.write(json.dumps(final) + "\n")
        print(json.dumps(final), flush=True)
    save_checkpoint(
        run_dir / "last.pt", algorithm, state, epochs - 1, config,
        scheduler=scheduler, best_epoch=best_epoch,
        best_validation_accuracy=best_accuracy, selection_split="validation",
        best_selection_accuracy=best_accuracy, noise=noise_metadata,
        component_states={"lend_completion": {"completed": True}},
    )
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    algorithm.close()
    return run_dir


__all__ = ["run_lend_experiment"]
