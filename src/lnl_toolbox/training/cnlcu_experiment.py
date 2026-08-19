from __future__ import annotations

"""CIFAR assembly and epoch-boundary lifecycle for CNLCU-S/H."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from lnl_toolbox.algorithms.cnlcu import CNLCUAlgorithm, CNLCUConfig
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from lnl_toolbox.training.coteaching_experiment import (
    _best_component_state, _build_peer_models, _evaluate_peers,
    _resume_noise_spec,
)
from lnl_toolbox.training.experiment import (
    _environment, _resolved_noise_config,
    build_optimizer, build_scheduler,
)
from lnl_toolbox.training.data_service import prepare_experiment_data
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata, effective_subset_actual_rate, noise_mode,
)


def _validate_resume_config(current: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    if CNLCUConfig.from_mapping(current) != CNLCUConfig.from_mapping(saved):
        raise ValueError("Resume configuration changed CNLCU settings")
    for key in ("seed", "data", "model", "optimizer", "scheduler", "loader"):
        if current.get(key) != saved.get(key):
            raise ValueError(f"Resume configuration changed {key}")
    current_trainer, saved_trainer = dict(current.get("trainer", {})), dict(saved.get("trainer", {}))
    current_trainer.pop("epochs", None); saved_trainer.pop("epochs", None)
    if current_trainer != saved_trainer:
        raise ValueError("Resume configuration changed trainer settings")
    if _resume_noise_spec(current["noise"]) != _resume_noise_spec(saved["noise"]):
        raise ValueError("Resume configuration changed noise settings")


def run_cnlcu_experiment(config: dict[str, Any], output_dir: str | Path | None = None,
                         resume: str | Path | None = None) -> Path:
    """Run a configured CNLCU variant with strict epoch-boundary resume."""

    config = deepcopy(config)
    method_config = CNLCUConfig.from_mapping(config)
    if dict(config.get("loss", {"name": "ce"})) != {"name": "ce"}:
        raise ValueError("CNLCU requires per-sample CE loss")
    seed, epochs = int(config.get("seed", 1)), int(config["trainer"]["epochs"])
    if epochs <= 0:
        raise ValueError("trainer.epochs must be positive")
    seed_everything(seed)
    device = resolve_device(config["trainer"].get("device", "auto"))
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
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
            raise ValueError("CNLCU checkpoint is missing resolved config")
        _validate_resume_config(config, saved_config)

    data_config = config["data"]
    validation_size = int(data_config["validation_size"])
    if validation_size <= 0:
        raise ValueError("CNLCU requires a non-empty noisy validation split")
    if str(config["noise"].get("validation_targets", "")).lower() != "noisy":
        raise ValueError("CNLCU checkpoint selection requires noisy validation targets")
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
        raise ValueError("CNLCU requires a noisy-label manifest")
    validation_loader = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False)
    effective_rate = effective_subset_actual_rate(manifest, prepared.train_indices)
    effective_validation_rate = effective_subset_actual_rate(manifest, prepared.validation_indices)
    noise_metadata = checkpoint_noise_metadata(
        manifest, manifest_path, run_dir, effective_rate, mode=noise_mode(config),
        validation_targets="noisy", effective_validation_rate=effective_validation_rate,
    )
    config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)

    model_a, model_b = _build_peer_models(config["model"], num_classes, seed,
                                           method_config.peer_seed_offset)
    optimizer_a, optimizer_b = build_optimizer(model_a, config["optimizer"]), build_optimizer(model_b, config["optimizer"])
    scheduler_a, scheduler_b = build_scheduler(optimizer_a, config.get("scheduler"), epochs), build_scheduler(optimizer_b, config.get("scheduler"), epochs)
    criterion = build_builtin_loss({"name": "ce"}).to(device)
    algorithm = CNLCUAlgorithm(
        model_a=model_a, model_b=model_b, optimizer_a=optimizer_a, optimizer_b=optimizer_b,
        scheduler_a=scheduler_a, scheduler_b=scheduler_b, loss=criterion, device=device,
        method_config=method_config,
        canonical_global_indices=torch.as_tensor(prepared.train_indices, dtype=torch.int64),
    )
    algorithm.setup(ExperimentContext(run_dir, config, seed))
    state, completed_epoch, best_epoch, best_primary = RunState(phase="train"), -1, -1, float("-inf")
    best_metrics = {"accuracy_a": float("-inf"), "accuracy_b": float("-inf"),
                    "mean_peer_accuracy": float("-inf"), "ensemble_accuracy": float("-inf")}
    if resume is not None:
        state, completed_epoch, checkpoint_payload = load_checkpoint(resume, algorithm, device, scheduler=None)
        best_epoch = int(checkpoint_payload["best_epoch"])
        best_primary = float(checkpoint_payload["best_selection_accuracy"])
        components = checkpoint_payload.get("component_states", {})
        if not isinstance(components, Mapping) or "cnlcu_best" not in components:
            raise ValueError("CNLCU checkpoint is missing peer best metrics")
        best_metrics = dict(components["cnlcu_best"])
        if completed_epoch + 1 >= epochs:
            return run_dir
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(_environment(seed, device), indent=2), encoding="utf-8")
    (run_dir / "noise_summary.json").write_text(json.dumps(noise_metadata, indent=2), encoding="utf-8")
    corruption = {int(i): bool(c) for i, c in zip(manifest.global_indices, manifest.corruption_mask)}
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(completed_epoch + 1, epochs):
            state.cycle = epoch; algorithm.on_cycle_start(state)
            sums: dict[str, float] = {}; samples = selected_a_total = selected_b_total = 0.0
            extrema: dict[str, float] = {}
            clean_a = clean_b = 0
            for raw_batch in prepared.loader(DataRole.TRAIN, epoch=epoch):
                result = algorithm.step(Batch(raw_batch), state)
                count = result.metrics["samples"]; samples += count
                selected_a_total += result.metrics["selected_by_a_count"]
                selected_b_total += result.metrics["selected_by_b_count"]
                aggregate_keys = ["current_loss_a", "current_loss_b", "robust_mean_a", "robust_mean_b",
                            "confidence_bonus_a", "confidence_bonus_b", "uncertainty_score_a", "uncertainty_score_b",
                            "history_length_a", "history_length_b", "effective_selected_count_a", "effective_selected_count_b",
                            "gradient_norm_a", "gradient_norm_b", "parameter_norm_a", "parameter_norm_b",
                            "selection_overlap_rate", "prediction_agreement_rate", "accuracy_a", "accuracy_b", "accuracy_ensemble"]
                if method_config.variant == "hard":
                    aggregate_keys.extend([
                        "outlier_count_a", "outlier_count_b", "retained_count_a",
                        "retained_count_b", "outlier_ratio_a", "outlier_ratio_b",
                        "hard_robust_mean_a", "hard_robust_mean_b",
                        "hard_confidence_bonus_a", "hard_confidence_bonus_b",
                    ])
                for key in aggregate_keys:
                    sums[key] = sums.get(key, 0.0) + result.metrics[key] * count
                for key in (
                    "robust_mean_min_a", "robust_mean_min_b",
                    "uncertainty_score_min_a", "uncertainty_score_min_b",
                    "history_length_min_a", "history_length_min_b",
                    "effective_selected_count_min_a", "effective_selected_count_min_b",
                ):
                    extrema[key] = min(extrema.get(key, float("inf")), result.metrics[key])
                for key in (
                    "robust_mean_max_a", "robust_mean_max_b",
                    "uncertainty_score_max_a", "uncertainty_score_max_b",
                    "history_length_max_a", "history_length_max_b",
                    "effective_selected_count_max_a", "effective_selected_count_max_b",
                    "gradient_norm_a", "gradient_norm_b",
                ):
                    extrema[key] = max(extrema.get(key, float("-inf")), result.metrics[key])
                sums["loss_a_on_selected_by_b"] = sums.get("loss_a_on_selected_by_b", 0.0) + result.metrics["loss_a_on_selected_by_b"] * result.metrics["selected_by_b_count"]
                sums["loss_b_on_selected_by_a"] = sums.get("loss_b_on_selected_by_a", 0.0) + result.metrics["loss_b_on_selected_by_a"] * result.metrics["selected_by_a_count"]
                clean_a += sum(not corruption[int(i)] for i in result.metadata["selected_by_a_indices"].tolist())
                clean_b += sum(not corruption[int(i)] for i in result.metadata["selected_by_b_indices"].tolist())
            algorithm.on_cycle_end(state)
            validation = _evaluate_peers(algorithm, validation_loader, criterion, device)
            lr_a, lr_b = float(optimizer_a.param_groups[0]["lr"]), float(optimizer_b.param_groups[0]["lr"])
            algorithm.step_schedulers()
            row = {"event": "epoch", "epoch": epoch + 1, "global_step": state.step,
                   "remember_rate": method_config.rate_at(epoch), "learning_rate_a": lr_a, "learning_rate_b": lr_b,
                   "train_loss_a_on_selected_by_b": sums["loss_a_on_selected_by_b"] / selected_b_total,
                   "train_loss_b_on_selected_by_a": sums["loss_b_on_selected_by_a"] / selected_a_total,
                   "selected_by_a_count": selected_a_total, "selected_by_b_count": selected_b_total,
                   "selected_by_a_ratio": selected_a_total / samples,
                   "selected_by_b_ratio": selected_b_total / samples,
                   "selected_clean_precision_a": clean_a / selected_a_total,
                   "selected_clean_precision_b": clean_b / selected_b_total,
                   "validation_accuracy_a": validation["accuracy_a"], "validation_accuracy_b": validation["accuracy_b"],
                   "mean_peer_accuracy": validation["mean_peer_accuracy"],
                   "validation_accuracy_ensemble": validation["ensemble_accuracy"],
                   "optimizer_steps_a": algorithm.private_state.optimizer_steps_a,
                   "optimizer_steps_b": algorithm.private_state.optimizer_steps_b,
                   "history_window_start_epoch": algorithm.private_state.history_a.window_start_epoch,
                   "history_window_epoch_count": algorithm.private_state.history_a.window_epoch_count}
            for key, value in sums.items():
                if key not in {"loss_a_on_selected_by_b", "loss_b_on_selected_by_a"}:
                    row["train_" + key] = value / samples
            for key, value in extrema.items():
                suffix = "_max" if key in {"gradient_norm_a", "gradient_norm_b"} else ""
                row["train_" + key + suffix] = value
            improved = validation["mean_peer_accuracy"] > best_primary
            if improved:
                best_primary, best_epoch, best_metrics = validation["mean_peer_accuracy"], epoch, _best_component_state(validation)
            kwargs = {"best_epoch": best_epoch, "best_validation_accuracy": best_primary,
                      "selection_split": "validation", "best_selection_accuracy": best_primary,
                      "noise": noise_metadata, "component_states": {"cnlcu_best": best_metrics}}
            save_checkpoint(run_dir / "last.pt", algorithm, state, epoch, config, scheduler=None, **kwargs)
            if improved:
                save_checkpoint(run_dir / "best.pt", algorithm, state, epoch, config, scheduler=None, **kwargs)
            metrics_file.write(json.dumps(row) + "\n"); metrics_file.flush(); print(json.dumps(row), flush=True)
        best_payload = read_checkpoint(run_dir / "best.pt", device)
        algorithm.model_a.load_state_dict(best_payload["model"]["a"])
        algorithm.model_b.load_state_dict(best_payload["model"]["b"])
        test = _evaluate_peers(algorithm, test_loader, criterion, device)
        final = {"event": "final", "completed_epochs": epochs, "global_step": state.step,
                 "best_epoch": best_epoch + 1, "best_validation_accuracy_a": best_metrics["accuracy_a"],
                 "best_validation_accuracy_b": best_metrics["accuracy_b"],
                 "best_mean_peer_accuracy": best_metrics["mean_peer_accuracy"],
                 "best_validation_accuracy_ensemble": best_metrics["ensemble_accuracy"],
                 "test_accuracy_a": test["accuracy_a"], "test_accuracy_b": test["accuracy_b"],
                 "test_mean_peer_accuracy": test["mean_peer_accuracy"],
                 "test_accuracy_ensemble": test["ensemble_accuracy"], "noise": noise_metadata}
        if device.type == "cuda": final["max_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        metrics_file.write(json.dumps(final) + "\n"); print(json.dumps(final), flush=True)
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    algorithm.on_run_end(state); algorithm.close()
    return run_dir


__all__ = ["run_cnlcu_experiment"]
