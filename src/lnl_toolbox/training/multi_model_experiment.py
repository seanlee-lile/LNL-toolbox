from __future__ import annotations

"""Reusable CIFAR experiment runner for jointly trained model groups."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.multi_model import ModelGroup
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.core.hyperparameters import resolve_parameter_sampling
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.evaluation.classification import evaluate_model_group
from lnl_toolbox.models.cifar_six_conv import CifarSixConvNet
from lnl_toolbox.plugins.builtin import (
    build_builtin_loss,
    build_builtin_multi_model_algorithm,
    build_builtin_selector,
)
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import (
    load_checkpoint,
    read_checkpoint,
    save_checkpoint,
)
from lnl_toolbox.training.data_service import prepare_experiment_data
from lnl_toolbox.training.experiment import build_model
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    effective_subset_actual_rate,
    noise_mode,
)
from lnl_toolbox.training.progress import (
    TerminalTrainingProgress,
    standardize_epoch_row,
    write_training_curves_svg,
)


def _build_member(config: Mapping[str, Any], num_classes: int):
    name = str(config.get("name", "")).strip().lower()
    if name == "cifar_six_conv":
        return CifarSixConvNet(
            num_classes,
            input_channels=int(config.get("input_channels", 3)),
            batch_norm_momentum=float(
                config.get("batch_norm_momentum", 0.1)
            ),
        )
    return build_model(config, num_classes)


def _build_optimizer(parameters, config: Mapping[str, Any]):
    name = str(config.get("name", "adam")).strip().lower()
    common = {
        "lr": float(config["lr"]),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            betas=tuple(float(value) for value in config.get("betas", (0.9, 0.999))),
            **common,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            betas=tuple(float(value) for value in config.get("betas", (0.9, 0.999))),
            **common,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            momentum=float(config.get("momentum", 0.9)),
            nesterov=bool(config.get("nesterov", False)),
            **common,
        )
    raise ValueError(f"Unsupported multi-model optimizer: {name}")


def _apply_epoch_optimizer_schedule(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any] | None,
    *,
    epoch: int,
    total_epochs: int,
) -> None:
    if not config or str(config.get("name", "none")).lower() == "none":
        return
    name = str(config["name"]).strip().lower()
    if name != "linear_decay":
        raise ValueError("multi-model scheduler must be none or linear_decay")
    start = int(config["start_epoch"])
    end = int(config.get("end_epoch", total_epochs))
    if not 0 <= start < end <= total_epochs:
        raise ValueError("linear_decay requires 0 <= start < end <= epochs")
    initial_lr = float(config["initial_lr"])
    final_lr = float(config.get("final_lr", 0.0))
    if epoch < start:
        learning_rate = initial_lr
    else:
        progress = min((epoch - start) / (end - start), 1.0)
        learning_rate = initial_lr + progress * (final_lr - initial_lr)
    beta1_before = config.get("beta1_before")
    beta1_after = config.get("beta1_after")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        if "betas" in group and beta1_before is not None:
            beta1 = (
                float(beta1_before)
                if epoch < start
                else float(beta1_after)
            )
            group["betas"] = (beta1, float(group["betas"][1]))


def _run_directory(config: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
    else:
        root = Path(config.get("output_root", "artifacts/runs"))
        path = root / datetime.now().strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _environment(seed: int, device: torch.device) -> dict[str, Any]:
    return {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "python_executable": __import__("sys").executable,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "device": str(device),
        "seed": seed,
    }


def run_multi_model_experiment(
    raw_config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Run one configured multi-model algorithm without paper-name branches."""

    config, parameter_record = resolve_parameter_sampling(raw_config)
    config = dict(config)
    seed = int(config.get("seed", 1))
    epochs = int(config["trainer"]["epochs"])
    if epochs <= 0:
        raise ValueError("trainer.epochs must be positive")
    seed_everything(seed)
    device = resolve_device(str(config["trainer"].get("device", "auto")))
    run_dir = (
        Path(resume).resolve().parent
        if resume is not None
        else _run_directory(config, output_dir)
    )
    checkpoint_payload = (
        None if resume is None else read_checkpoint(resume, device)
    )
    if checkpoint_payload is not None:
        saved = dict(checkpoint_payload.get("config") or {})
        if saved != config:
            raise ValueError("Resume configuration changed")

    data_config = dict(config["data"])
    validation_size = int(data_config.get("validation_size", 0))
    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.CLEAN_VALIDATION, DataRole.TEST}),
            manifest_scope="effective_train",
        ),
        run_dir=run_dir,
        seed=seed,
        checkpoint_payload=checkpoint_payload,
    )
    dataset_name = prepared.dataset
    num_classes = prepared.num_classes
    manifest, manifest_path = prepared.manifest, prepared.manifest_path
    noise_metadata = None
    if manifest is not None and manifest_path is not None:
        noise_metadata = checkpoint_noise_metadata(
            manifest,
            manifest_path,
            run_dir,
            effective_subset_actual_rate(manifest, prepared.train_indices),
            mode=noise_mode(config),
        )
    train_loader = prepared.loader(DataRole.TRAIN)
    validation_loader = (
        None
        if validation_size == 0
        else prepared.loader(DataRole.CLEAN_VALIDATION, shuffle=False)
    )
    test_loader = prepared.loader(DataRole.TEST, shuffle=False)

    model_configs = config.get("models")
    if not isinstance(model_configs, list) or len(model_configs) < 2:
        raise ValueError("models must contain at least two model mappings")
    members = {}
    for index, model_config in enumerate(model_configs, start=1):
        if not isinstance(model_config, Mapping):
            raise TypeError("each model configuration must be a mapping")
        members[f"model_{index}"] = _build_member(model_config, num_classes)
    models = ModelGroup(members)
    optimizer = _build_optimizer(models.parameters(), config["optimizer"])
    criterion = build_builtin_loss(config.get("loss", {"name": "ce"})).to(device)
    selector = build_builtin_selector(config["selector"])
    algorithm = build_builtin_multi_model_algorithm(
        config["algorithm"],
        models=models,
        optimizer=optimizer,
        loss=criterion,
        selector=selector,
        device=device,
    )
    algorithm.setup(ExperimentContext(run_dir, config, seed))

    state = RunState(phase="train")
    completed_epoch = -1
    best_epoch = -1
    best_accuracy = float("-inf")
    if resume is not None:
        state, completed_epoch, checkpoint_payload = load_checkpoint(
            resume, algorithm, device, scheduler=None
        )
        best_epoch = int(checkpoint_payload["best_epoch"])
        best_accuracy = float(checkpoint_payload["best_selection_accuracy"])

    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if parameter_record is not None:
        (run_dir / "parameter_record.json").write_text(
            json.dumps(parameter_record.to_dict(), indent=2), encoding="utf-8"
        )
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(seed, device), indent=2), encoding="utf-8"
    )
    if noise_metadata is not None:
        (run_dir / "noise_summary.json").write_text(
            json.dumps(noise_metadata, indent=2), encoding="utf-8"
        )

    evaluation = dict(config.get("evaluation") or {})
    selection_split = str(
        evaluation.get("selection_split", "validation")
    ).lower()
    if selection_split == "test":
        if not bool(evaluation.get("allow_test_selection", False)):
            raise ValueError("test selection requires allow_test_selection=true")
        selection_loader = test_loader
    elif selection_split == "validation":
        if validation_loader is None:
            raise ValueError("validation selection requires validation data")
        selection_loader = validation_loader
    else:
        raise ValueError("selection_split must be validation or test")

    progress_config = config["trainer"].get("progress", {})
    if isinstance(progress_config, bool):
        progress_config = {"enabled": progress_config}
    metrics_path = run_dir / "metrics.jsonl"
    curve_rows: list[dict[str, Any]] = []
    if metrics_path.is_file():
        curve_rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("event") == "epoch"
        ]

    algorithm.on_run_start(state)
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(completed_epoch + 1, epochs):
            state.cycle = epoch
            _apply_epoch_optimizer_schedule(
                optimizer,
                config.get("scheduler"),
                epoch=epoch,
                total_epochs=epochs,
            )
            algorithm.on_cycle_start(state)
            totals: dict[str, float] = {}
            samples = 0.0
            selected_samples = 0.0
            progress = TerminalTrainingProgress(
                epoch=epoch + 1,
                total_epochs=epochs,
                total_batches=len(train_loader),
                update_interval=int(progress_config.get("update_interval", 20)),
                enabled=bool(progress_config.get("enabled", False)),
            )
            for batch_number, raw_batch in enumerate(train_loader, start=1):
                result = algorithm.step(Batch(raw_batch), state)
                count = float(result.metrics["samples"])
                selected = float(result.metrics["selected_samples"])
                samples += count
                selected_samples += selected
                for name, value in result.metrics.items():
                    if name in {"samples", "selected_samples", "selected_ratio"}:
                        continue
                    weight = selected if name == "loss" else count
                    totals[name] = totals.get(name, 0.0) + float(value) * weight
                progress.update(
                    batch_number,
                    loss=float(result.metrics["loss"]),
                    accuracy=float(result.metrics["accuracy"]),
                )
            algorithm.on_cycle_end(state)
            selected_metrics = evaluate_model_group(
                models, selection_loader, criterion, device
            )
            row = standardize_epoch_row({
                "event": "epoch",
                "epoch": epoch + 1,
                "global_step": state.step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": totals["loss"] / selected_samples,
                "train_all_sample_loss": totals["all_sample_loss"] / samples,
                "train_accuracy": totals["accuracy"] / samples,
                "train_model_1_accuracy": totals["model_1_accuracy"] / samples,
                "train_model_2_accuracy": totals["model_2_accuracy"] / samples,
                "train_agreement_loss": totals["agreement_loss"] / samples,
                "selected_samples": selected_samples,
                "selected_ratio": selected_samples / samples,
                "selection_split": selection_split,
                "selection_loss": selected_metrics["loss"],
                "selection_accuracy": selected_metrics["accuracy"],
                "validation_loss": selected_metrics["loss"],
                "validation_accuracy": selected_metrics["accuracy"],
                "selection_model_1_accuracy": selected_metrics[
                    "model_1_accuracy"
                ],
                "selection_model_2_accuracy": selected_metrics[
                    "model_2_accuracy"
                ],
            })
            curve_rows.append(row)
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            if selected_metrics["accuracy"] > best_accuracy:
                best_accuracy = float(selected_metrics["accuracy"])
                best_epoch = epoch + 1
                save_checkpoint(
                    run_dir / "best.pt",
                    algorithm,
                    state,
                    epoch,
                    config,
                    best_epoch=best_epoch,
                    best_validation_accuracy=best_accuracy,
                    selection_split=selection_split,
                    best_selection_accuracy=best_accuracy,
                    noise=noise_metadata,
                    parameter_record=(
                        None
                        if parameter_record is None
                        else parameter_record.to_dict()
                    ),
                )
            save_checkpoint(
                run_dir / "last.pt",
                algorithm,
                state,
                epoch,
                config,
                best_epoch=best_epoch,
                best_validation_accuracy=best_accuracy,
                selection_split=selection_split,
                best_selection_accuracy=best_accuracy,
                noise=noise_metadata,
                parameter_record=(
                    None if parameter_record is None else parameter_record.to_dict()
                ),
            )
            if bool(progress_config.get("curves", False)):
                write_training_curves_svg(
                    curve_rows, run_dir / "training_curves.svg"
                )

        algorithm.on_run_end(state)
        test_metrics = evaluate_model_group(models, test_loader, criterion, device)
        report_last = int(evaluation.get("report_last_epochs", 0))
        window_rows = curve_rows[-report_last:] if report_last else []
        final = {
            "event": "final",
            "completed_epochs": epochs,
            "global_step": state.step,
            "best_epoch": best_epoch,
            "best_selection_accuracy": best_accuracy,
            "selection_split": selection_split,
            "test_selection_leakage": selection_split == "test",
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_model_1_accuracy": test_metrics["model_1_accuracy"],
            "test_model_2_accuracy": test_metrics["model_2_accuracy"],
        }
        if window_rows:
            member_values = [
                float(row[key])
                for row in window_rows
                for key in (
                    "selection_model_1_accuracy",
                    "selection_model_2_accuracy",
                )
            ]
            final["last_window_epochs"] = report_last
            final["last_window_member_mean_accuracy"] = float(
                np.mean(member_values)
            )
        if noise_metadata is not None:
            final["noise"] = noise_metadata
        metrics_file.write(json.dumps(final) + "\n")
        metrics_file.flush()
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    return run_dir


__all__ = ["run_multi_model_experiment"]
