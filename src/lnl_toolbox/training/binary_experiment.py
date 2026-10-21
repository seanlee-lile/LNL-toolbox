from __future__ import annotations

"""Framework-neutral binary experiment utilities for UCI and CIFAR views."""

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lnl_toolbox.algorithms.binary_risk import NatarajanUnbiasedRisk
from lnl_toolbox.core.config_schema import normalize_experiment_config
from lnl_toolbox.core.hyperparameters import resolve_parameter_sampling
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.data.binary_benchmarks import (
    BinaryBenchmark,
)
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.training.data_service import prepare_experiment_data


class BinaryTensorDataset(Dataset[dict[str, Any]]):
    def __init__(self, benchmark: BinaryBenchmark, targets: np.ndarray | None = None) -> None:
        self.features = torch.from_numpy(benchmark.features)
        self.targets = torch.from_numpy(benchmark.targets if targets is None else np.asarray(targets, dtype=np.int64))
        self.indices = torch.from_numpy(np.asarray(benchmark.global_indices, dtype=np.int64))
        if self.targets.shape != self.indices.shape:
            raise ValueError("binary targets must align with indices")

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input": self.features[index],
            "target": self.targets[index],
            "index": self.indices[index],
        }


def build_binary_mlp(input_dim: int, hidden_width: int = 128) -> nn.Module:
    if input_dim <= 0 or hidden_width <= 0:
        raise ValueError("input_dim and hidden_width must be positive")
    return nn.Sequential(nn.Linear(input_dim, hidden_width), nn.ReLU(), nn.Linear(hidden_width, 2))


def build_binary_linear(input_dim: int) -> nn.Module:
    """Return the linear two-logit classifier used by the paper's logistic loss."""

    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    return nn.Linear(input_dim, 2)


def build_binary_model(input_dim: int, config: Mapping[str, Any]) -> nn.Module:
    name = str(config.get("name", "mlp")).strip().lower()
    if name == "linear":
        return build_binary_linear(input_dim)
    if name == "mlp":
        return build_binary_mlp(input_dim, int(config.get("hidden_width", 128)))
    raise ValueError("binary model must be 'linear' or 'mlp'")


def train_binary_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    *,
    risk: Any | None = None,
) -> dict[str, float]:
    model.train()
    resolved = torch.device(device)
    criterion = CrossEntropyLoss().to(resolved)
    total_loss = 0.0
    correct = 0
    samples = 0
    input_square_sum = logit_square_sum = 0.0
    input_elements = logit_elements = 0
    gradient_norm_max = 0.0
    for batch in loader:
        inputs = batch["input"].to(resolved, dtype=torch.float32)
        targets = batch["target"].to(resolved, dtype=torch.long)
        logits = model(inputs)
        values = criterion(logits, targets) if risk is None else risk.per_sample_risk(
            logits=logits, noisy_targets=targets, base_loss=criterion, transition=None
        )
        loss = values.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = sum(float(parameter.grad.detach().square().sum())
                            for parameter in model.parameters() if parameter.grad is not None) ** 0.5
        gradient_norm_max = max(gradient_norm_max, gradient_norm)
        if not np.isfinite(float(loss.detach())) or not np.isfinite(gradient_norm):
            raise ValueError("Binary Risk produced a non-finite objective or gradient")
        input_square_sum += float(inputs.detach().double().square().sum())
        input_elements += inputs.numel()
        logit_square_sum += float(logits.detach().double().square().sum())
        logit_elements += logits.numel()
        optimizer.step()
        count = int(targets.numel())
        total_loss += float(loss.detach().item()) * count
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        samples += count
    if samples == 0:
        raise ValueError("binary loader must not be empty")
    return {"loss": total_loss / samples, "accuracy": correct / samples, "samples": float(samples),
            "input_rms": (input_square_sum / input_elements) ** 0.5,
            "logit_rms": (logit_square_sum / logit_elements) ** 0.5,
            "gradient_norm_max": gradient_norm_max,
            "parameter_norm": sum(float(parameter.detach().double().square().sum()) for parameter in model.parameters()) ** 0.5}


@torch.no_grad()
def evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device | str = "cpu") -> dict[str, float]:
    model.eval()
    resolved = torch.device(device)
    criterion = CrossEntropyLoss().to(resolved)
    total_loss = 0.0
    correct = 0
    samples = 0
    for batch in loader:
        inputs = batch["input"].to(resolved, dtype=torch.float32)
        targets = batch["target"].to(resolved, dtype=torch.long)
        logits = model(inputs)
        values = criterion(logits, targets)
        count = int(targets.numel())
        total_loss += float(values.mean().item()) * count
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        samples += count
    if samples == 0:
        raise ValueError("binary loader must not be empty")
    return {"loss": total_loss / samples, "accuracy": correct / samples, "samples": float(samples)}


def run_binary_experiment(
    config: Mapping[str, Any], output_dir: str | Path | None = None, *,
    requirements: DataRequirements | None = None,
) -> Path:
    """Run a single configured binary experiment and persist its metrics."""

    resolved_config, record = resolve_parameter_sampling(config)
    resolved_config = dict(resolved_config)
    resolved_config.setdefault("execution", {"runner": "binary"})
    resolved_config = normalize_experiment_config(resolved_config)
    risk_config = resolved_config.get("risk")
    risk = None
    if risk_config is not None:
        if not isinstance(risk_config, Mapping):
            raise TypeError("risk configuration must be a mapping")
        risk_name = str(risk_config.get("name", "")).strip().lower()
        if risk_name not in {"natarajan", "natarajan_unbiased"}:
            raise ValueError("binary experiment supports risk.name: natarajan or natarajan_unbiased")
        risk = NatarajanUnbiasedRisk(float(risk_config["rho_positive"]), float(risk_config["rho_negative"]))
        noise = resolved_config.get("noise", {})
        mismatch = any(key in noise and float(noise[key]) != float(risk_config[key])
                       for key in ("rho_positive", "rho_negative"))
        allow_mismatch = risk_config.get("allow_noise_prior_mismatch", False)
        if not isinstance(allow_mismatch, bool):
            raise ValueError("risk.allow_noise_prior_mismatch must be boolean")
        if mismatch and not allow_mismatch:
            raise ValueError("Binary Risk correction rates differ from noise generation rates; set risk.allow_noise_prior_mismatch explicitly to study misspecification")
    destination = Path(output_dir or resolved_config.get("output_root", "artifacts/binary")).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seed = int(resolved_config.get("seed", 1))
    if requirements is None:
        from lnl_toolbox.training.runners import resolve_data_requirements

        requirements = resolve_data_requirements(resolved_config, expected_runner="binary")
    prepared = prepare_experiment_data(
        resolved_config,
        requirements=requirements,
        run_dir=destination,
        seed=seed,
    )
    if prepared.num_classes != 2:
        raise ValueError(
            "binary experiment requires a registered binary dataset view "
            f"with exactly two classes; got {prepared.dataset!r} with "
            f"{prepared.num_classes} classes"
        )
    prepared.loader_config = {**prepared.loader_config, "persistent_workers": True}
    loader = prepared.loader(DataRole.TRAIN)
    validation_loader = prepared.validation_loader(shuffle=False) if prepared.validation_indices.size else None
    test_loader = prepared.loader(DataRole.TEST, shuffle=False)
    if prepared.input_spec.shape is None:
        sample = prepared.dataset_for(DataRole.TRAIN)[0]
        input_dim = int(torch.as_tensor(sample["input"]).numel())
    else:
        input_dim = int(np.prod(prepared.input_spec.shape))
    model_config = dict(resolved_config.get("model", {}))
    model = nn.Sequential(nn.Flatten(start_dim=1), build_binary_model(input_dim, model_config))
    optimizer_config = dict(resolved_config["optimizer"])
    if str(optimizer_config.get("name", "sgd")).lower() != "sgd":
        raise ValueError("binary experiment currently requires optimizer.name: sgd")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        momentum=float(optimizer_config.get("momentum", 0.9)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    epochs = int(resolved_config["trainer"]["epochs"])
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    rows = []
    web_rows = []
    metrics_path = destination / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    from copy import deepcopy
    from lnl_toolbox.training.early_stopping import StageEarlyStopping
    monitor = str(resolved_config["trainer"].get("early_stopping_monitor", "validation_accuracy"))
    if monitor not in {"validation_accuracy", "validation_loss"}:
        raise ValueError("trainer.early_stopping_monitor must be validation_accuracy or validation_loss")
    stopping = StageEarlyStopping(resolved_config, "trainer.early_stopping", monitor,
                                 mode="min" if monitor == "validation_loss" else "max", run_dir=destination)
    if stopping.enabled and not prepared.validation_indices.size:
        raise ValueError("trainer.early_stopping requires a validation split")
    best_model = None
    best_epoch = None
    for epoch in range(epochs):
        row = train_binary_epoch(model, loader, optimizer, risk=risk)
        row["epoch"] = float(epoch + 1)
        if prepared.validation_indices.size:
            validation = evaluate_binary(model, validation_loader)
            row["validation_loss"] = validation["loss"]
            row["validation_accuracy"] = validation["accuracy"]
        if len(prepared.dataset_for(DataRole.TEST)) > 0:
            evaluation = evaluate_binary(model, test_loader)
            row["test_loss"] = evaluation["loss"]
            row["test_accuracy"] = evaluation["accuracy"]
        rows.append(row)
        web_row = {"event": "epoch", **row, "epoch": epoch + 1}
        web_rows.append(web_row)
        metrics_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in web_rows),
            encoding="utf-8",
        )
        previous_best = stopping.counter.best
        stopped = stopping.update(web_row)
        if stopping.enabled and stopping.counter.best != previous_best:
            best_model = deepcopy(model.state_dict())
            best_epoch = epoch + 1
        if stopped:
            break

    if best_model is not None:
        model.load_state_dict(best_model)
        torch.save({"model": best_model, "config": resolved_config}, destination / "best.pt")

    final = {
        "event": "final",
        "completed_epochs": len(rows),
        "best_epoch": best_epoch,
        "selection_split": "validation" if stopping.enabled else None,
        "selection_metric": monitor,
        "best_selection_value": stopping.counter.best if stopping.enabled else None,
        "best_selection_accuracy": stopping.counter.best if stopping.enabled and monitor == "validation_accuracy" else None,
        "test_checkpoint": "best.pt" if best_model is not None else "last_model",
    }
    if len(prepared.dataset_for(DataRole.TEST)) > 0:
        evaluation = evaluate_binary(model, test_loader)
        final.update(test_loss=evaluation["loss"], test_accuracy=evaluation["accuracy"],
                     test_samples=int(evaluation["samples"]))
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(final, sort_keys=True) + "\n")
    (destination / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    (destination / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
    if record is not None:
        (destination / "parameter_record.json").write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return destination


__all__ = [
    "BinaryTensorDataset",
    "build_binary_mlp",
    "build_binary_linear",
    "build_binary_model",
    "evaluate_binary",
    "run_binary_experiment",
    "train_binary_epoch",
]
