from __future__ import annotations

"""Framework-neutral binary experiment utilities for UCI and CIFAR views."""

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lnl_toolbox.algorithms.binary_risk import NatarajanUnbiasedRisk
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
        optimizer.step()
        count = int(targets.numel())
        total_loss += float(loss.detach().item()) * count
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        samples += count
    if samples == 0:
        raise ValueError("binary loader must not be empty")
    return {"loss": total_loss / samples, "accuracy": correct / samples, "samples": float(samples)}


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


def run_binary_experiment(config: Mapping[str, Any], output_dir: str | Path | None = None) -> Path:
    """Run a single configured binary experiment and persist its metrics."""

    resolved_config, record = resolve_parameter_sampling(config)
    resolved_config = dict(resolved_config)
    resolved_config.setdefault("loader", {"batch_size": int(resolved_config.get("batch_size", 64))})
    destination = Path(output_dir or resolved_config.get("output_root", "artifacts/binary")).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seed = int(resolved_config.get("seed", 1))
    prepared = prepare_experiment_data(
        resolved_config,
        requirements=DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.TEST})),
        run_dir=destination,
        seed=seed,
    )
    if prepared.num_classes != 2:
        raise ValueError(
            "binary experiment requires a registered binary dataset view "
            f"with exactly two classes; got {prepared.dataset!r} with "
            f"{prepared.num_classes} classes"
        )
    loader = prepared.loader(DataRole.TRAIN)
    sample = prepared.dataset_for(DataRole.TRAIN)[0]
    input_dim = int(torch.as_tensor(sample["input"]).numel())
    model_config = dict(resolved_config.get("model", {}))
    if not model_config:
        model_config = {"name": "mlp", "hidden_width": resolved_config.get("hidden_width", 128)}
    model = build_binary_model(input_dim, model_config)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(resolved_config.get("learning_rate", 0.01)), momentum=0.9)
    risk = None
    risk_config = resolved_config.get("risk")
    if isinstance(risk_config, Mapping):
        risk = NatarajanUnbiasedRisk(float(risk_config["rho_positive"]), float(risk_config["rho_negative"]))
    epochs = int(resolved_config.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    rows = []
    for epoch in range(epochs):
        row = train_binary_epoch(model, loader, optimizer, risk=risk)
        row["epoch"] = float(epoch + 1)
        if len(prepared.dataset_for(DataRole.TEST)) > 0:
            test_loader = prepared.loader(DataRole.TEST, shuffle=False)
            evaluation = evaluate_binary(model, test_loader)
            row["test_loss"] = evaluation["loss"]
            row["test_accuracy"] = evaluation["accuracy"]
        rows.append(row)
    import json
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
