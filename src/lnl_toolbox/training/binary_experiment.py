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
from lnl_toolbox.data.binary_benchmarks import (
    BinaryBenchmark,
    corrupt_binary_labels,
    load_binary_npz,
)
from lnl_toolbox.data.binary_synthetic import generate_synthetic_binary_2d
from lnl_toolbox.data.preprocessing import BinaryPreprocessingConfig, BinaryPreprocessor
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss


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
    data_config = dict(resolved_config.get("data", {}))
    source = Path(data_config["path"]) if data_config.get("path") else None
    preprocessor = None
    noise_manifest = None
    data_name = str(data_config.get("name", "")).strip().lower()
    if data_name in {"synthetic_binary_2d", "synthetic_binary"}:
        size = int(data_config.get("train_size", 512))
        data_seed = int(data_config.get("seed", resolved_config.get("seed", 1)))
        clean = generate_synthetic_binary_2d(
            size=size,
            seed=data_seed,
            split="train",
        )
        noise_config = dict(resolved_config.get("noise", {}))
        risk_config = dict(resolved_config.get("risk", {}))
        rho_positive = float(
            noise_config.get("rho_positive", risk_config.get("rho_positive", 0.2))
        )
        rho_negative = float(
            noise_config.get("rho_negative", risk_config.get("rho_negative", 0.1))
        )
        noise_manifest = corrupt_binary_labels(
            clean.labels,
            rho_positive,
            rho_negative,
            int(noise_config.get("seed", data_seed)),
        )
        benchmark = BinaryBenchmark(
            clean.features,
            noise_manifest.noisy_targets,
            data_name,
            global_indices=clean.global_indices,
        )
    elif source is not None and source.suffix.lower() == ".npz":
        benchmark = load_binary_npz(source)
    elif source is not None:
        preprocessor = BinaryPreprocessor(BinaryPreprocessingConfig.from_mapping(data_config.get("preprocessing")))
        benchmark = preprocessor.fit_transform(source, dataset=data_config.get("name", source.stem))
    else:
        raise ValueError("binary data requires data.path or a supported synthetic data.name")
    dataset = BinaryTensorDataset(benchmark)
    loader = DataLoader(dataset, batch_size=int(resolved_config.get("batch_size", 64)), shuffle=True)
    model = build_binary_mlp(benchmark.features.shape[1], int(resolved_config.get("hidden_width", 128)))
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
        rows.append(row)
    destination = Path(output_dir or resolved_config.get("output_root", "artifacts/binary"))
    destination.mkdir(parents=True, exist_ok=True)
    import json
    (destination / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
    if record is not None:
        (destination / "parameter_record.json").write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    if preprocessor is not None:
        preprocessor.save(destination / "preprocessing.json")
    if noise_manifest is not None:
        noise_manifest.save(destination / "noise_manifest.npz")
    (destination / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return destination


__all__ = [
    "BinaryTensorDataset",
    "build_binary_mlp",
    "evaluate_binary",
    "run_binary_experiment",
    "train_binary_epoch",
]
