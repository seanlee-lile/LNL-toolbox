from __future__ import annotations

"""Small, resumable VolMinNet lifecycle over the toolbox data contract.

The runner keeps the paper's joint classifier/transition update separate from
PCSE's downstream statistics workflow.  The synthetic backend is intentional
for smoke tests; CIFAR data adapters can supply the same batch contract.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lnl_toolbox.algorithms.pcse.volmin import PaperVolMinTransition, paper_volmin_objective
from lnl_toolbox.data.multiclass_synthetic import MulticlassTensorDataset, generate_synthetic_multiclass
from lnl_toolbox.noise.generators import generate_symmetric
from lnl_toolbox.runtime import seed_everything
from lnl_toolbox.training.reproduction_data import build_reproduction_model, prepare_noisy_classification
from lnl_toolbox.training.interfaces import RunContext


class _VolMinMLP(nn.Module):
    def __init__(self, dimension: int, width: int, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(dimension, width), nn.ReLU())
        self.classifier = nn.Linear(width, classes)

    def forward(self, value: Tensor) -> Tensor:
        return self.classifier(self.features(value))


def _run_dir(config: Mapping[str, Any], output_dir: str | Path | None, resume: str | Path | None) -> Path:
    if resume is not None:
        path = Path(resume).resolve().parent
    elif output_dir is not None:
        path = Path(output_dir).expanduser().resolve()
    else:
        path = Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_loaders(config: Mapping[str, Any], run_dir: Path) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    data = config.get("data", {})
    classes, dimension = int(data.get("num_classes", 3)), int(data.get("dimension", 6))
    if str(data.get("name", "synthetic_multiclass")).lower() in {"cifar10", "cifar100"}:
        prepared = prepare_noisy_classification(config, run_dir, int(config.get("seed", 1)))
        return prepared.train_loader, prepared.validation_loader, prepared.test_loader, 0, prepared.num_classes
    train_n, val_n, test_n = int(data.get("train_size", 90)), int(data.get("validation_size", 30)), int(data.get("test_size", 30))
    seed = int(config.get("seed", 1))
    train = generate_synthetic_multiclass(train_n, dimension, classes, seed, start_index=0, split="train")
    val = generate_synthetic_multiclass(val_n, dimension, classes, seed + 1, start_index=train_n, split="validation")
    test = generate_synthetic_multiclass(test_n, dimension, classes, seed + 2, start_index=train_n + val_n, split="test")
    noise = config.get("noise", {})
    manifest = generate_symmetric(train.labels, classes, float(noise.get("rate", 0.2)), int(noise.get("seed", seed + 10)), "synthetic_multiclass", sampling=str(noise.get("sampling", "per_class")), rng=str(noise.get("rng", "default_rng")))
    train_set = MulticlassTensorDataset(train, manifest.noisy_targets)
    batch = int(config.get("loader", {}).get("batch_size", 30))
    return DataLoader(train_set, batch_size=batch, shuffle=True), DataLoader(MulticlassTensorDataset(val), batch_size=batch), DataLoader(MulticlassTensorDataset(test), batch_size=batch), dimension, classes


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input"].to(device))
            correct += int(logits.argmax(1).eq(batch["target"].to(device)).sum())
            total += int(logits.shape[0])
    return correct / max(total, 1)


def run_volmin_experiment(config: dict[str, Any], output_dir: str | Path | None = None, resume: str | Path | None = None, *, context: RunContext | None = None) -> Path:
    run_dir = _run_dir(config, output_dir, resume)
    seed_everything(int(config.get("seed", 1)))
    train_loader, val_loader, test_loader, dimension, classes = _make_loaders(config, run_dir)
    model_cfg = config.get("model", {})
    device = torch.device(str(config.get("trainer", {}).get("device", "cpu")))
    model = (_VolMinMLP(dimension, int(model_cfg.get("hidden_width", 16)), classes) if dimension else build_reproduction_model(model_cfg, config["data"], classes)).to(device)
    trans_cfg = config.get("transition", {})
    transition = PaperVolMinTransition(classes, initial_weight=float(trans_cfg.get("initial_weight", 4.5))).to(device=device, dtype=torch.float64)
    opt_cfg = config.get("optimizer", {"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4})
    optimizer = torch.optim.SGD(model.parameters(), lr=float(opt_cfg.get("lr", 0.01)), momentum=float(opt_cfg.get("momentum", 0.9)), weight_decay=float(opt_cfg.get("weight_decay", 1e-4)))
    transition_optimizer = torch.optim.SGD(transition.parameters(), lr=float(opt_cfg.get("transition_lr", opt_cfg.get("lr", 0.01))), momentum=float(opt_cfg.get("momentum", 0.9)), weight_decay=float(opt_cfg.get("transition_weight_decay", 0.0)))
    scheduler_cfg = config.get("scheduler", {}) or {}
    milestones = [int(value) for value in scheduler_cfg.get("milestones", [])]
    model_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=float(scheduler_cfg.get("gamma", 0.1))) if milestones else None
    transition_scheduler = torch.optim.lr_scheduler.MultiStepLR(transition_optimizer, milestones=milestones, gamma=float(scheduler_cfg.get("gamma", 0.1))) if milestones else None
    start = 0
    checkpoint = run_dir / "last.pt"
    if resume is not None and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"]); transition.load_state_dict(payload["transition"]); optimizer.load_state_dict(payload["optimizer"]); transition_optimizer.load_state_dict(payload["transition_optimizer"]); start = int(payload["epoch"])
    epochs = int(config.get("trainer", {}).get("epochs", 1))
    metrics_path = run_dir / "metrics.jsonl"
    session = context.session if context is not None and context.state.get("lifecycle_active") else None
    if session is not None:
        session.start_phase("joint_training", total_units=epochs)
    with metrics_path.open("a", encoding="utf-8") as metrics:
        for epoch in range(start, epochs):
            model.train(); total = 0.0
            for batch in train_loader:
                inputs, targets = batch["input"].to(device), batch["target"].to(device)
                logits = model(inputs).to(torch.float64)
                objective, info = paper_volmin_objective(logits, targets, transition.matrix(), lambda_volume=float(trans_cfg.get("lambda_volume", 1e-4)), determinant_tolerance=float(trans_cfg.get("determinant_tolerance", 1e-8)), condition_limit=float(trans_cfg.get("condition_limit", 1e8)))
                optimizer.zero_grad(set_to_none=True); transition_optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step(); transition_optimizer.step(); total += float(objective.detach()) * inputs.shape[0]
            if model_scheduler is not None:
                model_scheduler.step()
                transition_scheduler.step()
            record = {"epoch": epoch + 1, "train_loss": total / len(train_loader.dataset), "validation_accuracy": _accuracy(model, val_loader, device), "test_accuracy": _accuracy(model, test_loader, device), "transition": transition.matrix().detach().cpu().tolist()}
            if session is not None:
                session.log_epoch(
                    epoch + 1,
                    phase="joint_training",
                    **{key: value for key, value in record.items() if key != "epoch"},
                )
            else:
                metrics.write(json.dumps(record) + "\n"); metrics.flush()
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "transition": transition.state_dict(), "optimizer": optimizer.state_dict(), "transition_optimizer": transition_optimizer.state_dict(), "config": config}, checkpoint)
    if session is not None:
        session.end_phase("joint_training", completed_units=max(0, epochs - start))
        session.emit("final", phase="evaluation", method="volmin", completed_epochs=epochs,
                     test_accuracy=record.get("test_accuracy") if epochs else None)
    return run_dir


__all__ = ["run_volmin_experiment"]
