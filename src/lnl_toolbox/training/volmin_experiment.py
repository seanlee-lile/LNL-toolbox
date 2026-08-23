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

from lnl_toolbox.algorithms.pcse.volmin import PaperVolMinTransition, paper_volmin_objective
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.runtime import seed_everything
from lnl_toolbox.training.reproduction_data import build_reproduction_model
from lnl_toolbox.training.data_service import prepare_experiment_data


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


def _make_loaders(config: Mapping[str, Any], run_dir: Path):
    data = config.get("data", {})
    classes, dimension = int(data.get("num_classes", 3)), int(data.get("dimension", 6))
    seed = int(config.get("seed", 1))
    synthetic = str(data.get("name", "synthetic_multiclass")).lower() == "synthetic_multiclass"
    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.CLEAN_VALIDATION, DataRole.TEST})
        ),
        run_dir=run_dir,
        seed=seed - 1 if synthetic else seed,
    )
    return (
        prepared.loader(DataRole.TRAIN, generator_seed=seed),
        prepared.loader(DataRole.CLEAN_VALIDATION, shuffle=False, generator_seed=seed),
        prepared.loader(DataRole.TEST, shuffle=False, generator_seed=seed),
        dimension if synthetic else 0,
        prepared.num_classes,
    )


def _accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input"].to(device))
            correct += int(logits.argmax(1).eq(batch["target"].to(device)).sum())
            total += int(logits.shape[0])
    return correct / max(total, 1)


def run_volmin_experiment(config: dict[str, Any], output_dir: str | Path | None = None, resume: str | Path | None = None) -> Path:
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
            metrics.write(json.dumps(record) + "\n"); metrics.flush()
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "transition": transition.state_dict(), "optimizer": optimizer.state_dict(), "transition_optimizer": transition_optimizer.state_dict(), "config": config}, checkpoint)
    return run_dir


__all__ = ["run_volmin_experiment"]
