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
import yaml
from torch import Tensor, nn

from lnl_toolbox.algorithms.pcse.volmin import PaperVolMinTransition, paper_volmin_objective
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state, read_checkpoint, restore_rng_state
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
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


def _make_loaders(
    config: Mapping[str, Any], run_dir: Path, requirements: DataRequirements
):
    data = config.get("data", {})
    classes, dimension = int(data.get("num_classes", 3)), int(data.get("dimension", 6))
    seed = int(config.get("seed", 1))
    synthetic = str(data.get("name", "synthetic_multiclass")).lower() == "synthetic_multiclass"
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=seed - 1 if synthetic else seed,
    )
    return (
        prepared.loader(DataRole.TRAIN, generator_seed=seed),
        prepared.validation_loader(shuffle=False, generator_seed=seed),
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


def _write_epoch_metrics(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_volmin_experiment(
    config: dict[str, Any], output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *, requirements: DataRequirements | None = None,
) -> Path:
    run_dir = _run_dir(config, output_dir, resume)
    seed_everything(int(config.get("seed", 1)))
    if requirements is None:
        from lnl_toolbox.training.runners import resolve_data_requirements
        requirements = resolve_data_requirements(config, expected_runner="volmin")
    train_loader, val_loader, test_loader, dimension, classes = _make_loaders(
        config, run_dir, requirements
    )
    model_cfg = config.get("model", {})
    device = resolve_device(str(config.get("trainer", {}).get("device", "auto")))
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
    rows: list[dict[str, Any]] = []
    payload = read_checkpoint(resume, device) if resume is not None else None
    if payload is not None:
        if payload.get("method") != "volmin" or payload.get("config") != config:
            raise ValueError("VolMin resume configuration mismatch")
        model.load_state_dict(payload["model"])
        transition.load_state_dict(payload["transition"])
        optimizer.load_state_dict(payload["optimizer"])
        transition_optimizer.load_state_dict(payload["transition_optimizer"])
        if model_scheduler is not None and payload.get("model_scheduler") is not None:
            model_scheduler.load_state_dict(payload["model_scheduler"])
        if transition_scheduler is not None and payload.get("transition_scheduler") is not None:
            transition_scheduler.load_state_dict(payload["transition_scheduler"])
        if payload.get("rng_state") is not None:
            restore_rng_state(payload["rng_state"])
        start = int(payload.get("completed_epoch", payload.get("epoch", 0)))
        rows = [
            dict(row)
            for row in payload.get("metrics", [])
            if isinstance(row, Mapping) and row.get("event", "epoch") == "epoch"
        ]
    epochs = int(config.get("trainer", {}).get("epochs", 1))
    metrics_path = run_dir / "metrics.jsonl"
    _write_epoch_metrics(rows, metrics_path)
    criterion = CrossEntropyLoss().to(device)
    for epoch in range(start, epochs):
        model.train(); total_loss = 0.0; total = correct = 0
        for batch in train_loader:
            inputs, targets = batch["input"].to(device), batch["target"].to(device)
            logits = model(inputs).to(torch.float64)
            objective, _ = paper_volmin_objective(
                logits,
                targets,
                transition.matrix(),
                lambda_volume=float(trans_cfg.get("lambda_volume", 1e-4)),
                determinant_tolerance=float(trans_cfg.get("determinant_tolerance", 1e-8)),
                condition_limit=float(trans_cfg.get("condition_limit", 1e8)),
            )
            optimizer.zero_grad(set_to_none=True)
            transition_optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            transition_optimizer.step()
            count = int(targets.numel())
            total_loss += float(objective.detach()) * count
            total += count
            correct += int(logits.argmax(1).eq(targets).sum())
        validation = evaluate_classification(model, val_loader, criterion, device)
        row = standardize_epoch_row({
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "method": "volmin",
            "transition": transition.matrix().detach().cpu().tolist(),
        })
        rows.append(row)
        _write_epoch_metrics(rows, metrics_path)
        print(
            f"VolMin epoch {epoch + 1}/{epochs} loss={row['train_loss']:.5f} "
            f"val={row['validation_accuracy']:.4f}",
            flush=True,
        )
        if model_scheduler is not None:
            model_scheduler.step()
            transition_scheduler.step()
        atomic_save({
            "format_version": 2,
            "method": "volmin",
            "config": config,
            "model": model.state_dict(),
            "transition": transition.state_dict(),
            "optimizer": optimizer.state_dict(),
            "transition_optimizer": transition_optimizer.state_dict(),
            "model_scheduler": None if model_scheduler is None else model_scheduler.state_dict(),
            "transition_scheduler": None if transition_scheduler is None else transition_scheduler.state_dict(),
            "epoch": epoch + 1,
            "completed_epoch": epoch + 1,
            "metrics": rows,
            "rng_state": capture_rng_state(),
        }, run_dir / "last.pt")
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if rows:
        write_training_curves_svg(rows, run_dir / "training_curves.svg")
    test = evaluate_classification(model, test_loader, criterion, device)
    final = {
        "event": "final",
        "completed_epochs": len(rows),
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "selection_split": "none",
        "test_selection_leakage": False,
        "method": "volmin",
    }
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in [*rows, final]),
        encoding="utf-8",
    )
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )
    return run_dir


__all__ = ["run_volmin_experiment"]
