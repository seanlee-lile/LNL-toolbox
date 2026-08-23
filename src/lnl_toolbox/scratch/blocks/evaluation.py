"""Evaluation and lightweight run-artifact blocks."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Evaluation blocks require PyTorch; install the `train` extra.") from exc
    return torch, F


def _batch(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, dict):
        return batch.get("images", batch.get("inputs", batch.get("input", batch.get("x")))), batch.get("labels", batch.get("targets", batch.get("target", batch.get("y"))))
    return batch[0], batch[1]


@block(
    id="evaluate_accuracy",
    name="Evaluate Accuracy",
    category="Evaluation",
    description="Evaluate a model on a loader and append an accuracy metric.",
    params={"model": {"type": "slot", "default": "model"}, "loader": {"type": "slot", "default": "test_loader"}, "save_as": {"type": "slot", "default": "accuracy"}},
    requires=("model", "loader"),
    provides=("save_as", "metrics"),
    placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
)
def evaluate_accuracy(ctx: ScratchContext, model: str = "model", loader: str = "test_loader", save_as: str = "accuracy") -> None:
    torch, _ = _torch()
    network = ctx[model]
    was_training = network.training
    network.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in ctx[loader]:
            inputs, labels = _batch(batch)
            predictions = network(inputs).argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    if was_training:
        network.train()
    value = float(correct / total) if total else 0.0
    ctx[save_as] = value
    ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "accuracy": value})


@block(
    id="track_best_model",
    name="Keep Best Validation Model",
    category="Evaluation",
    description="Keep the model state from the epoch with the highest validation accuracy.",
    params={
        "model": {"type": "slot", "default": "model"},
        "metric": {"type": "slot", "default": "validation_accuracy"},
        "save_as": {"type": "slot", "default": "best_model_state"},
    },
    requires=("model", "metric"),
    provides=("save_as", "best_epoch"),
    placement=("epoch",), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def track_best_model(
    ctx: ScratchContext,
    model: str = "model",
    metric: str = "validation_accuracy",
    save_as: str = "best_model_state",
) -> None:
    value = float(ctx[metric])
    if value > float(ctx.get("best_validation_accuracy", float("-inf"))):
        ctx["best_validation_accuracy"] = value
        ctx["best_epoch"] = int(ctx.get("epoch", 0))
        ctx[save_as] = deepcopy(ctx[model].state_dict())


@block(
    id="restore_best_model",
    name="Restore Best Validation Model",
    category="Evaluation",
    description="Restore the checkpoint selected by validation accuracy before final test evaluation.",
    params={
        "model": {"type": "slot", "default": "model"},
        "state": {"type": "slot", "default": "best_model_state"},
    },
    requires=("model", "state"),
    placement=("top",), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def restore_best_model(ctx: ScratchContext, model: str = "model", state: str = "best_model_state") -> None:
    ctx[model].load_state_dict(ctx[state])


@block(
    id="record_metrics",
    name="Record Metrics",
    category="Evaluation",
    description="Record selected Context scalar values in an in-memory metrics list.",
    params={"values": {"type": "value", "default": ["loss", "accuracy"]}},
    provides=("metrics",),
    placement=("top", "epoch", "batch"), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def record_metrics(ctx: ScratchContext, values: list[str] | tuple[str, ...] = ("loss", "accuracy")) -> None:
    row = {"epoch": int(ctx.get("epoch", 0))}
    for name in values:
        if name in ctx and isinstance(ctx[name], (int, float)):
            row[name] = float(ctx[name])
        elif name in ctx and hasattr(ctx[name], "item"):
            row[name] = float(ctx[name].item())
    ctx.setdefault("metrics", []).append(row)


@block(
    id="save_checkpoint",
    name="Save Checkpoint",
    category="Evaluation",
    description="Save a model state dictionary below the current artifact directory.",
    params={"model": {"type": "slot", "default": "model"}, "path": {"type": "str", "default": "checkpoints/model.pt"}},
    requires=("model",),
    placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def save_checkpoint(ctx: ScratchContext, model: str = "model", path: str = "checkpoints/model.pt") -> None:
    torch, _ = _torch()
    root = Path(str(ctx.get("artifact_dir", ".")))
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ctx[model].state_dict(), destination)


@block(
    id="write_metrics",
    name="Write Metrics",
    category="Evaluation",
    description="Write in-memory metrics as JSON Lines below the artifact directory.",
    params={"path": {"type": "str", "default": "metrics.jsonl"}},
    requires=("metrics",),
    placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def write_metrics(ctx: ScratchContext, path: str = "metrics.jsonl") -> None:
    destination = Path(str(ctx.get("artifact_dir", "."))) / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ctx["metrics"]), encoding="utf-8")
