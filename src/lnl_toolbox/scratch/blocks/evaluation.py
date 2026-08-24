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
    params={"model": {"type": "slot", "default": "model"}, "loader": {"type": "slot", "default": "test_loader"}, "save_as": {"type": "slot", "default": "accuracy"}, "final": {"type": "bool", "default": False}},
    requires=("model", "loader"),
    provides=("save_as", "metrics"),
    placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
)
def evaluate_accuracy(ctx: ScratchContext, model: str = "model", loader: str = "test_loader", save_as: str = "accuracy", final: bool = False) -> None:
    torch, _ = _torch()
    if bool(final) and bool(ctx.get("_runtime_limits", {}).get("skip_final_test")) and str(loader) == "test_loader":
        ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "test_skipped": True})
        return
    network = ctx[model]
    was_training = network.training
    network.eval()
    device = next(network.parameters()).device
    correct = total = 0
    max_batches = ctx.get("_runtime_limits", {}).get("max_batches")
    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx[loader]):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            inputs, labels = _batch(batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = network(inputs).argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    if was_training:
        network.train()
    value = float(correct / total) if total else 0.0
    ctx[save_as] = value
    ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "accuracy": value})


@block(
    id="evaluate_cwd_binary",
    name="Evaluate CWD Binary Scalar",
    category="Evaluation",
    description="Evaluate a one-output CWD classifier by the paper's zero-margin binary decision rule.",
    params={"model": {"type": "slot", "default": "model"}, "loader": {"type": "slot", "default": "test_loader"}, "save_as": {"type": "slot", "default": "accuracy"}, "final": {"type": "bool", "default": False}},
    requires=("model", "loader"), provides=("save_as", "metrics"), placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
    formula="y_hat = 1[m(x) >= 0]", formula_ref="CWD binary scalar evaluation", paper="Class-Wise Denoising",
)
def evaluate_cwd_binary(ctx: ScratchContext, model: str = "model", loader: str = "test_loader", save_as: str = "accuracy", final: bool = False) -> None:
    torch, _ = _torch()
    if bool(final) and bool(ctx.get("_runtime_limits", {}).get("skip_final_test")) and str(loader) == "test_loader":
        ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "test_skipped": True})
        return
    network = ctx[model]
    was_training = network.training
    network.eval()
    device = next(network.parameters()).device
    correct = total = 0
    max_batches = ctx.get("_runtime_limits", {}).get("max_batches")
    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx[loader]):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            inputs, labels = _batch(batch)
            logits = network(inputs.to(device))
            predictions = (logits[:, 0] >= 0).long()
            labels = labels.to(device).long()
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    if was_training:
        network.train()
    value = float(correct / total) if total else 0.0
    ctx[save_as] = value
    ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "cwd_accuracy": value})


@block(
    id="track_best_model",
    name="Keep Best Model",
    category="Evaluation",
    description="Keep the model state from the epoch with the highest configured selection metric.",
    params={
        "model": {"type": "slot", "default": "model"},
        "metric": {"type": "slot", "default": "validation_accuracy"},
        "metric_name": {"type": "str", "default": "validation_accuracy"},
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
    metric_name: str = "validation_accuracy",
    save_as: str = "best_model_state",
) -> None:
    value = float(ctx[metric])
    best_key = f"best_{str(metric_name).strip()}"
    if value > float(ctx.get(best_key, float("-inf"))):
        ctx[best_key] = value
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
    id="evaluate_peer_ensemble",
    name="Evaluate Peer Ensemble",
    category="Evaluation",
    description="Evaluate two peer models and their mean-probability ensemble on a loader.",
    params={"model_a": {"type": "slot", "default": "model_a"}, "model_b": {"type": "slot", "default": "model_b"}, "loader": {"type": "slot", "default": "validation_loader"}, "save_as": {"type": "slot", "default": "ensemble_accuracy"}, "final": {"type": "bool", "default": False}},
    requires=("model_a", "model_b", "loader"),
    provides=("save_as", "peer_accuracy_a", "peer_accuracy_b", "metrics"),
    placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def evaluate_peer_ensemble(ctx: ScratchContext, model_a: str = "model_a", model_b: str = "model_b", loader: str = "validation_loader", save_as: str = "ensemble_accuracy", final: bool = False) -> None:
    torch, _ = _torch()
    if bool(final) and bool(ctx.get("_runtime_limits", {}).get("skip_final_test")) and str(loader) == "test_loader":
        ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "test_skipped": True})
        return
    network_a, network_b = ctx[model_a], ctx[model_b]
    was_training = (network_a.training, network_b.training)
    network_a.eval(); network_b.eval()
    device = next(network_a.parameters()).device
    correct_a = correct_b = correct_ensemble = total = 0
    max_batches = ctx.get("_runtime_limits", {}).get("max_batches")
    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx[loader]):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            inputs, labels = _batch(batch)
            inputs, labels = inputs.to(device), labels.to(device)
            logits_a, logits_b = network_a(inputs), network_b(inputs)
            ensemble = (torch.softmax(logits_a, dim=-1) + torch.softmax(logits_b, dim=-1)) / 2.0
            correct_a += int((logits_a.argmax(-1) == labels).sum().item())
            correct_b += int((logits_b.argmax(-1) == labels).sum().item())
            correct_ensemble += int((ensemble.argmax(-1) == labels).sum().item())
            total += int(labels.numel())
    if was_training[0]: network_a.train()
    if was_training[1]: network_b.train()
    denominator = float(total) if total else 1.0
    ctx["peer_accuracy_a"] = correct_a / denominator
    ctx["peer_accuracy_b"] = correct_b / denominator
    ctx[save_as] = correct_ensemble / denominator
    ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "peer_accuracy_a": ctx["peer_accuracy_a"], "peer_accuracy_b": ctx["peer_accuracy_b"], "ensemble_accuracy": ctx[save_as]})


@block(
    id="track_best_peer_models",
    name="Keep Best Peer Ensemble",
    category="Evaluation",
    description="Keep both peer states from the epoch with the highest ensemble validation accuracy.",
    params={"model_a": {"type": "slot", "default": "model_a"}, "model_b": {"type": "slot", "default": "model_b"}, "metric": {"type": "slot", "default": "ensemble_accuracy"}, "save_as": {"type": "slot", "default": "best_peer_model_state"}},
    requires=("model_a", "model_b", "metric"),
    provides=("save_as", "best_epoch"),
    placement=("epoch",), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def track_best_peer_models(ctx: ScratchContext, model_a: str = "model_a", model_b: str = "model_b", metric: str = "ensemble_accuracy", save_as: str = "best_peer_model_state") -> None:
    value = float(ctx[metric])
    if value > float(ctx.get("best_ensemble_accuracy", float("-inf"))):
        ctx["best_ensemble_accuracy"] = value
        ctx["best_epoch"] = int(ctx.get("epoch", 0))
        ctx[save_as] = {"a": deepcopy(ctx[model_a].state_dict()), "b": deepcopy(ctx[model_b].state_dict())}


@block(
    id="restore_best_peer_models",
    name="Restore Best Peer Ensemble",
    category="Evaluation",
    description="Restore both peer checkpoints selected by ensemble validation accuracy.",
    params={"model_a": {"type": "slot", "default": "model_a"}, "model_b": {"type": "slot", "default": "model_b"}, "state": {"type": "slot", "default": "best_peer_model_state"}},
    requires=("model_a", "model_b", "state"),
    placement=("top",), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def restore_best_peer_models(ctx: ScratchContext, model_a: str = "model_a", model_b: str = "model_b", state: str = "best_peer_model_state") -> None:
    ctx[model_a].load_state_dict(ctx[state]["a"])
    ctx[model_b].load_state_dict(ctx[state]["b"])


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
