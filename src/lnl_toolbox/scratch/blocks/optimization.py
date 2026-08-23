"""Optimizer and training-update blocks."""

from __future__ import annotations

from typing import Any

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Optimization blocks require PyTorch; install the `train` extra.") from exc
    return torch


@block(
    id="create_optimizer",
    name="Create Optimizer",
    category="Optimization",
    description="Create SGD or Adam for a model slot.",
    params={
        "optimizer": {"type": "str", "default": "sgd"},
        "model": {"type": "slot", "default": "model"},
        "lr": {"type": "float", "default": 0.1, "min": 0.0},
        "momentum": {"type": "float", "default": 0.9, "min": 0.0},
        "weight_decay": {"type": "float", "default": 0.0, "min": 0.0},
        "save_as": {"type": "slot", "default": "optimizer"},
    },
    requires=("model",),
    provides=("save_as",),
)
def create_optimizer(
    ctx: ScratchContext,
    optimizer: str = "sgd",
    model: str = "model",
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    save_as: str = "optimizer",
) -> None:
    torch = _torch()
    name = str(optimizer).strip().lower()
    parameters = ctx[model].parameters()
    if name == "sgd":
        value = torch.optim.SGD(parameters, lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay))
    elif name == "adam":
        value = torch.optim.Adam(parameters, lr=float(lr), weight_decay=float(weight_decay))
    else:
        raise ValueError(f"unknown Scratch optimizer `{optimizer}`")
    ctx[save_as] = value


@block(
    id="zero_grad",
    name="Zero Grad",
    category="Optimization",
    description="Clear gradients before a new update.",
    params={"optimizer": {"type": "slot", "default": "optimizer"}},
    requires=("optimizer",),
)
def zero_grad(ctx: ScratchContext, optimizer: str = "optimizer") -> None:
    ctx[optimizer].zero_grad()


@block(
    id="backward",
    name="Backward",
    category="Optimization",
    description="Backpropagate the scalar loss.",
    params={"loss": {"type": "slot", "default": "loss"}},
    requires=("loss",),
)
def backward(ctx: ScratchContext, loss: str = "loss") -> None:
    ctx[loss].backward()


@block(
    id="optimizer_step",
    name="Optimizer Step",
    category="Optimization",
    description="Apply gradients with an optimizer.",
    params={"optimizer": {"type": "slot", "default": "optimizer"}},
    requires=("optimizer",),
)
def optimizer_step(ctx: ScratchContext, optimizer: str = "optimizer") -> None:
    ctx[optimizer].step()


@block(
    id="scheduler_step",
    name="Scheduler Step",
    category="Optimization",
    description="Advance a scheduler slot.",
    params={"scheduler": {"type": "slot", "default": "scheduler"}},
    requires=("scheduler",),
)
def scheduler_step(ctx: ScratchContext, scheduler: str = "scheduler") -> None:
    ctx[scheduler].step()
