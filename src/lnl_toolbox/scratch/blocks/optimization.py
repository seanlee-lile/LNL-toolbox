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
        "nesterov": {"type": "bool", "default": False},
        "weight_decay": {"type": "float", "default": 0.0, "min": 0.0},
        "save_as": {"type": "slot", "default": "optimizer"},
    },
    requires=("model",),
    provides=("save_as",),
    placement=("top",), stage="setup", ui_group="② 初始化",
)
def create_optimizer(
    ctx: ScratchContext,
    optimizer: str = "sgd",
    model: str = "model",
    lr: float = 0.1,
    momentum: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    save_as: str = "optimizer",
) -> None:
    torch = _torch()
    name = str(optimizer).strip().lower()
    parameters = ctx[model].parameters()
    if name == "sgd":
        value = torch.optim.SGD(
            parameters,
            lr=float(lr),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            weight_decay=float(weight_decay),
        )
    elif name == "adam":
        value = torch.optim.Adam(parameters, lr=float(lr), weight_decay=float(weight_decay))
    else:
        raise ValueError(f"unknown Scratch optimizer `{optimizer}`")
    ctx[save_as] = value


@block(
    id="create_scheduler",
    name="Create MultiStep Scheduler",
    category="Optimization",
    description="Create the paper's epoch-level MultiStepLR schedule.",
    params={
        "scheduler": {"type": "enum", "options": ["multistep"], "default": "multistep"},
        "optimizer": {"type": "slot", "default": "optimizer"},
        "milestones": {"type": "value", "default": [40, 80]},
        "gamma": {"type": "float", "default": 0.1, "min": 0.0},
        "save_as": {"type": "slot", "default": "scheduler"},
    },
    requires=("optimizer",),
    provides=("save_as",),
    placement=("top",), stage="setup", ui_group="② 初始化", beginner_visible=False,
)
def create_scheduler(
    ctx: ScratchContext,
    scheduler: str = "multistep",
    optimizer: str = "optimizer",
    milestones: list[int] | tuple[int, ...] = (40, 80),
    gamma: float = 0.1,
    save_as: str = "scheduler",
) -> None:
    torch = _torch()
    if str(scheduler).lower() != "multistep":
        raise ValueError(f"unknown Scratch scheduler `{scheduler}`")
    if isinstance(milestones, str):
        values = [int(value.strip()) for value in milestones.split(",") if value.strip()]
    else:
        values = [int(value) for value in milestones]
    ctx[save_as] = torch.optim.lr_scheduler.MultiStepLR(
        ctx[optimizer], milestones=values, gamma=float(gamma)
    )


@block(
    id="zero_grad",
    name="Zero Grad",
    category="Optimization",
    description="Clear gradients before a new update.",
    params={"optimizer": {"type": "slot", "default": "optimizer"}},
    requires=("optimizer",),
    placement=("batch",), stage="train", ui_group="⑧ 反向传播与更新",
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
    placement=("batch",), stage="train", ui_group="⑧ 反向传播与更新",
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
    placement=("batch",), stage="train", ui_group="⑧ 反向传播与更新",
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
    placement=("epoch", "batch"), stage="train", ui_group="⑧ 反向传播与更新", beginner_visible=False,
)
def scheduler_step(ctx: ScratchContext, scheduler: str = "scheduler") -> None:
    ctx[scheduler].step()
