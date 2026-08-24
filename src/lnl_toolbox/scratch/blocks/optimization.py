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
    id="create_joint_optimizer",
    name="Create Joint Optimizer",
    category="Optimization",
    description="Create one optimizer over two peer model parameter sets.",
    params={
        "optimizer": {"type": "enum", "options": ["adam", "sgd"], "default": "adam"},
        "model_a": {"type": "slot", "default": "model_a"},
        "model_b": {"type": "slot", "default": "model_b"},
        "lr": {"type": "float", "default": 0.001, "min": 0.0},
        "weight_decay": {"type": "float", "default": 0.0, "min": 0.0},
        "beta1": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0},
        "beta2": {"type": "float", "default": 0.999, "min": 0.0, "max": 1.0},
        "save_as": {"type": "slot", "default": "optimizer"},
    },
    requires=("model_a", "model_b"),
    provides=("save_as",),
    placement=("top",), stage="setup", ui_group="② 初始化",
)
def create_joint_optimizer(
    ctx: ScratchContext,
    optimizer: str = "adam",
    model_a: str = "model_a",
    model_b: str = "model_b",
    lr: float = 0.001,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    save_as: str = "optimizer",
) -> None:
    torch = _torch()
    parameters = list(ctx[model_a].parameters()) + list(ctx[model_b].parameters())
    name = str(optimizer).strip().lower()
    if name == "adam":
        value = torch.optim.Adam(parameters, lr=float(lr), betas=(float(beta1), float(beta2)), weight_decay=float(weight_decay))
    elif name == "sgd":
        value = torch.optim.SGD(parameters, lr=float(lr), weight_decay=float(weight_decay))
    else:
        raise ValueError(f"unknown Scratch joint optimizer `{optimizer}`")
    ctx[save_as] = value


class _LinearDecayBetaScheduler:
    def __init__(self, optimizer, start_epoch, end_epoch, initial_lr, final_lr, beta1_before, beta1_after):
        self.optimizer = optimizer
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.initial_lr = float(initial_lr)
        self.final_lr = float(final_lr)
        self.beta1_before = float(beta1_before)
        self.beta1_after = float(beta1_after)
        self.last_epoch = -1

    def step(self):
        self.last_epoch += 1
        progress = min(max((self.last_epoch - self.start_epoch) / max(self.end_epoch - self.start_epoch, 1), 0.0), 1.0)
        lr = self.initial_lr + progress * (self.final_lr - self.initial_lr)
        beta1 = self.beta1_before if self.last_epoch < self.start_epoch else self.beta1_after
        for group in self.optimizer.param_groups:
            group["lr"] = lr
            if "betas" in group:
                group["betas"] = (beta1, float(group["betas"][1]))


@block(
    id="create_scheduler",
    name="Create Scheduler",
    category="Optimization",
    description="Create an epoch-level MultiStepLR or CosineAnnealingLR schedule.",
    params={
        "scheduler": {"type": "enum", "options": ["multistep", "cosine", "linear_decay"], "default": "multistep"},
        "optimizer": {"type": "slot", "default": "optimizer"},
        "milestones": {"type": "value", "default": [40, 80]},
        "gamma": {"type": "float", "default": 0.1, "min": 0.0},
        "t_max": {"type": "int", "default": 120, "min": 1},
        "eta_min": {"type": "float", "default": 0.0, "min": 0.0},
        "start_epoch": {"type": "int", "default": 80, "min": 0},
        "end_epoch": {"type": "int", "default": 200, "min": 1},
        "initial_lr": {"type": "float", "default": 0.001, "min": 0.0},
        "final_lr": {"type": "float", "default": 0.0, "min": 0.0},
        "beta1_before": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0},
        "beta1_after": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
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
    t_max: int = 120,
    eta_min: float = 0.0,
    start_epoch: int = 80,
    end_epoch: int = 200,
    initial_lr: float = 0.001,
    final_lr: float = 0.0,
    beta1_before: float = 0.9,
    beta1_after: float = 0.1,
    save_as: str = "scheduler",
) -> None:
    torch = _torch()
    name = str(scheduler).lower()
    if name == "cosine":
        ctx[save_as] = torch.optim.lr_scheduler.CosineAnnealingLR(
            ctx[optimizer], T_max=int(t_max), eta_min=float(eta_min)
        )
        return
    if name != "multistep":
        if name == "linear_decay":
            ctx[save_as] = _LinearDecayBetaScheduler(ctx[optimizer], start_epoch, end_epoch, initial_lr, final_lr, beta1_before, beta1_after)
            return
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
    id="clip_grad_norm",
    name="Clip Gradient Norm",
    category="Optimization",
    description="Clip model gradients to the configured maximum norm before the optimizer update.",
    params={
        "model": {"type": "slot", "default": "model"},
        "max_norm": {"type": "float", "default": 5.0, "min": 0.0},
    },
    requires=("model",),
    placement=("batch",), stage="train", ui_group="⑧ 反向传播与更新",
)
def clip_grad_norm(ctx: ScratchContext, model: str = "model", max_norm: float = 5.0) -> None:
    torch = _torch()
    if float(max_norm) <= 0.0:
        raise ValueError("max_norm must be positive")
    torch.nn.utils.clip_grad_norm_(ctx[model].parameters(), max_norm=float(max_norm))


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
