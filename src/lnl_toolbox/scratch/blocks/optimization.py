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
    # Recipes that discover feature width on the first batch may invoke this
    # construction block inside a bounded batch loop.  Reusing an explicitly
    # named slot keeps optimizer state (momentum/Adam moments) intact while
    # preserving the single-module construction contract.
    if save_as in ctx:
        return
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
    id="create_parameter_group_optimizer",
    name="Create Parameter-group Optimizer",
    category="Optimization",
    description="Construct one optimizer from explicit parameter/module slots and group specifications.",
    params={"optimizer_spec": {"type": "value", "default": {"name": "adam"}},
            "groups": {"type": "value", "default": []},
            "save_as": {"type": "slot", "default": "optimizer"}},
    requires=(), provides=("save_as",), placement=("top",), stage="setup", ui_group="② 初始化",
)
def create_parameter_group_optimizer(ctx: ScratchContext, optimizer_spec: Any | None = None,
                                     groups: Any = (), save_as: str = "optimizer") -> None:
    """Build an optimizer without creating or mutating the referenced modules.

    Each group is a serialisable mapping with either ``source`` (a Context
    slot containing a module/parameter iterable) or ``parameters`` (a slot
    containing parameters), plus optimizer options such as ``lr``.
    """
    torch = _torch()
    spec = dict(optimizer_spec or {}) if isinstance(optimizer_spec, dict) else {"name": str(optimizer_spec or "adam")}
    name = str(spec.pop("name", spec.pop("optimizer", "adam"))).lower()
    parameter_groups = []
    for raw in groups or ():
        if not isinstance(raw, dict):
            raise TypeError("optimizer groups must be mappings")
        group = dict(raw)
        source = group.pop("source", group.pop("parameters", None))
        if source is None:
            raise ValueError("each optimizer group requires a source slot")
        value = ctx[source] if isinstance(source, str) else source
        parameters = value.parameters() if hasattr(value, "parameters") else value
        group["params"] = parameters
        parameter_groups.append(group)
    if not parameter_groups:
        raise ValueError("create_parameter_group_optimizer requires at least one group")
    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                     "sgd": torch.optim.SGD}.get(name)
    if optimizer_cls is None:
        raise ValueError(f"unknown Scratch optimizer `{name}`")
    ctx[save_as] = optimizer_cls(parameter_groups, **spec)


class _ScratchModelEMA:
    """Small model-agnostic EMA container owned by Scratch."""

    def __init__(self, model: Any, momentum: float, update_buffers: bool = False) -> None:
        import copy
        self.model = copy.deepcopy(model)
        self.momentum = float(momentum)
        self.update_buffers = bool(update_buffers)
        self.model.eval()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def update(self, model: Any) -> None:
        torch = _torch()
        with torch.no_grad():
            for target, source in zip(self.model.parameters(), model.parameters()):
                target.mul_(self.momentum).add_(source.detach(), alpha=1.0 - self.momentum)
            if self.update_buffers:
                for target, source in zip(self.model.buffers(), model.buffers()):
                    target.copy_(source)


@block(
    id="create_model_ema",
    name="Create Model EMA",
    category="State",
    description="Create an independent exponential moving-average copy for one model slot.",
    params={"model": {"type": "slot", "default": "model"}, "momentum": {"type": "float", "default": 0.999, "min": 0.0, "max": 1.0}, "update_buffers": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "ema_model"}},
    requires=("model",), provides=("save_as",), placement=("top",), stage="setup", ui_group="④ 状态更新",
)
def create_model_ema(ctx: ScratchContext, model: str = "model", momentum: float = 0.999, update_buffers: bool = False, save_as: str = "ema_model") -> None:
    ctx[save_as] = _ScratchModelEMA(ctx[model], float(momentum), bool(update_buffers))


@block(
    id="update_model_ema",
    name="Update Model EMA",
    category="State",
    description="Update one model EMA from its corresponding live model; parameter and buffer mutation is explicit.",
    params={"model": {"type": "slot", "default": "model"}, "ema": {"type": "slot", "default": "ema_model"}},
    requires=("model", "ema"), provides=(), placement=("batch", "epoch"), stage="train", ui_group="④ 状态更新",
)
def update_model_ema(ctx: ScratchContext, model: str = "model", ema: str = "ema_model") -> None:
    value = ctx[ema]
    if not hasattr(value, "update") or not hasattr(value, "model"):
        raise TypeError("update_model_ema requires a Scratch model EMA container")
    value.update(ctx[model])


@block(
    id="set_optimizer_learning_rate",
    name="Set Optimizer Learning Rate",
    category="Optimization",
    description="Set the learning rate of every parameter group in an existing optimizer.",
    params={
        "optimizer": {"type": "slot", "default": "optimizer"},
        "learning_rate": {"type": "float", "required": True, "min": 0.0},
    },
    requires=("optimizer",),
    placement=("epoch",), stage="train", ui_group="⑩ 论文专用", beginner_visible=False,
)
def set_optimizer_learning_rate(
    ctx: ScratchContext,
    optimizer: str = "optimizer",
    learning_rate: float = 0.0,
) -> None:
    value = ctx[optimizer]
    if not hasattr(value, "param_groups"):
        raise TypeError("set_optimizer_learning_rate requires a torch optimizer")
    for group in value.param_groups:
        group["lr"] = float(learning_rate)


@block(
    id="create_scaled_scheduler",
    name="Create Scaled Scheduler",
    category="Optimization",
    description="Create a generic milestone scheduler whose learning rate can be scaled by an explicit value at step time.",
    params={"optimizer": {"type": "slot", "default": "optimizer"}, "milestones": {"type": "value", "default": [60]}, "gamma": {"type": "float", "default": 0.1, "min": 0.0}, "save_as": {"type": "slot", "default": "scheduler"}},
    requires=("optimizer",), provides=("save_as",), placement=("top",), stage="setup", ui_group="② 初始化", beginner_visible=False,
)
def create_scaled_scheduler(ctx: ScratchContext, optimizer: str = "optimizer", milestones: Any = (60,), gamma: float = 0.1, save_as: str = "scheduler") -> None:
    torch = _torch()
    # Keep this block generic: alpha is supplied at step time and only scales
    # the current learning rate.  No legacy experiment helper is imported.
    class _ScaledScheduler:
        def __init__(self, value):
            self.optimizer = value
            self.base = torch.optim.lr_scheduler.MultiStepLR(
                value, milestones=list(milestones), gamma=float(gamma))
            self.last_epoch = -1

        def step(self, confidence_weight: float = 0.0):
            self.base.step()
            self.last_epoch = self.base.last_epoch
            scale = 1.0 / (1.0 + max(float(confidence_weight), 0.0))
            for group, base_lr in zip(self.optimizer.param_groups, self.base.get_last_lr()):
                group["lr"] = base_lr * scale

        def state_dict(self):
            return {"base": self.base.state_dict()}

        def load_state_dict(self, state):
            self.base.load_state_dict(state["base"])
    ctx[save_as] = _ScaledScheduler(ctx[optimizer])


@block(
    id="scaled_scheduler_step",
    name="Scaled Scheduler Step",
    category="Optimization",
    description="Advance a scaled scheduler using an explicit nonnegative scaling value.",
    params={"scheduler": {"type": "slot", "default": "scheduler"}, "scale_value": {"type": "slot", "default": "confidence_weight"}},
    requires=("scheduler",), placement=("epoch",), stage="train", ui_group="⑩ 论文专用", beginner_visible=False,
)
def scaled_scheduler_step(ctx: ScratchContext, scheduler: str = "scheduler", scale_value: str = "confidence_weight") -> None:
    if scale_value not in ctx:
        raise ValueError(f"scaled_scheduler_step requires scale slot `{scale_value}`")
    ctx[scheduler].step(float(ctx[scale_value]))


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


@block(
    id="step_milestone_update",
    name="Step-Milestone Learning-Rate Update",
    category="Optimization",
    description="Apply optimizer learning-rate drops at explicit global-step milestones.",
    params={
        "optimizer": {"type": "slot", "default": "optimizer"},
        "milestones": {"type": "value", "default": [19500, 25000, 30000]},
        "gamma": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
        "steps_per_epoch": {"type": "int", "default": 391, "min": 1},
        "global_step": {"type": "slot", "default": "global_step"},
    },
    requires=("optimizer",), placement=("batch",), stage="train", ui_group="⑧ 反向传播与更新", beginner_visible=False,
    formula="lr_t=lr_0 gamma^{|{m: m<=t}|}, t=epoch*S+batch+1",
    formula_ref="step-milestone learning-rate schedule",
)
def step_milestone_update(
    ctx: ScratchContext,
    optimizer: str = "optimizer",
    milestones: Any = (19500, 25000, 30000),
    gamma: float = 0.1,
    steps_per_epoch: int = 391,
    global_step: str = "global_step",
) -> None:
    step = int(ctx[global_step]) + 1 if global_step in ctx else int(ctx.get("epoch", 0)) * int(steps_per_epoch) + int(ctx.get("batch_idx", 0)) + 1
    count = sum(step >= int(milestone) for milestone in milestones)
    optimizer_value = ctx[optimizer]
    if not hasattr(optimizer_value, "param_groups"):
        raise TypeError("step-milestone update requires a torch optimizer")
    for group in optimizer_value.param_groups:
        base = group.setdefault("_scratch_base_lr", float(group["lr"]))
        group["lr"] = float(base) * float(gamma) ** count
