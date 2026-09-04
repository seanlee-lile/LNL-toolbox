"""Minimal action, loop, and condition blocks for the Scratch interpreter."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..context import ScratchContext
from ..registry import block


@block(
    id="set_value",
    name="Set Value",
    category="Runtime",
    description="Store a literal value in a named context slot.",
    params={
        "value": {"type": "value", "required": True},
        "save_as": {"type": "slot", "default": "value"},
    },
    provides=("save_as",),
    placement=("any",), stage="setup", ui_group="② 初始化", beginner_visible=False,
)
def set_value(ctx: ScratchContext, value: Any, save_as: str = "value") -> None:
    ctx[save_as] = value


@block(
    id="repeat_n",
    name="Repeat N",
    category="Control",
    description="Execute child blocks a fixed number of times.",
    kind="loop",
    params={
        "count": {"type": "int", "required": True, "min": 0},
        "index_as": {"type": "slot", "default": "repeat_index"},
    },
    provides=("index_as",),
    placement=("top",), stage="train", ui_group="③ 训练结构", beginner_visible=False,
)
def repeat_n(
    ctx: ScratchContext,
    *,
    params: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    execute: Callable[..., ScratchContext],
) -> None:
    for index in range(int(params["count"])):
        ctx[str(params["index_as"])] = index
        execute(children, ctx)


@block(
    id="if_context",
    name="If Context Flag",
    category="Control",
    description="Execute child blocks when a named context value is truthy.",
    kind="condition",
    params={
        "flag": {"type": "slot", "required": True},
    },
    requires=("flag",),
    placement=("any",), stage="train", ui_group="③ 训练结构", beginner_visible=False,
)
def if_context(
    ctx: ScratchContext,
    *,
    params: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    execute: Callable[..., ScratchContext],
) -> None:
    if bool(ctx[str(params["flag"])]):
        execute(children, ctx)


@block(
    id="if_epoch_ge",
    name="If Epoch >=",
    category="Control",
    description="Execute child blocks when the current epoch reaches a threshold.",
    kind="condition",
    params={"epoch": {"type": "int", "required": True, "min": 0}},
    requires=(), provides=("loss",),
    placement=("epoch",), stage="train", ui_group="③ 训练结构", beginner_visible=False,
)
def if_epoch_ge(
    ctx: ScratchContext,
    *,
    params: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    execute: Callable[..., ScratchContext],
) -> None:
    if int(ctx["epoch"]) >= int(params["epoch"]):
        execute(children, ctx)


@block(
    id="if_epoch_lt",
    name="If Epoch <",
    category="Control",
    description="Execute child blocks while the current epoch is below a threshold.",
    kind="condition",
    params={"epoch": {"type": "int", "required": True, "min": 0}},
    requires=(), provides=("loss",),
    placement=("epoch",), stage="train", ui_group="③ 训练结构", beginner_visible=False,
)
def if_epoch_lt(
    ctx: ScratchContext,
    *,
    params: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    execute: Callable[..., ScratchContext],
) -> None:
    if int(ctx["epoch"]) < int(params["epoch"]):
        execute(children, ctx)


@block(
    id="if_epoch_eq",
    name="If Epoch =",
    category="Control",
    description="Execute child blocks exactly at one training epoch.",
    kind="condition",
    params={"epoch": {"type": "int", "required": True, "min": 0}},
    requires=(),
    placement=("epoch",), stage="train", ui_group="③ 训练结构", beginner_visible=False,
)
def if_epoch_eq(
    ctx: ScratchContext,
    *,
    params: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    execute: Callable[..., ScratchContext],
) -> None:
    if int(ctx["epoch"]) == int(params["epoch"]):
        execute(children, ctx)


@block(
    id="if_epoch_periodic",
    name="If Epoch Is Periodic",
    category="Control",
    description="Execute child blocks at a fixed interval from an inclusive start epoch.",
    kind="condition",
    params={"start": {"type": "int", "default": 0, "min": 0}, "interval": {"type": "int", "default": 1, "min": 1}},
    requires=(), provides=(),
    placement=("epoch",), stage="train", ui_group="③ 训练结构", beginner_visible=False,
)
def if_epoch_periodic(
    ctx: ScratchContext,
    *,
    params: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    execute: Callable[..., ScratchContext],
) -> None:
    epoch = int(ctx["epoch"])
    start = int(params.get("start", 0))
    interval = int(params.get("interval", 1))
    if epoch >= start and (epoch - start) % interval == 0:
        execute(children, ctx)


@block(
    id="epoch_loop",
    name="Epoch Loop",
    category="Control",
    description="Run child blocks once for each training epoch.",
    kind="loop",
    params={
        "epochs": {"type": "int", "required": True, "min": 0},
        "start_epoch": {"type": "int", "default": 0, "min": 0},
    },
    provides=("epoch",),
    placement=("top",), stage="train", ui_group="③ 训练结构",
)
def epoch_loop(
    ctx: ScratchContext,
    *,
    params: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    execute: Callable[..., ScratchContext],
) -> None:
    start_epoch = int(params["start_epoch"])
    epochs = int(params["epochs"])
    limit = ctx.get("_runtime_limits", {}).get("max_epochs")
    if limit is not None:
        epochs = min(epochs, int(limit))
    ctx["_progress_total_epochs"] = epochs
    ctx["_progress_start_epoch"] = start_epoch
    for epoch in range(start_epoch, start_epoch + epochs):
        ctx["epoch"] = epoch
        execute(children, ctx)
        # Publish a compact scalar snapshot after every completed epoch.  The
        # executor owns the observation channel; importing lazily here keeps
        # the control block independent during registry initialisation.
        try:
            from ..executor import publish_epoch_output

            publish_epoch_output(ctx)
        except Exception:
            # Progress is best-effort and must never change recipe semantics.
            pass


@block(
    id="batch_loop",
    name="Batch Loop",
    category="Control",
    description="Iterate a loader; use Get Batch inside for explicit slot assignment.",
    kind="loop",
    params={
        "loader": {"type": "slot", "default": "train_loader"},
        "max_steps": {"type": "int", "default": 0, "min": 0},
        "global_step_as": {"type": "slot", "default": "global_step"},
    },
    requires=("loader",),
    provides=("batch_idx", "batch", "global_step_as"),
    placement=("epoch",), stage="train", ui_group="③ 训练结构",
)
def batch_loop(
    ctx: ScratchContext,
    *,
    params: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    execute: Callable[..., ScratchContext],
) -> None:
    loader = ctx[str(params["loader"])]
    limit = ctx.get("_runtime_limits", {}).get("max_batches")
    max_steps = int(params.get("max_steps", 0))
    global_step_as = str(params.get("global_step_as", "global_step"))
    if global_step_as not in ctx:
        ctx[global_step_as] = 0
    try:
        total_batches = len(loader)
        if limit is not None:
            total_batches = min(total_batches, max(0, int(limit)))
        if max_steps:
            remaining = max(0, max_steps - int(ctx[global_step_as]))
            total_batches = min(total_batches, remaining)
        ctx["_progress_total_batches"] = total_batches
    except (TypeError, AttributeError):
        ctx.pop("_progress_total_batches", None)
    for batch_idx, batch in enumerate(loader):
        if limit is not None and batch_idx >= int(limit):
            break
        if max_steps and int(ctx[global_step_as]) >= max_steps:
            break
        ctx["batch_idx"] = batch_idx
        ctx["batch"] = batch
        execute(children, ctx)
        ctx[global_step_as] = int(ctx[global_step_as]) + 1
