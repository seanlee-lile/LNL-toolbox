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
    for epoch in range(start_epoch, start_epoch + epochs):
        ctx["epoch"] = epoch
        execute(children, ctx)


@block(
    id="batch_loop",
    name="Batch Loop",
    category="Control",
    description="Iterate a loader; use Get Batch inside for explicit slot assignment.",
    kind="loop",
    params={"loader": {"type": "slot", "default": "train_loader"}},
    requires=("loader",),
    provides=("batch_idx", "batch"),
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
    for batch_idx, batch in enumerate(loader):
        if limit is not None and batch_idx >= int(limit):
            break
        ctx["batch_idx"] = batch_idx
        ctx["batch"] = batch
        execute(children, ctx)
