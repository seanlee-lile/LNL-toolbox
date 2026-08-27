"""Shared scalar schedules."""

from __future__ import annotations

from typing import Any

from ..context import ScratchContext
from ..registry import block


@block(
    id="piecewise_rate_schedule",
    name="Piecewise Rate Schedule",
    category="Schedule",
    description="Resolve a scalar by explicit epoch milestones and values.",
    params={"epoch": {"type": "slot", "default": "epoch"}, "default": {"type": "float", "default": 1.0}, "milestones": {"type": "value", "default": []}, "values": {"type": "value", "default": []}, "epoch_offset": {"type": "int", "default": 0, "min": 0}, "save_as": {"type": "slot", "default": "rate"}},
    requires=("epoch",), provides=("save_as",), placement=("epoch",), stage="train", ui_group="⑥ 样本选择",
)
def piecewise_rate_schedule(ctx: ScratchContext, epoch: str = "epoch", default: float = 1.0, milestones: Any = (), values: Any = (), epoch_offset: int = 0, save_as: str = "rate") -> None:
    points = list(milestones); outputs = list(values)
    if len(points) != len(outputs):
        raise ValueError("milestones and values must have equal length")
    current = int(ctx.get(epoch, 0)) + int(epoch_offset)
    result = float(default)
    for milestone, value in zip(points, outputs):
        if current >= int(milestone):
            result = float(value)
    ctx[save_as] = result
