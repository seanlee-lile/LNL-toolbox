"""Semantic sample-selection blocks."""

from __future__ import annotations

from typing import Any

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Selection blocks require PyTorch; install the `train` extra.") from exc
    return torch


@block(
    id="select_all",
    name="Select All",
    category="Sample Selection",
    description="Select every sample in an input score vector.",
    params={"input": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "selected_indices"}},
    requires=("input",),
    provides=("save_as", "selected_mask"),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择", beginner_visible=False,
)
def select_all(ctx: ScratchContext, input: str = "loss_per_sample", save_as: str = "selected_indices") -> None:
    torch = _torch()
    values = ctx[input].reshape(-1)
    ctx[save_as] = torch.arange(values.numel(), device=values.device)
    ctx["selected_mask"] = torch.ones(values.numel(), dtype=torch.bool, device=values.device)


@block(
    id="small_loss",
    name="Small-Loss Selection",
    category="Sample Selection",
    description="Keep the lowest-loss examples and reduce them for Backward.",
    params={
        "input": {"type": "slot", "default": "loss_per_sample"},
        "keep_rate": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0},
        "save_as": {"type": "slot", "default": "loss"},
    },
    requires=("input",),
    provides=("save_as", "selected_indices", "selected_mask"),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择", beginner_visible=False,
    formula="选择逐样本损失最小的前 R(T) 比例", formula_ref="small-loss selection step",
)
def small_loss(ctx: ScratchContext, input: str = "loss_per_sample", keep_rate: float = 0.8, save_as: str = "loss") -> None:
    torch = _torch()
    values = ctx[input].reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot select from an empty loss vector")
    count = max(1, int(torch.ceil(torch.tensor(values.numel() * float(keep_rate))).item()))
    count = min(values.numel(), count)
    selected = torch.argsort(values, stable=True)[:count]
    mask = torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    mask[selected] = True
    ctx["selected_indices"] = selected
    ctx["selected_mask"] = mask
    ctx[save_as] = values[selected].mean()


@block(
    id="small_loss_indices",
    name="Co-teaching Small-loss Set",
    category="Sample Selection",
    description="Select the lowest-loss examples using the current Co-teaching remember rate.",
    params={
        "input": {"type": "slot", "default": "loss_per_sample"},
        "remember_rate": {"type": "slot", "default": "remember_rate"},
        "save_as": {"type": "slot", "default": "selected_indices"},
    },
    requires=("input", "remember_rate"),
    provides=("save_as", "selected_mask"),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择",
    formula="selected = lowest_loss(loss_per_sample, floor(R(T) × batch_size))",
    formula_ref="Co-teaching stable small-loss selection",
    paper="Co-teaching",
)
def small_loss_indices(
    ctx: ScratchContext,
    input: str = "loss_per_sample",
    remember_rate: str = "remember_rate",
    save_as: str = "selected_indices",
) -> None:
    torch = _torch()
    values = ctx[input].detach().reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot select from an empty loss vector")
    count = max(1, min(values.numel(), int(torch.floor(torch.tensor(values.numel() * float(ctx[remember_rate]))).item())))
    selected = torch.argsort(values, stable=True)[:count]
    mask = torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    mask[selected] = True
    ctx[save_as] = selected
    ctx["selected_mask"] = mask


@block(
    id="top_k_confidence",
    name="Top-k Confidence",
    category="Sample Selection",
    description="Select the highest confidence examples from a probability slot.",
    params={"input": {"type": "slot", "default": "probabilities"}, "keep_rate": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "selected_indices"}},
    requires=("input",),
    provides=("save_as", "selected_mask"),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择", beginner_visible=False,
)
def top_k_confidence(ctx: ScratchContext, input: str = "probabilities", keep_rate: float = 0.8, save_as: str = "selected_indices") -> None:
    torch = _torch()
    confidence = ctx[input].max(dim=-1).values.reshape(-1)
    count = max(1, min(confidence.numel(), int(torch.ceil(torch.tensor(confidence.numel() * float(keep_rate))).item())))
    selected = torch.argsort(confidence, descending=True, stable=True)[:count]
    mask = torch.zeros(confidence.numel(), dtype=torch.bool, device=confidence.device)
    mask[selected] = True
    ctx[save_as] = selected
    ctx["selected_mask"] = mask
