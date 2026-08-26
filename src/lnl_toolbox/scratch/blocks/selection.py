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
    id="linear_rate_schedule",
    name="Linear Rate Schedule",
    category="Schedule",
    description="Interpolate a scalar rate between explicit endpoints during warm-up.",
    params={
        "epoch": {"type": "slot", "default": "epoch"},
        "start": {"type": "float", "default": 1.0},
        "end": {"type": "float", "default": 0.5},
        "warmup_epochs": {"type": "int", "default": 10, "min": 0},
        "save_as": {"type": "slot", "default": "keep_rate"},
    },
    requires=("epoch",),
    provides=("save_as",),
    placement=("epoch",), stage="train", ui_group="⑥ 样本选择",
    formula="r(t)=start+clip(t/T,0,1)(end-start)",
    formula_ref="shared linear keep-rate schedule",
)
def linear_rate_schedule(ctx: ScratchContext, epoch: str = "epoch", start: float = 1.0,
                         end: float = 0.5, warmup_epochs: int = 10,
                         save_as: str = "keep_rate") -> None:
    if int(warmup_epochs) <= 0:
        progress = 1.0
    else:
        progress = min(max(float(ctx[epoch]), 0.0) / int(warmup_epochs), 1.0)
    ctx[save_as] = float(start) + progress * (float(end) - float(start))


@block(
    id="select_lowest_scores",
    name="Select Lowest Scores",
    category="Sample Selection",
    description="Select stable local indices for the lowest score values.",
    params={
        "scores": {"type": "slot", "default": "loss_per_sample"},
        "keep_fraction": {"type": "slot", "default": "keep_rate"},
        "stable_sample_indices": {"type": "slot", "default": "indices"},
        "rounding": {"type": "enum", "options": ["floor", "ceil"], "default": "floor"},
        "minimum_count": {"type": "int", "default": 1, "min": 0},
        "detach": {"type": "bool", "default": True},
        "save_as": {"type": "slot", "default": "selected_indices"},
    },
    requires=("scores", "keep_fraction"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择",
    formula="S=arg lowest_k(score), k=rounding(N·fraction)",
    formula_ref="stable lowest-score selection",
)
def select_lowest_scores(ctx: ScratchContext, scores: str = "loss_per_sample",
                         keep_fraction: str = "keep_rate",
                         stable_sample_indices: str = "indices",
                         rounding: str = "floor", minimum_count: int = 1,
                         detach: bool = True, save_as: str = "selected_indices") -> None:
    torch = _torch()
    values = ctx[scores].reshape(-1)
    if bool(detach):
        values = values.detach()
    if values.numel() == 0:
        raise ValueError("cannot select from an empty score vector")
    fraction = float(ctx[keep_fraction])
    raw = values.numel() * fraction
    count = int(torch.ceil(torch.tensor(raw)).item()) if str(rounding) == "ceil" else int(torch.floor(torch.tensor(raw)).item())
    count = min(values.numel(), max(int(minimum_count), count))
    if stable_sample_indices in ctx:
        stable = ctx[stable_sample_indices].reshape(-1).to(values.device)
        order = torch.argsort(stable, stable=True)
        selected = order[torch.argsort(values[order], stable=True)[:count]]
    else:
        selected = torch.argsort(values, stable=True)[:count]
    ctx[save_as] = selected


@block(
    id="select_by_indices",
    name="Select By Indices",
    category="Sample Selection",
    description="Gather selected local values without reducing them.",
    params={"values": {"type": "slot", "default": "loss_per_sample"}, "indices": {"type": "slot", "default": "selected_indices"}, "save_as": {"type": "slot", "default": "selected_values"}},
    requires=("values", "indices"), provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择",
    formula="v_selected=v[indices]", formula_ref="indexed selection",
)
def select_by_indices(ctx: ScratchContext, values: str = "loss_per_sample", indices: str = "selected_indices", save_as: str = "selected_values") -> None:
    ctx[save_as] = ctx[values].reshape(-1)[ctx[indices]]


@block(
    id="indices_to_mask",
    name="Indices To Mask",
    category="Sample Selection",
    description="Convert local selected indices into a boolean mask for an explicit reference vector.",
    params={"indices": {"type": "slot", "default": "selected_indices"}, "reference": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "selected_mask"}},
    requires=("indices", "reference"), provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择",
    formula="m_j=1[j in indices]", formula_ref="explicit index-to-mask conversion",
)
def indices_to_mask(ctx: ScratchContext, indices: str = "selected_indices", reference: str = "loss_per_sample", save_as: str = "selected_mask") -> None:
    torch = _torch()
    values = ctx[reference].reshape(-1)
    mask = torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    mask[ctx[indices].to(values.device)] = True
    ctx[save_as] = mask


@block(
    id="mean_by_indices",
    name="Mean By Indices",
    category="Sample Selection",
    description="Reduce selected local values to their scalar mean.",
    params={"values": {"type": "slot", "default": "loss_per_sample"}, "indices": {"type": "slot", "default": "selected_indices"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("values", "indices"), provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择",
    formula="mean(v[indices])", formula_ref="indexed mean reduction",
)
def mean_by_indices(ctx: ScratchContext, values: str = "loss_per_sample", indices: str = "selected_indices", save_as: str = "loss") -> None:
    ctx[save_as] = ctx[values].reshape(-1)[ctx[indices]].mean()


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
