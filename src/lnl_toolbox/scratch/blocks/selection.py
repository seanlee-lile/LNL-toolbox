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
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择", beginner_visible=False,
)
def select_all(ctx: ScratchContext, input: str = "loss_per_sample", save_as: str = "selected_indices") -> None:
    torch = _torch()
    values = ctx[input].reshape(-1)
    ctx[save_as] = torch.arange(values.numel(), device=values.device)


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
    id="mask_to_indices",
    name="Mask To Indices",
    category="Sample Selection",
    description="Convert a boolean selection mask into stable local indices.",
    params={"mask": {"type": "slot", "default": "selected_mask"}, "save_as": {"type": "slot", "default": "selected_indices"}},
    requires=("mask",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 样本选择",
)
def mask_to_indices(ctx: ScratchContext, mask: str = "selected_mask", save_as: str = "selected_indices") -> None:
    torch = _torch()
    ctx[save_as] = torch.where(ctx[mask].bool())[0]


@block(
    id="invert_mask",
    name="Invert Mask",
    category="Sample Selection",
    description="Invert a boolean selection mask explicitly.",
    params={"mask": {"type": "slot", "default": "selected_mask"}, "save_as": {"type": "slot", "default": "inverted_mask"}},
    requires=("mask",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 样本选择",
)
def invert_mask(ctx: ScratchContext, mask: str = "selected_mask", save_as: str = "inverted_mask") -> None:
    ctx[save_as] = ~ctx[mask].bool()


@block(
    id="threshold_mask",
    name="Threshold Mask",
    category="Sample Selection",
    description="Select values using an explicit scalar threshold comparison.",
    params={"values": {"type": "slot", "default": "scores"}, "threshold": {"type": "float", "default": 0.5}, "comparison": {"type": "enum", "options": ["ge", "gt", "le", "lt"], "default": "ge"}, "save_as": {"type": "slot", "default": "selected_mask"}},
    requires=("values",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 样本选择",
)
def threshold_mask(ctx: ScratchContext, values: str = "scores", threshold: float = 0.5, comparison: str = "ge", save_as: str = "selected_mask") -> None:
    tensor = ctx[values].reshape(-1)
    if str(comparison) == "ge":
        mask = tensor >= float(threshold)
    elif str(comparison) == "gt":
        mask = tensor > float(threshold)
    elif str(comparison) == "le":
        mask = tensor <= float(threshold)
    elif str(comparison) == "lt":
        mask = tensor < float(threshold)
    else:
        raise ValueError(f"unsupported threshold comparison: {comparison}")
    ctx[save_as] = mask


@block(
    id="top_k_mask",
    name="Top-k Class Mask",
    category="Sample Selection",
    description="Select the top-k classes in each row and return only a boolean mask.",
    params={"scores": {"type": "slot", "default": "logits"}, "k": {"type": "int", "default": 1, "min": 0}, "save_as": {"type": "slot", "default": "top_k_mask"}},
    requires=("scores",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 样本选择",
)
def top_k_mask(ctx: ScratchContext, scores: str = "logits", k: int = 1, save_as: str = "top_k_mask") -> None:
    values = ctx[scores]
    if values.ndim != 2:
        raise ValueError("top_k_mask expects a [N,C] score matrix")
    count = min(max(int(k), 0), int(values.shape[1]))
    mask = _torch().zeros_like(values, dtype=_torch().bool)
    if count:
        mask.scatter_(1, _torch().topk(values, count, dim=1).indices, True)
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
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择", beginner_visible=False,
)
def top_k_confidence(ctx: ScratchContext, input: str = "probabilities", keep_rate: float = 0.8, save_as: str = "selected_indices") -> None:
    torch = _torch()
    confidence = ctx[input].max(dim=-1).values.reshape(-1)
    count = max(1, min(confidence.numel(), int(torch.ceil(torch.tensor(confidence.numel() * float(keep_rate))).item())))
    selected = torch.argsort(confidence, descending=True, stable=True)[:count]
    ctx[save_as] = selected
