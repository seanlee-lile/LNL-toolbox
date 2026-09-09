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


# Kept as a Python import alias for callers that historically imported this
# helper from ``selection``; the registered block lives in ``schedule`` where
# it belongs semantically.
from .schedule import linear_rate_schedule


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
    formula="S=arg lowest_k(score), k=rounding(N·fraction)", formula_kind="primitive",
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
    if not torch.isfinite(torch.tensor(fraction)) or not 0.0 <= fraction <= 1.0:
        raise ValueError("keep_fraction must be a finite value in [0, 1]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("scores must be finite for stable lowest-score selection")
    if str(rounding) not in {"floor", "ceil"}:
        raise ValueError("rounding must be 'floor' or 'ceil'")
    if int(minimum_count) < 0:
        raise ValueError("minimum_count must be non-negative")
    raw = values.numel() * fraction
    count = int(torch.ceil(torch.tensor(raw)).item()) if str(rounding) == "ceil" else int(torch.floor(torch.tensor(raw)).item())
    count = min(values.numel(), max(int(minimum_count), count))
    if stable_sample_indices in ctx:
        stable = ctx[stable_sample_indices].reshape(-1).to(values.device)
        if stable.numel() != values.numel() or stable.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        }:
            raise ValueError("stable_sample_indices must be integer values aligned with scores")
        if torch.unique(stable).numel() != stable.numel():
            raise ValueError("stable_sample_indices must be unique")
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
    formula="v_selected=v[indices]", formula_ref="indexed selection", formula_kind="primitive",
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
    formula="m_j=1[j in indices]", formula_ref="explicit index-to-mask conversion", formula_kind="primitive",
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
    id="agreement_mask",
    name="Agreement Mask",
    category="Sample Selection",
    description="Select rows whose hard label agrees with the argmax of a soft history.",
    params={"labels": {"type": "slot", "default": "labels"}, "history": {"type": "slot", "default": "history_values"}, "save_as": {"type": "slot", "default": "selected_mask"}},
    requires=("labels", "history"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑦ 样本选择",
)
def agreement_mask(ctx: ScratchContext, labels: str = "labels", history: str = "history_values", save_as: str = "selected_mask") -> None:
    ctx[save_as] = ctx[history].detach().argmax(dim=-1).eq(ctx[labels].long())


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
    id="quantile",
    name="Quantile",
    category="Sample Selection",
    description="Compute a tensor quantile without embedding a threshold or selection policy.",
    params={"values": {"type": "slot", "default": "values"},
            "q": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "dim": {"type": "int", "default": -1},
            "save_as": {"type": "slot", "default": "quantile_value"}},
    requires=("values",), provides=("save_as",), placement=("batch", "epoch"), stage="train", ui_group="⑥ 样本选择",
)
def quantile(ctx: ScratchContext, values: str = "values", q: float = 0.5, dim: int = -1,
             save_as: str = "quantile_value") -> None:
    torch = _torch()
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("quantile q must be in [0, 1]")
    ctx[save_as] = torch.quantile(ctx[values], float(q), dim=None if int(dim) == -1 else int(dim))


@block(
    id="mask_logits",
    name="Mask Logits",
    category="Sample Selection",
    description="Set logits for masked classes to negative infinity without changing the mask.",
    params={"logits": {"type": "slot", "default": "logits"},
            "mask": {"type": "slot", "default": "mask"},
            "masked_value": {"type": "float", "default": float("-inf")},
            "save_as": {"type": "slot", "default": "masked_logits"}},
    requires=("logits", "mask"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 样本选择",
)
def mask_logits(ctx: ScratchContext, logits: str = "logits", mask: str = "mask",
                masked_value: float = float("-inf"), save_as: str = "masked_logits") -> None:
    values = ctx[logits]
    selector = ctx[mask].bool()
    if selector.shape != values.shape:
        raise ValueError("mask_logits mask and logits must have identical shapes")
    ctx[save_as] = values.masked_fill(selector, float(masked_value))


@block(
    id="classwise_percentile_anchor_candidates",
    name="Classwise Percentile Anchor Candidates",
    category="Sample Selection",
    description="Choose one highest-scoring stable sample per class and percentile level.",
    params={"scores": {"type": "slot", "default": "posterior"},
            "percentiles": {"type": "value", "default": [0.97, 0.98, 0.99]},
            "stable_sample_indices": {"type": "slot", "default": "indices"},
            "save_as": {"type": "slot", "default": "anchor_indices"}},
    requires=("scores",), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 样本选择",
)
def classwise_percentile_anchor_candidates(ctx: ScratchContext, scores: str = "posterior",
                                           percentiles: Any = (0.97, 0.98, 0.99),
                                           stable_sample_indices: str = "indices",
                                           save_as: str = "anchor_indices") -> None:
    torch = _torch(); source = ctx[scores]
    # Posterior snapshots are a public statistics output.  Accepting one here
    # keeps selection composable without making callers unpack its fields by
    # hand; plain score tensors remain the canonical operation input.
    if hasattr(source, "noisy_probabilities"):
        values = source.noisy_probabilities
        if stable_sample_indices not in ctx and hasattr(source, "global_indices"):
            stable = source.global_indices
        else:
            stable = None
    elif isinstance(source, dict) and "noisy_probabilities" in source:
        values = source["noisy_probabilities"]
        if stable_sample_indices not in ctx:
            stable = source.get("global_indices", source.get("indices"))
        else:
            stable = None
    else:
        values = source
        stable = None
    values = torch.as_tensor(values)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("classwise_percentile_anchor_candidates expects a non-empty [N,C] matrix")
    levels = [float(value) for value in (percentiles or (0.97, 0.98, 0.99))]
    if any(value > 1.0 for value in levels):
        levels = [value / 100.0 for value in levels]
    if any(not 0.0 <= value <= 1.0 for value in levels):
        raise ValueError("percentiles must lie in [0,1] or [0,100]")
    if stable is not None:
        stable = torch.as_tensor(stable, device=values.device).reshape(-1)
    elif stable_sample_indices in ctx:
        stable = torch.as_tensor(ctx[stable_sample_indices], device=values.device).reshape(-1)
        if stable.numel() != values.shape[0] or stable.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("stable_sample_indices must be integer values aligned with scores")
    else:
        stable = torch.arange(values.shape[0], device=values.device)
    import numpy as np
    result = torch.empty((values.shape[1], len(levels)), dtype=stable.dtype, device=values.device)
    for cls in range(values.shape[1]):
        column = values[:, cls].detach().cpu().numpy()
        ids = stable.detach().cpu().numpy()
        for pos, level in enumerate(levels):
            threshold = float(np.quantile(column, level, method="higher"))
            eligible = np.flatnonzero(column >= threshold)
            if eligible.size == 0:
                eligible = np.arange(column.size)
            # Highest score wins; stable sample identity resolves ties.
            best = sorted(eligible.tolist(), key=lambda row: (-float(column[row]), int(ids[row])))[0]
            result[cls, pos] = stable[best]
    ctx[save_as] = result


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
        # Stable sorting is part of the public selection contract.  ``topk``
        # does not define tie ordering across devices, while argsort(stable)
        # preserves the input/sample order for equal scores.
        indices = _torch().argsort(values, dim=1, descending=True, stable=True)[:, :count]
        mask.scatter_(1, indices, True)
    ctx[save_as] = mask


@block(
    id="mean_by_indices",
    name="Mean By Indices",
    category="Sample Selection",
    description="Reduce selected local values to their scalar mean.",
    params={"values": {"type": "slot", "default": "loss_per_sample"}, "indices": {"type": "slot", "default": "selected_indices"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("values", "indices"), provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 样本选择",
    formula="mean(v[indices])", formula_ref="indexed mean reduction", formula_kind="composite",
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
