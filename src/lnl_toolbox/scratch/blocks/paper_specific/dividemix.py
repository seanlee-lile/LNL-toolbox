"""Large-grain DivideMix lifecycle blocks for a recipe interpreter."""

from __future__ import annotations

from typing import Any

from ...context import ScratchContext
from ...registry import block


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("DivideMix blocks require PyTorch; install the `train` extra.") from exc
    return torch, F


@block(
    id="warmup",
    name="DivideMix Warmup",
    category="Paper Specific",
    description="Mark a warm-up lifecycle stage; child steps perform ordinary training.",
    kind="loop",
    params={"epochs": {"type": "int", "required": True, "min": 0}},
    provides=("epoch", "warmup_epoch"),
)
def warmup(ctx: ScratchContext, *, params: dict[str, Any], children, execute) -> None:
    for epoch in range(int(params["epochs"])):
        ctx["epoch"] = epoch
        ctx["warmup_epoch"] = epoch
        execute(children, ctx)


@block(
    id="fit_gmm",
    name="DivideMix: Fit Two-component GMM",
    category="Paper Specific",
    description="Fit a deterministic two-cluster loss mixture and save clean probabilities.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "clean_probability"}},
    requires=("losses",),
    provides=("save_as",),
)
def fit_gmm(ctx: ScratchContext, losses: str = "loss_per_sample", save_as: str = "clean_probability") -> None:
    torch, _ = _torch()
    values = ctx[losses].detach().reshape(-1).float()
    if values.numel() < 2:
        raise ValueError("DivideMix GMM needs at least two loss values")
    low, high = values.min(), values.max()
    for _ in range(20):
        distances = torch.stack(((values - low).abs(), (values - high).abs()), dim=1)
        assignment = distances.argmin(dim=1)
        if bool((assignment == 0).any()):
            low = values[assignment == 0].mean()
        if bool((assignment == 1).any()):
            high = values[assignment == 1].mean()
    clean_cluster = 0 if low <= high else 1
    clean = assignment == clean_cluster
    probability = torch.where(clean, torch.ones_like(values), torch.zeros_like(values))
    ctx[save_as] = probability


@block(
    id="split_clean_noisy",
    name="DivideMix: Split Clean/Noisy",
    category="Paper Specific",
    description="Turn clean probabilities into a clean mask and index list.",
    params={"probability": {"type": "slot", "default": "clean_probability"}, "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}, "indices_as": {"type": "slot", "default": "clean_indices"}},
    requires=("probability",),
    provides=("indices_as", "clean_mask"),
)
def split_clean_noisy(ctx: ScratchContext, probability: str = "clean_probability", threshold: float = 0.5, indices_as: str = "clean_indices") -> None:
    torch, _ = _torch()
    mask = ctx[probability].reshape(-1) >= float(threshold)
    ctx["clean_mask"] = mask
    ctx[indices_as] = torch.where(mask)[0]


@block(
    id="co_refine",
    name="DivideMix: Co-refine Labels",
    category="Paper Specific",
    description="Blend noisy one-hot labels with model probabilities using clean probabilities.",
    params={"probability": {"type": "slot", "default": "clean_probability"}, "labels": {"type": "slot", "default": "labels"}, "probs": {"type": "slot", "default": "probabilities"}, "save_as": {"type": "slot", "default": "refined_labels"}},
    requires=("probability", "labels", "probs"),
    provides=("save_as",),
)
def co_refine(ctx: ScratchContext, probability: str = "clean_probability", labels: str = "labels", probs: str = "probabilities", save_as: str = "refined_labels") -> None:
    torch, F = _torch()
    target = F.one_hot(ctx[labels].long(), num_classes=ctx[probs].shape[-1]).float()
    weight = ctx[probability].reshape(-1, 1).clamp(0.0, 1.0)
    ctx[save_as] = weight * target + (1.0 - weight) * ctx[probs].detach()


@block(
    id="mixmatch_step",
    name="DivideMix: MixMatch Step",
    category="Paper Specific",
    description="Compute the supervised part of a MixMatch-style update from refined labels.",
    params={"logits": {"type": "slot", "default": "logits"}, "targets": {"type": "slot", "default": "refined_labels"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "targets"),
    provides=("save_as",),
)
def mixmatch_step(ctx: ScratchContext, logits: str = "logits", targets: str = "refined_labels", save_as: str = "loss") -> None:
    _, F = _torch()
    ctx[save_as] = -(ctx[targets] * F.log_softmax(ctx[logits], dim=-1)).sum(dim=-1).mean()
