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
    from lnl_toolbox.estimators import DivideMixGMMCleanProbabilityEstimator, DivideMixGMMLossInput
    normalized = (values - values.min()) / (values.max() - values.min()).clamp_min(torch.finfo(values.dtype).eps)
    try:
        result = DivideMixGMMCleanProbabilityEstimator(random_seed=0, max_iter=10, tolerance=1e-2, covariance_regularization=5e-4, minimum_mean_separation=1e-6).estimate(DivideMixGMMLossInput(normalized, torch.arange(values.numel(), device=values.device)))
        ctx[save_as] = result.scores.to(values.device, dtype=values.dtype)
    except (ValueError, RuntimeError):
        ctx[save_as] = 1.0 - normalized


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
    torch, _ = _torch()
    from lnl_toolbox.algorithms.dividemix.objective import dividemix_objective, unsupervised_weight
    mask = ctx.get("clean_mask")
    if mask is None or not bool(mask.any()) or not bool((~mask).any()):
        mask = __import__("torch").ones(ctx[logits].shape[0], dtype=torch.bool, device=ctx[logits].device)
    labeled_logits, labeled_targets = ctx[logits][mask], ctx[targets][mask]
    unlabeled_logits, unlabeled_targets = ctx[logits][~mask], ctx[targets][~mask]
    if unlabeled_logits.numel() == 0:
        unlabeled_logits, unlabeled_targets = labeled_logits, labeled_targets
    objective, metrics = dividemix_objective(
        labeled_logits, labeled_targets, unlabeled_logits, unlabeled_targets, ctx[logits],
        lambda_u=unsupervised_weight(25.0, float(ctx.get("epoch", 0)), 10, 16), lambda_r=1.0,
    )
    ctx[save_as] = objective
    ctx["dividemix_metrics"] = metrics
