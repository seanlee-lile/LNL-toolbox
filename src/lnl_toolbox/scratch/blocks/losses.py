"""Independent per-sample noisy-label losses."""

from __future__ import annotations

from typing import Any

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Loss blocks require PyTorch; install the `train` extra.") from exc
    return torch, F


def _save_loss(ctx: ScratchContext, values: Any, save_as: str) -> None:
    ctx[save_as] = values


@block(
    id="per_sample_ce",
    name="Per-sample Cross Entropy",
    category="Loss",
    description="Compute one cross-entropy value per example.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
)
def per_sample_ce(
    ctx: ScratchContext,
    logits: str = "logits",
    labels: str = "labels",
    save_as: str = "loss_per_sample",
) -> None:
    _, F = _torch()
    _save_loss(ctx, F.cross_entropy(ctx[logits], ctx[labels].long(), reduction="none"), save_as)


@block(
    id="cross_entropy",
    name="Cross Entropy",
    category="Loss",
    description="Compute standard per-sample cross entropy for later reduction.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
)
def cross_entropy(ctx: ScratchContext, **params: Any) -> None:
    per_sample_ce(ctx, **params)


@block(
    id="gce_loss",
    name="GCE Loss",
    category="Loss",
    description="Generalized cross entropy, Lq, computed per sample.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "q": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
)
def gce_loss(
    ctx: ScratchContext,
    logits: str = "logits",
    labels: str = "labels",
    q: float = 0.7,
    save_as: str = "loss_per_sample",
) -> None:
    torch, F = _torch()
    if float(q) == 0.0:
        values = F.cross_entropy(ctx[logits], ctx[labels].long(), reduction="none")
    else:
        probabilities = torch.softmax(ctx[logits], dim=-1)
        target_probability = probabilities.gather(1, ctx[labels].long().view(-1, 1)).squeeze(1).clamp_min(1e-12)
        values = (1.0 - target_probability.pow(float(q))) / float(q)
    _save_loss(ctx, values, save_as)


@block(
    id="mae_loss",
    name="MAE Loss",
    category="Loss",
    description="Mean absolute error against one-hot labels, per sample.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
)
def mae_loss(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", save_as: str = "loss_per_sample") -> None:
    torch, _ = _torch()
    probabilities = torch.softmax(ctx[logits], dim=-1)
    target = torch.zeros_like(probabilities).scatter_(1, ctx[labels].long().view(-1, 1), 1.0)
    _save_loss(ctx, (probabilities - target).abs().mean(dim=1), save_as)


@block(
    id="nce_loss",
    name="NCE Loss",
    category="Loss",
    description="Normalized cross entropy, per sample.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
)
def nce_loss(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", save_as: str = "loss_per_sample") -> None:
    torch, F = _torch()
    log_probabilities = F.log_softmax(ctx[logits], dim=-1)
    ce = -log_probabilities.gather(1, ctx[labels].long().view(-1, 1)).squeeze(1)
    _save_loss(ctx, ce / (-log_probabilities).sum(dim=1).clamp_min(1e-12), save_as)


@block(
    id="rce_loss",
    name="RCE Loss",
    category="Loss",
    description="Reverse cross entropy against clipped one-hot labels.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "log_zero": {"type": "float", "default": -4.0},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
)
def rce_loss(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", log_zero: float = -4.0, save_as: str = "loss_per_sample") -> None:
    torch, _ = _torch()
    probabilities = torch.softmax(ctx[logits], dim=-1).clamp_min(1e-12)
    target = torch.full_like(probabilities, float(log_zero)).scatter_(1, ctx[labels].long().view(-1, 1), 0.0)
    _save_loss(ctx, -(probabilities * target).sum(dim=1), save_as)


@block(
    id="apl_loss",
    name="APL Loss",
    category="Loss",
    description="Active-passive loss: alpha NCE plus beta RCE.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "alpha": {"type": "float", "default": 1.0, "min": 0.0},
        "beta": {"type": "float", "default": 1.0, "min": 0.0},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
)
def apl_loss(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", alpha: float = 1.0, beta: float = 1.0, save_as: str = "loss_per_sample") -> None:
    torch, F = _torch()
    log_probabilities = F.log_softmax(ctx[logits], dim=-1)
    probabilities = log_probabilities.exp().clamp_min(1e-12)
    targets = ctx[labels].long().view(-1, 1)
    ce = -log_probabilities.gather(1, targets).squeeze(1)
    nce = ce / (-log_probabilities).sum(dim=1).clamp_min(1e-12)
    reverse_targets = torch.full_like(probabilities, -4.0).scatter_(1, targets, 0.0)
    rce = -(probabilities * reverse_targets).sum(dim=1)
    _save_loss(ctx, float(alpha) * nce + float(beta) * rce, save_as)


@block(
    id="mean_loss",
    name="Mean Loss",
    category="Loss",
    description="Reduce a per-sample loss to the scalar used by Backward.",
    params={"input": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("input",),
    provides=("save_as",),
)
def mean_loss(ctx: ScratchContext, input: str = "loss_per_sample", save_as: str = "loss") -> None:
    ctx[save_as] = ctx[input].mean()


@block(
    id="binary_risk",
    name="Binary Risk",
    category="Correction",
    description="Natarajan's unbiased binary risk for known class-dependent noise rates.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "rho_positive": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.999},
        "rho_negative": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.999},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
)
def binary_risk(
    ctx: ScratchContext,
    logits: str = "logits",
    labels: str = "labels",
    rho_positive: float = 0.1,
    rho_negative: float = 0.1,
    save_as: str = "loss_per_sample",
) -> None:
    torch, F = _torch()
    gap = 1.0 - float(rho_positive) - float(rho_negative)
    if gap <= 0.0:
        raise ValueError("binary noise rates must have positive identifiability gap")
    log_probabilities = F.log_softmax(ctx[logits], dim=-1)
    losses = -log_probabilities
    zero = ((1.0 - float(rho_positive)) * losses[:, 0] - float(rho_negative) * losses[:, 1]) / gap
    one = (-float(rho_positive) * losses[:, 0] + (1.0 - float(rho_negative)) * losses[:, 1]) / gap
    _save_loss(ctx, torch.where(ctx[labels].long() == 0, zero, one), save_as)


@block(
    id="forward_correction",
    name="Forward Correction",
    category="Correction",
    description="Map clean posterior through a row-stochastic transition matrix before CE.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "loss_per_sample"}},
    requires=("logits", "labels", "transition"),
    provides=("save_as",),
)
def forward_correction(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", transition: str = "transition", save_as: str = "loss_per_sample") -> None:
    torch, _ = _torch()
    probabilities = torch.softmax(ctx[logits], dim=-1) @ ctx[transition].to(ctx[logits])
    observed = probabilities.gather(1, ctx[labels].long().view(-1, 1)).squeeze(1)
    _save_loss(ctx, -torch.log(observed.clamp_min(torch.finfo(probabilities.dtype).tiny)), save_as)


@block(
    id="backward_correction",
    name="Backward Correction",
    category="Correction",
    description="Apply the inverse transition matrix to per-class clean losses.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "loss_per_sample"}},
    requires=("logits", "labels", "transition"),
    provides=("save_as",),
)
def backward_correction(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", transition: str = "transition", save_as: str = "loss_per_sample") -> None:
    torch, F = _torch()
    log_probabilities = F.log_softmax(ctx[logits], dim=-1)
    all_losses = -log_probabilities
    corrected = torch.linalg.solve(ctx[transition].to(ctx[logits]), all_losses.transpose(0, 1)).transpose(0, 1)
    _save_loss(ctx, corrected.gather(1, ctx[labels].long().view(-1, 1)).squeeze(1), save_as)
