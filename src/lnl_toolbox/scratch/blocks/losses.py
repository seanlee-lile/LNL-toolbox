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
    placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="CE(z, y) = -log softmax(z)_y", formula_ref="standard cross-entropy definition",
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
    id="weighted_loss",
    name="Apply Sample Weights",
    category="Weighting",
    description="Multiply a detached per-sample weight by a per-sample loss before the common reduction.",
    params={
        "losses": {"type": "slot", "default": "loss_per_sample"},
        "weights": {"type": "slot", "default": "sample_weights"},
        "save_as": {"type": "slot", "default": "weighted_loss_per_sample"},
    },
    requires=("losses", "weights"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="l_i^weighted = w_i l_i",
    formula_ref="Li et al., importance-weighted empirical risk minimization",
    paper="Learning from Noisy Labels with Importance Reweighting",
)
def weighted_loss(
    ctx: ScratchContext,
    losses: str = "loss_per_sample",
    weights: str = "sample_weights",
    save_as: str = "weighted_loss_per_sample",
) -> None:
    values = ctx[losses]
    factors = ctx[weights].detach()
    if values.shape != factors.shape:
        raise ValueError("sample weights and per-sample losses must have the same shape")
    _save_loss(ctx, values * factors, save_as)


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
    placement=("batch",), stage="train", ui_group="⑤ 损失公式", beginner_visible=False,
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
    placement=("batch",), stage="train", ui_group="⑤ 损失公式", beginner_visible=False,
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
    id="gather_target_probability",
    name="Gather Target Probability",
    category="Loss",
    description="Gather the probability assigned to each example's observed target class.",
    params={
        "probabilities": {"type": "slot", "default": "probabilities"},
        "labels": {"type": "slot", "default": "labels"},
        "save_as": {"type": "slot", "default": "target_probability"},
    },
    requires=("probabilities", "labels"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="p_y = f_y(x)", formula_ref="GCE target-class probability definition", paper="Generalized Cross Entropy",
)
def gather_target_probability(
    ctx: ScratchContext,
    probabilities: str = "probabilities",
    labels: str = "labels",
    save_as: str = "target_probability",
) -> None:
    ctx[save_as] = ctx[probabilities].gather(1, ctx[labels].long().view(-1, 1)).squeeze(1).clamp_min(1e-12)


@block(
    id="gce_q_formula",
    name="GCE q Formula",
    category="Loss",
    description="Compute the per-sample Generalized Cross Entropy value from target probability.",
    params={
        "input": {"type": "slot", "default": "target_probability"},
        "q": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("input",),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L_q(f(x), y) = (1 - p_y^q) / q", formula_ref="Generalized Cross Entropy definition", paper="Generalized Cross Entropy",
)
def gce_q_formula(
    ctx: ScratchContext,
    input: str = "target_probability",
    q: float = 0.7,
    save_as: str = "loss_per_sample",
) -> None:
    values = ctx[input].clamp_min(1e-12)
    if float(q) == 0.0:
        ctx[save_as] = -values.log()
    else:
        ctx[save_as] = (1.0 - values.pow(float(q))) / float(q)


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
    placement=("batch",), stage="train", ui_group="⑤ 损失公式", beginner_visible=False,
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
    description="Normalized cross entropy active loss, one value per sample.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L_NCE = -log p_y / Σ_j(-log p_j)",
    formula_ref="Normalized Cross Entropy definition in Ma et al. (2020), Eq. (2)",
    paper="Normalized Loss Functions for Deep Learning with Noisy Labels",
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
    description="Reverse cross entropy passive loss with clipped one-hot labels, one value per sample.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "log_zero": {"type": "float", "default": -4.0},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L_RCE = -Σ_j p_j log(ŷ_j), where log(ŷ_y)=0 and log(ŷ_j)=log_zero otherwise",
    formula_ref="Reverse Cross Entropy definition in Wang et al. (2019), as used by Ma et al. (2020)",
    paper="Normalized Loss Functions for Deep Learning with Noisy Labels",
)
def rce_loss(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", log_zero: float = -4.0, save_as: str = "loss_per_sample") -> None:
    torch, _ = _torch()
    probabilities = torch.softmax(ctx[logits], dim=-1).clamp_min(1e-12)
    target = torch.full_like(probabilities, float(log_zero)).scatter_(1, ctx[labels].long().view(-1, 1), 0.0)
    _save_loss(ctx, -(probabilities * target).sum(dim=1), save_as)


@block(
    id="active_passive_composition",
    name="Active-Passive Composition",
    category="Loss",
    description="Combine two per-sample active and passive losses before the common reduction.",
    params={
        "active": {"type": "slot", "default": "active_loss_per_sample"},
        "passive": {"type": "slot", "default": "passive_loss_per_sample"},
        "alpha": {"type": "float", "default": 1.0},
        "beta": {"type": "float", "default": 1.0},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("active", "passive"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L_APL = α L_active + β L_passive",
    formula_ref="Active-passive loss composition in Ma et al. (2020), Eq. (5)",
    paper="Normalized Loss Functions for Deep Learning with Noisy Labels",
)
def active_passive_composition(
    ctx: ScratchContext,
    active: str = "active_loss_per_sample",
    passive: str = "passive_loss_per_sample",
    alpha: float = 1.0,
    beta: float = 1.0,
    save_as: str = "loss_per_sample",
) -> None:
    active_values = ctx[active]
    passive_values = ctx[passive]
    if active_values.shape != passive_values.shape:
        raise ValueError("active and passive losses must have the same per-sample shape")
    ctx[save_as] = float(alpha) * active_values + float(beta) * passive_values


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
    provides=("save_as",), beginner_visible=False,
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
    placement=("top", "batch"), stage="train", ui_group="⑤ 损失公式",
    formula="L = mean_i l_i", formula_ref="method definition",
)
def mean_loss(ctx: ScratchContext, input: str = "loss_per_sample", save_as: str = "loss") -> None:
    ctx[save_as] = ctx[input].mean()


@block(
    id="binary_risk",
    name="Binary Risk",
    category="Correction",
    description="Natarajan's unbiased binary risk for known class-dependent noise rates, one value per sample.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "rho_positive": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.999},
        "rho_negative": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.999},
        "save_as": {"type": "slot", "default": "loss_per_sample"},
    },
    requires=("logits", "labels"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="l̃_0=((1-ρ₊)l_0-ρ₋l_1)/(1-ρ₊-ρ₋); l̃_1=(-ρ₊l_0+(1-ρ₋)l_1)/(1-ρ₊-ρ₋)",
    formula_ref="Natarajan et al. (2013), unbiased risk estimator for class-dependent label noise",
    paper="Learning with Noisy Labels",
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
    placement=("batch",), stage="train", ui_group="⑦ 标签与矩阵", beginner_visible=True,
    formula="p_tilde = p T; L_forward = -log p_tilde[y_tilde]",
    formula_ref="Patrini et al. (CVPR 2017), Forward correction risk",
    paper="Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach",
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
    placement=("batch",), stage="train", ui_group="⑦ 标签与矩阵", beginner_visible=False,
)
def backward_correction(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", transition: str = "transition", save_as: str = "loss_per_sample") -> None:
    torch, F = _torch()
    log_probabilities = F.log_softmax(ctx[logits], dim=-1)
    all_losses = -log_probabilities
    corrected = torch.linalg.solve(ctx[transition].to(ctx[logits]), all_losses.transpose(0, 1)).transpose(0, 1)
    _save_loss(ctx, corrected.gather(1, ctx[labels].long().view(-1, 1)).squeeze(1), save_as)
