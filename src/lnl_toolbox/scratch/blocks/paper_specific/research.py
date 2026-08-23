"""Independent paper-level noisy-label operations.

These blocks intentionally operate on Context slots instead of calling the
legacy algorithm package.  They are small, semantic operations that can be
composed in a recipe and are useful for shape/value smoke checks.
"""

from __future__ import annotations

from typing import Any

from ...context import ScratchContext
from ...registry import block


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional train extra
        raise RuntimeError("Paper blocks require PyTorch; install the `train` extra.") from exc
    return torch, F


def _row_normalize(value):
    torch, _ = _torch()
    return value / value.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(value.dtype).tiny)


def _ce(logits, labels):
    _, F = _torch()
    return F.cross_entropy(logits, labels.long(), reduction="none")


@block(
    id="prepare_paper_batch",
    name="Prepare Paper Smoke Batch",
    category="Paper Specific",
    description="Create deterministic logits, labels, features, and a transition matrix for recipe shape checks.",
    params={
        "samples": {"type": "int", "default": 16, "min": 2},
        "features": {"type": "int", "default": 8, "min": 2},
        "classes": {"type": "int", "default": 3, "min": 2},
    },
    provides=("labels", "logits", "logits_a", "logits_b", "features", "probabilities", "loss_per_sample", "transition", "transition_a", "transition_b"),
)
def prepare_paper_batch(ctx: ScratchContext, samples: int = 16, features: int = 8, classes: int = 3) -> None:
    torch, F = _torch()
    generator = torch.Generator().manual_seed(int(ctx.get("seed", 1)))
    labels = torch.randint(classes, (samples,), generator=generator)
    logits = torch.randn(samples, classes, generator=generator, requires_grad=True)
    logits_a = torch.randn(samples, classes, generator=generator, requires_grad=True)
    logits_b = torch.randn(samples, classes, generator=generator, requires_grad=True)
    features_value = torch.randn(samples, features, generator=generator)
    transition = torch.rand(classes, classes, generator=generator) + 0.2
    transition = _row_normalize(transition)
    ctx.update({
        "labels": labels,
        "logits": logits,
        "logits_a": logits_a,
        "logits_b": logits_b,
        "features": features_value,
        "probabilities": F.softmax(logits, dim=-1),
        "loss_per_sample": _ce(logits, labels),
        "transition": transition,
        "transition_a": transition,
        "transition_b": torch.eye(classes),
        "num_classes": int(classes),
    })


@block(
    id="pdl_instance_transition",
    name="PDL: Instance Transition",
    category="Paper Specific",
    description="Build feature-dependent row-stochastic transition matrices and corrected per-sample risk.",
    params={"logits": {"type": "slot", "default": "logits"}, "features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "pdl_loss"}},
    requires=("logits", "features", "labels"),
    provides=("save_as", "instance_transition"),
)
def pdl_instance_transition(ctx: ScratchContext, logits: str = "logits", features: str = "features", labels: str = "labels", save_as: str = "pdl_loss") -> None:
    torch, _ = _torch()
    classes = ctx[logits].shape[-1]
    scale = torch.sigmoid(ctx[features].mean(dim=-1, keepdim=True))
    base = torch.eye(classes, device=ctx[logits].device).expand(ctx[logits].shape[0], -1, -1).clone()
    base = base * (0.7 + 0.2 * scale.unsqueeze(-1)) + (1.0 - base) * (0.3 - 0.2 * scale.unsqueeze(-1)) / (classes - 1)
    base = _row_normalize(base)
    observed = torch.bmm(torch.softmax(ctx[logits], -1).unsqueeze(1), base).squeeze(1)
    ctx["instance_transition"] = base
    ctx[save_as] = -torch.log(observed.gather(1, ctx[labels].long()[:, None]).squeeze(1).clamp_min(1e-12))


@block(
    id="jocor_agreement",
    name="JoCoR: Joint Agreement",
    category="Paper Specific",
    description="Combine two peer CE losses with symmetric prediction agreement.",
    params={"logits_a": {"type": "slot", "default": "logits_a"}, "logits_b": {"type": "slot", "default": "logits_b"}, "labels": {"type": "slot", "default": "labels"}, "agreement": {"type": "float", "default": 0.1, "min": 0.0}, "save_as": {"type": "slot", "default": "joint_loss_per_sample"}},
    requires=("logits_a", "logits_b", "labels"),
    provides=("save_as",),
)
def jocor_agreement(ctx: ScratchContext, logits_a: str = "logits_a", logits_b: str = "logits_b", labels: str = "labels", agreement: float = 0.1, save_as: str = "joint_loss_per_sample") -> None:
    torch, F = _torch()
    pa, pb = F.softmax(ctx[logits_a], -1), F.softmax(ctx[logits_b], -1)
    kl = F.kl_div(pa.clamp_min(1e-12).log(), pb, reduction="none").sum(-1) + F.kl_div(pb.clamp_min(1e-12).log(), pa, reduction="none").sum(-1)
    ctx[save_as] = (_ce(ctx[logits_a], ctx[labels]) + _ce(ctx[logits_b], ctx[labels])) / 2 + float(agreement) * kl / 2


@block(
    id="dss_evidence",
    name="DSS: Evidence Update",
    category="Paper Specific",
    description="Convert per-sample loss evidence into a stable debiased selection score.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "dss_score"}},
    requires=("losses",),
    provides=("save_as",),
)
def dss_evidence(ctx: ScratchContext, losses: str = "loss_per_sample", save_as: str = "dss_score") -> None:
    values = ctx[losses].detach()
    normalized = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
    ctx[save_as] = torch_sigmoid(-normalized)


def torch_sigmoid(value):
    torch, _ = _torch()
    return torch.sigmoid(value)


@block(
    id="cdr_parameter_mask",
    name="CDR: Critical Parameter Mask",
    category="Paper Specific",
    description="Keep the lower-loss half as a simple critical-update mask for smoke execution.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "keep_rate": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "critical_mask"}},
    requires=("losses",),
    provides=("save_as",),
)
def cdr_parameter_mask(ctx: ScratchContext, losses: str = "loss_per_sample", keep_rate: float = 0.8, save_as: str = "critical_mask") -> None:
    torch, _ = _torch()
    values = ctx[losses].reshape(-1)
    count = max(1, min(values.numel(), int(torch.ceil(torch.tensor(values.numel() * float(keep_rate))).item())))
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[torch.argsort(values, stable=True)[:count]] = True
    ctx[save_as] = mask


@block(
    id="mentor_weight",
    name="MentorNet: Weight Estimation",
    category="Paper Specific",
    description="Turn loss and epoch into a curriculum weight in [0, 1].",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "sample_weights"}},
    requires=("losses",),
    provides=("save_as",),
)
def mentor_weight(ctx: ScratchContext, losses: str = "loss_per_sample", save_as: str = "sample_weights") -> None:
    values = ctx[losses].detach()
    ctx[save_as] = torch_sigmoid(-(values - values.median()))


@block(
    id="estimate_transition",
    name="Estimate Transition Matrix",
    category="Transition",
    description="Estimate a class-conditional transition matrix from model predictions and observed labels.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "estimated_transition"}},
    requires=("logits", "labels"),
    provides=("save_as",),
)
def estimate_transition(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", save_as: str = "estimated_transition") -> None:
    torch, F = _torch()
    probabilities = F.softmax(ctx[logits].detach(), -1)
    classes = probabilities.shape[-1]
    result = torch.zeros(classes, classes, device=probabilities.device)
    for label in range(classes):
        mask = ctx[labels].long() == label
        result[label] = probabilities[mask].mean(0) if bool(mask.any()) else torch.full((classes,), 1.0 / classes, device=probabilities.device)
    ctx[save_as] = _row_normalize(result)


@block(
    id="compose_transition",
    name="Compose Transition Matrices",
    category="Transition",
    description="Compose two row-stochastic transition matrices for Dual-T style correction.",
    params={"first": {"type": "slot", "default": "transition_a"}, "second": {"type": "slot", "default": "transition_b"}, "save_as": {"type": "slot", "default": "composed_transition"}},
    requires=("first", "second"),
    provides=("save_as",),
)
def compose_transition(ctx: ScratchContext, first: str = "transition_a", second: str = "transition_b", save_as: str = "composed_transition") -> None:
    ctx[save_as] = _row_normalize(ctx[first] @ ctx[second])


@block(
    id="importance_reweight",
    name="Importance Reweight",
    category="Weighting",
    description="Weight per-sample CE by clean posterior over observed noisy posterior.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "weighted_loss"}},
    requires=("logits", "labels", "transition"),
    provides=("save_as",),
)
def importance_reweight(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", transition: str = "transition", save_as: str = "weighted_loss") -> None:
    torch, F = _torch()
    clean = F.softmax(ctx[logits], -1)
    noisy = clean @ ctx[transition].to(clean)
    target = ctx[labels].long()[:, None]
    weights = clean.gather(1, target).squeeze(1) / noisy.gather(1, target).squeeze(1).clamp_min(torch.finfo(clean.dtype).tiny)
    ctx[save_as] = _ce(ctx[logits], ctx[labels]) * weights.detach()


@block(
    id="cwd_statistics",
    name="CWD: Class-wise Statistics",
    category="Paper Specific",
    description="Compute class centroids and a global denoising objective from features.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "cwd_loss"}},
    requires=("features", "labels"),
    provides=("save_as", "class_centroids"),
)
def cwd_statistics(ctx: ScratchContext, features: str = "features", labels: str = "labels", save_as: str = "cwd_loss") -> None:
    torch, _ = _torch()
    values, targets = ctx[features], ctx[labels].long()
    classes = int(ctx.get("num_classes", int(targets.max().item()) + 1))
    centroids = torch.stack([values[targets == c].mean(0) if bool((targets == c).any()) else torch.zeros(values.shape[1], device=values.device) for c in range(classes)])
    ctx["class_centroids"] = centroids
    ctx[save_as] = ((values - centroids[targets]) ** 2).mean(dim=1)


@block(
    id="pcse_statistics",
    name="PCSE: Recover Per-class Statistics",
    category="Paper Specific",
    description="Recover means and variances from feature snapshots grouped by observed class.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "pcse_statistics"}},
    requires=("features", "labels"),
    provides=("save_as",),
)
def pcse_statistics(ctx: ScratchContext, features: str = "features", labels: str = "labels", save_as: str = "pcse_statistics") -> None:
    torch, _ = _torch()
    values, targets = ctx[features], ctx[labels].long()
    classes = int(ctx.get("num_classes", int(targets.max().item()) + 1))
    rows = []
    for c in range(classes):
        group = values[targets == c]
        rows.append(torch.cat((group.mean(0), group.var(0, unbiased=False)) if group.numel() else (torch.zeros(values.shape[1], device=values.device), torch.zeros(values.shape[1], device=values.device))))
    ctx[save_as] = torch.stack(rows)


@block(
    id="fine_feature_filter",
    name="FINE: Feature Filtering",
    category="Paper Specific",
    description="Filter feature embeddings by robust norm threshold.",
    params={"features": {"type": "slot", "default": "features"}, "quantile": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "fine_mask"}},
    requires=("features",),
    provides=("save_as",),
)
def fine_feature_filter(ctx: ScratchContext, features: str = "features", quantile: float = 0.5, save_as: str = "fine_mask") -> None:
    values = ctx[features].norm(dim=-1)
    ctx[save_as] = values >= values.quantile(float(quantile))


@block(
    id="cnlcu_uncertainty",
    name="CNLCU: Uncertainty Selection",
    category="Paper Specific",
    description="Select low-uncertainty examples from a loss history vector.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "cnlcu_mask"}},
    requires=("losses",),
    provides=("save_as",),
)
def cnlcu_uncertainty(ctx: ScratchContext, losses: str = "loss_per_sample", threshold: float = 0.5, save_as: str = "cnlcu_mask") -> None:
    values = ctx[losses]
    confidence = torch_sigmoid(-(values - values.median()))
    ctx[save_as] = confidence >= float(threshold)


@block(
    id="revise_transition",
    name="T-Revision: Revise Transition",
    category="Transition",
    description="Blend an estimated transition matrix with its row-stochastic identity prior.",
    params={"transition": {"type": "slot", "default": "transition"}, "strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "revised_transition"}},
    requires=("transition",),
    provides=("save_as",),
)
def revise_transition(ctx: ScratchContext, transition: str = "transition", strength: float = 0.5, save_as: str = "revised_transition") -> None:
    torch, _ = _torch()
    matrix = ctx[transition]
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    ctx[save_as] = _row_normalize((1.0 - float(strength)) * matrix + float(strength) * identity)


@block(
    id="dld_label_diffusion",
    name="DLD: Directional Label Diffusion",
    category="Paper Specific",
    description="Diffuse one-hot labels toward a feature-induced directional posterior.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "dld_labels"}},
    requires=("features", "labels"),
    provides=("save_as",),
)
def dld_label_diffusion(ctx: ScratchContext, features: str = "features", labels: str = "labels", save_as: str = "dld_labels") -> None:
    torch, F = _torch()
    values, targets = ctx[features], ctx[labels].long()
    classes = int(ctx.get("num_classes", int(targets.max().item()) + 1))
    prototypes = torch.stack([values[targets == c].mean(0) if bool((targets == c).any()) else torch.zeros(values.shape[1], device=values.device) for c in range(classes)])
    posterior = F.softmax(values @ prototypes.t(), -1)
    ctx[save_as] = 0.5 * F.one_hot(targets, classes).float() + 0.5 * posterior


@block(
    id="volminnet_objective",
    name="VolMinNet: Minimum-volume Objective",
    category="Paper Specific",
    description="Combine noisy-label NLL with a positive log-determinant volume penalty.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "transition": {"type": "slot", "default": "transition"}, "volume_weight": {"type": "float", "default": 0.01, "min": 0.0}, "save_as": {"type": "slot", "default": "volmin_loss"}},
    requires=("logits", "labels", "transition"),
    provides=("save_as",),
)
def volminnet_objective(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", transition: str = "transition", volume_weight: float = 0.01, save_as: str = "volmin_loss") -> None:
    torch, _ = _torch()
    observed = torch.softmax(ctx[logits], -1) @ ctx[transition].to(ctx[logits])
    nll = -torch.log(observed.gather(1, ctx[labels].long()[:, None]).squeeze(1).clamp_min(1e-12)).mean()
    volume = -torch.logdet(ctx[transition].to(ctx[logits]).clamp_min(1e-6).abs())
    ctx[save_as] = nll + float(volume_weight) * volume


@block(
    id="upm_eta_update",
    name="UPM: Confusing Probability Update",
    category="Paper Specific",
    description="Update instance-dependent corruption probabilities with a projected step.",
    params={"probabilities": {"type": "slot", "default": "probabilities"}, "labels": {"type": "slot", "default": "labels"}, "step_size": {"type": "float", "default": 0.1, "min": 0.0}, "save_as": {"type": "slot", "default": "eta"}},
    requires=("probabilities", "labels"),
    provides=("save_as",),
)
def upm_eta_update(ctx: ScratchContext, probabilities: str = "probabilities", labels: str = "labels", step_size: float = 0.1, save_as: str = "eta") -> None:
    torch, _ = _torch()
    observed = ctx[probabilities].gather(1, ctx[labels].long()[:, None]).squeeze(1)
    ctx[save_as] = (observed + float(step_size) * (1.0 - observed)).clamp(0.0, 1.0).detach()


@block(
    id="lend_label_dilution",
    name="LEND: Label Dilution",
    category="Paper Specific",
    description="Use feature-centroid agreement to dilute observed one-hot labels.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "lend_labels"}},
    requires=("features", "labels"),
    provides=("save_as",),
)
def lend_label_dilution(ctx: ScratchContext, features: str = "features", labels: str = "labels", save_as: str = "lend_labels") -> None:
    torch, F = _torch()
    values, targets = ctx[features], ctx[labels].long()
    classes = int(ctx.get("num_classes", int(targets.max().item()) + 1))
    prototypes = torch.stack([values[targets == c].mean(0) if bool((targets == c).any()) else torch.zeros(values.shape[1], device=values.device) for c in range(classes)])
    posterior = F.softmax(F.normalize(values, dim=-1) @ F.normalize(prototypes, dim=-1).t(), -1)
    ctx[save_as] = 0.25 * F.one_hot(targets, classes).float() + 0.75 * posterior


@block(
    id="cal_second_order_risk",
    name="CAL: Second-order Risk",
    category="Correction",
    description="Use a centered squared correction term around the batch risk mean.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "cal_loss"}},
    requires=("logits", "labels"),
    provides=("save_as",),
)
def cal_second_order_risk(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", save_as: str = "cal_loss") -> None:
    losses = _ce(ctx[logits], ctx[labels])
    centered = losses - losses.detach().mean()
    ctx[save_as] = losses + centered.square()


@block(
    id="mc_ldce_centroid_risk",
    name="MC-LDCE: Centroid Risk",
    category="Paper Specific",
    description="Penalize feature distance from the observed-class centroid alongside CE.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "logits": {"type": "slot", "default": "logits"}, "save_as": {"type": "slot", "default": "mc_ldce_loss"}},
    requires=("features", "labels", "logits"),
    provides=("save_as",),
)
def mc_ldce_centroid_risk(ctx: ScratchContext, features: str = "features", labels: str = "labels", logits: str = "logits", save_as: str = "mc_ldce_loss") -> None:
    values, targets = ctx[features], ctx[labels].long()
    prototypes = values.new_zeros((int(ctx.get("num_classes", int(targets.max().item()) + 1)), values.shape[-1]))
    for c in range(prototypes.shape[0]):
        if bool((targets == c).any()):
            prototypes[c] = values[targets == c].mean(0)
    ctx[save_as] = _ce(ctx[logits], targets) + (values - prototypes[targets]).square().mean(-1)


@block(
    id="ca2c_candidate_memory",
    name="CA2C: Candidate Memory",
    category="Paper Specific",
    description="Keep low-loss candidates as a stable-memory mask for asymmetric co-learning.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "keep_rate": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "ca2c_mask"}},
    requires=("losses",),
    provides=("save_as",),
)
def ca2c_candidate_memory(ctx: ScratchContext, losses: str = "loss_per_sample", keep_rate: float = 0.8, save_as: str = "ca2c_mask") -> None:
    torch, _ = _torch()
    values = ctx[losses].reshape(-1)
    count = max(1, min(values.numel(), int(torch.ceil(torch.tensor(values.numel() * float(keep_rate))).item())))
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[torch.argsort(values, stable=True)[:count]] = True
    ctx[save_as] = mask


@block(
    id="l2rw_meta_weight",
    name="L2RW: Meta Weight",
    category="Weighting",
    description="Produce detached normalized example weights from a virtual meta objective.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "meta_weights"}},
    requires=("losses",),
    provides=("save_as",),
)
def l2rw_meta_weight(ctx: ScratchContext, losses: str = "loss_per_sample", save_as: str = "meta_weights") -> None:
    weights = (-ctx[losses].detach()).softmax(dim=0)
    ctx[save_as] = weights * weights.numel()
