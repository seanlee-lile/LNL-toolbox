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


def _class_labels(value: Any, classes: int, *, device: Any, slot: str) -> Any:
    """Validate class ids before an indexed CUDA operation.

    CUDA scatter/gather and cross-entropy report out-of-range labels as an
    asynchronous ``device-side assert``.  Moving a small copy to CPU for the
    precondition check lets Scratch report the actual contract violation
    (usually a dataset/model class-count mismatch) at the originating block.
    The returned tensor remains an ordinary long class-id tensor on the
    operation's device; no label value is changed.
    """
    torch, _ = _torch()
    raw = torch.as_tensor(value)
    flat = raw.detach().to("cpu").reshape(-1)
    if flat.numel() == 0:
        raise ValueError(f"label slot '{slot}' is empty")
    if flat.dtype.is_floating_point:
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError(f"label slot '{slot}' contains non-finite class ids")
        if not bool(torch.equal(flat, flat.round())):
            raise ValueError(f"label slot '{slot}' contains non-integral class ids")
    labels_cpu = flat.to(torch.long)
    minimum, maximum = int(labels_cpu.min().item()), int(labels_cpu.max().item())
    if minimum < 0 or maximum >= int(classes):
        raise ValueError(
            f"label slot '{slot}' contains class ids in [{minimum}, {maximum}], "
            f"but the logits/probability tensor has {int(classes)} classes; "
            "check that the selected dataset and model num_classes match"
        )
    return labels_cpu.to(device=device).view(-1)


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
    torch, F = _torch()
    scores = ctx[logits]
    targets = _class_labels(ctx[labels], scores.shape[-1], device=scores.device, slot=labels)
    _save_loss(ctx, F.cross_entropy(scores, targets, reduction="none"), save_as)


@block(
    id="symmetric_kl",
    name="Symmetric KL",
    category="Loss",
    description="Compute the per-sample symmetric KL divergence between two logits distributions.",
    params={
        "logits_a": {"type": "slot", "default": "logits_a"},
        "logits_b": {"type": "slot", "default": "logits_b"},
        "save_as": {"type": "slot", "default": "symmetric_kl_per_sample"},
    },
    requires=("logits_a", "logits_b"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="D_SKL(p_a,p_b)=KL(p_a||p_b)+KL(p_b||p_a)",
    formula_ref="symmetric KL divergence definition",
)
def symmetric_kl(
    ctx: ScratchContext,
    logits_a: str = "logits_a",
    logits_b: str = "logits_b",
    save_as: str = "symmetric_kl_per_sample",
) -> None:
    torch, F = _torch()
    log_a = F.log_softmax(ctx[logits_a], dim=-1)
    log_b = F.log_softmax(ctx[logits_b], dim=-1)
    values = F.kl_div(log_a, log_b.exp(), reduction="none").sum(-1)
    values = values + F.kl_div(log_b, log_a.exp(), reduction="none").sum(-1)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("symmetric KL produced non-finite values")
    _save_loss(ctx, values, save_as)


@block(
    id="partial_label_loss",
    name="Partial-label Loss",
    category="Loss",
    description="Compute a positive partial-label objective from candidate-class masks.",
    params={"logits": {"type": "slot", "default": "logits"},
            "candidates": {"type": "slot", "default": "candidate_mask"},
            "hard_weight": {"type": "float", "default": 0.99, "min": 0.0, "max": 1.0},
            "reduction": {"type": "enum", "options": ["per_sample", "mean"], "default": "mean"},
            "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "candidates"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def partial_label_loss(ctx: ScratchContext, logits: str = "logits", candidates: str = "candidate_mask",
                      hard_weight: float = 0.99, reduction: str = "mean", save_as: str = "loss") -> None:
    torch, F = _torch(); values = ctx[candidates].bool(); scores = ctx[logits]
    if scores.ndim != 2 or values.shape != scores.shape:
        raise ValueError("partial_label_loss expects aligned [N,C] logits and candidate mask")
    count = values.sum(dim=1)
    if bool((count == 0).any()):
        raise ValueError("partial_label_loss requires at least one candidate class per sample")
    soft_targets = values.to(scores.dtype) / count[:, None].to(scores.dtype)
    logp = F.log_softmax(scores, dim=-1)
    soft = -(soft_targets * logp).sum(dim=1)
    hard_labels = soft_targets.argmax(dim=1)
    hard = F.nll_loss(logp, hard_labels, reduction="none")
    result = float(hard_weight) * hard + (1.0 - float(hard_weight)) * soft
    ctx[save_as] = result if str(reduction) == "per_sample" else result.mean()


@block(
    id="complementary_negative_loss",
    name="Complementary-label Negative Loss",
    category="Loss",
    description="Penalize probability mass assigned to explicitly complementary classes.",
    params={"logits": {"type": "slot", "default": "logits"},
            "complements": {"type": "slot", "default": "complement_mask"},
            "reduction": {"type": "enum", "options": ["per_sample", "mean"], "default": "mean"},
            "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "complements"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def complementary_negative_loss(ctx: ScratchContext, logits: str = "logits", complements: str = "complement_mask",
                                reduction: str = "mean", save_as: str = "loss") -> None:
    torch, F = _torch(); scores = ctx[logits]; mask = ctx[complements].bool()
    if scores.ndim != 2 or mask.shape != scores.shape:
        raise ValueError("complementary_negative_loss expects aligned [N,C] logits and mask")
    # sigmoid(logit) is the binary positive probability used by the CA2C
    # complementary objective; log1p(-sigmoid) remains stable for large logits.
    per_sample = -(F.logsigmoid(-scores) * mask.to(scores.dtype)).sum(dim=1)
    ctx[save_as] = per_sample if str(reduction) == "per_sample" else per_sample.mean()


@block(
    id="gather_by_label",
    name="Gather By Label",
    category="Tensor Operation",
    description="Gather one value per row using the corresponding integer label.",
    params={
        "values": {"type": "slot", "default": "probabilities"},
        "labels": {"type": "slot", "default": "labels"},
        "save_as": {"type": "slot", "default": "gathered_values"},
    },
    requires=("values", "labels"),
    provides=("save_as",),
    placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="v_i=values[i,labels_i]", formula_ref="row-wise indexed gather",
)
def gather_by_label(ctx: ScratchContext, values: str = "probabilities", labels: str = "labels", save_as: str = "gathered_values") -> None:
    torch = __import__("torch")
    matrix = torch.as_tensor(ctx[values])
    target = _class_labels(ctx[labels], matrix.shape[1], device=matrix.device, slot=labels).view(-1, 1)
    if target.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"gather_by_label expects one label per row: values has {matrix.shape[0]} rows, "
            f"but '{labels}' has {target.shape[0]} labels"
        )
    ctx[save_as] = matrix.gather(1, target).squeeze(1)


@block(
    id="clamp_min",
    name="Clamp Minimum",
    category="Tensor Operation",
    description="Apply an explicit lower bound to a tensor.",
    params={"input": {"type": "slot", "default": "gathered_values"}, "minimum": {"type": "float", "default": 1e-12}, "save_as": {"type": "slot", "default": "clamped_values"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="x'=max(x,c)", formula_ref="explicit numerical lower bound",
)
def clamp_min(ctx: ScratchContext, input: str = "gathered_values", minimum: float = 1e-12, save_as: str = "clamped_values") -> None:
    ctx[save_as] = ctx[input].clamp_min(float(minimum))


@block(
    id="elementwise_power",
    name="Elementwise Power",
    category="Tensor Operation",
    description="Raise each tensor element to an explicit scalar exponent.",
    params={
        "input": {"type": "slot", "default": "clamped_values"},
        "q": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0},
        "save_as": {"type": "slot", "default": "powered_values"},
    },
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="z=x^q", formula_ref="elementwise power operation",
)
def elementwise_power(ctx: ScratchContext, input: str = "clamped_values", q: float = 0.7, save_as: str = "powered_values") -> None:
    ctx[save_as] = ctx[input].pow(float(q))


@block(
    id="affine_transform",
    name="Affine Transform",
    category="Tensor Operation",
    description="Apply scale times input plus bias without changing reduction or gradient semantics.",
    params={"input": {"type": "slot", "default": "powered_values"}, "scale": {"type": "float", "default": 1.0}, "bias": {"type": "float", "default": 0.0}, "save_as": {"type": "slot", "default": "loss_per_sample"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="z=scale*x+bias", formula_ref="explicit affine combination",
)
def affine_transform(ctx: ScratchContext, input: str = "powered_values", scale: float = 1.0, bias: float = 0.0, save_as: str = "loss_per_sample") -> None:
    ctx[save_as] = float(scale) * ctx[input] + float(bias)


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
    target_labels = _class_labels(ctx[labels], probabilities.shape[1], device=probabilities.device, slot=labels)
    target = torch.zeros_like(probabilities).scatter_(1, target_labels.view(-1, 1), 1.0)
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
    scores = ctx[logits]
    log_probabilities = F.log_softmax(scores, dim=-1)
    target_labels = _class_labels(ctx[labels], scores.shape[-1], device=scores.device, slot=labels)
    ce = -log_probabilities.gather(1, target_labels.view(-1, 1)).squeeze(1)
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
    target_labels = _class_labels(ctx[labels], probabilities.shape[1], device=probabilities.device, slot=labels)
    target = torch.full_like(probabilities, float(log_zero)).scatter_(1, target_labels.view(-1, 1), 0.0)
    _save_loss(ctx, -(probabilities * target).sum(dim=1), save_as)


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
    target_labels = _class_labels(ctx[labels], 2, device=losses.device, slot=labels)
    _save_loss(ctx, torch.where(target_labels == 0, zero, one), save_as)


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
    scores = ctx[logits]
    log_probabilities = F.log_softmax(scores, dim=-1)
    all_losses = -log_probabilities
    corrected = torch.linalg.solve(ctx[transition].to(scores), all_losses.transpose(0, 1)).transpose(0, 1)
    target_labels = _class_labels(ctx[labels], corrected.shape[1], device=corrected.device, slot=labels)
    _save_loss(ctx, corrected.gather(1, target_labels.view(-1, 1)).squeeze(1), save_as)
