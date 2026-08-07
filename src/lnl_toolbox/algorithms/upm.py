from __future__ import annotations

"""Equation-level primitives for UPM (AAAI 2021)."""

import torch
from torch import Tensor
from torch.nn import functional as F

from lnl_toolbox.noise.upm import UPMNoiseState


def estimate_clean_posterior(
    logits: Tensor,
    noisy_targets: Tensor,
    noisy_label_probability: Tensor,
    confusion_probability: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Implement Eq. (8): ``q ∝ h(x) * ((1-eta)e_y + eta psi 1)``.

    The posterior used as the target is detached by the caller/algorithm; the
    construction itself never mutates model gradients.
    """
    if logits.ndim != 2 or noisy_targets.ndim != 1:
        raise ValueError("UPM logits/targets must have shapes [B,C] and [B]")
    if logits.shape[0] != noisy_targets.numel():
        raise ValueError("UPM batch dimensions do not match")
    psi = noisy_label_probability.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    eta = confusion_probability.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if psi.shape != noisy_targets.shape or eta.shape != noisy_targets.shape:
        raise ValueError("UPM psi and eta must have shape [B]")
    if eps <= 0.0:
        raise ValueError("UPM eps must be positive")
    if int(noisy_targets.min()) < 0 or int(noisy_targets.max()) >= logits.shape[1]:
        raise ValueError("UPM noisy target is outside class range")
    h = torch.softmax(logits.detach(), dim=1)
    target = torch.zeros_like(h)
    target.scatter_(1, noisy_targets[:, None], (1.0 - eta)[:, None])
    target = target + eta[:, None] * psi[:, None]
    q = h * target
    return q / q.sum(dim=1, keepdim=True).clamp_min(eps)


def upm_soft_target_objective(logits: Tensor, clean_posterior: Tensor) -> Tensor:
    if logits.shape != clean_posterior.shape or logits.ndim != 2:
        raise ValueError("UPM logits and posterior must have shape [B,C]")
    q = clean_posterior.detach()
    if not bool(torch.isfinite(q).all()) or bool((q < 0.0).any()):
        raise ValueError("UPM posterior must be finite and non-negative")
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(q * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def update_confusion_probabilities_(
    state: UPMNoiseState,
    global_indices: Tensor,
    posterior: Tensor,
    noisy_targets: Tensor,
    *,
    learning_rate: float,
    eps: float = 1e-8,
) -> Tensor:
    """Apply the Eq. (11) ascent step and Eq. (12) projection to ``eta``.

    For a fixed ``q`` and ``psi``, the per-example log likelihood derivative is
    ``sum_j q_j (psi - 1[j=y]) / ((1-eta)1[j=y] + eta psi)``.
    """
    if learning_rate < 0.0 or not torch.isfinite(torch.tensor(learning_rate)):
        raise ValueError("UPM eta learning rate must be finite and non-negative")
    q = posterior.detach()
    psi, eta = state.lookup(global_indices)
    psi = psi.to(device=q.device, dtype=q.dtype)
    eta = eta.to(device=q.device, dtype=q.dtype)
    labels = noisy_targets.to(dtype=torch.long, device=q.device)
    if q.ndim != 2 or q.shape[0] != labels.numel() or q.shape[1] != state.num_classes:
        raise ValueError("UPM posterior shape is invalid")
    denominator = torch.zeros_like(q)
    denominator.scatter_(1, labels[:, None], (1.0 - eta)[:, None])
    denominator = denominator + eta[:, None] * psi[:, None]
    component = torch.where(
        torch.arange(state.num_classes, device=q.device)[None, :] == labels[:, None],
        psi[:, None] - 1.0,
        psi[:, None],
    )
    gradient = (q * component / denominator.clamp_min(eps)).sum(dim=1)
    updated = eta + float(learning_rate) * gradient
    state.update_eta(global_indices, updated)
    return updated.clamp(0.0, 1.0)


__all__ = [
    "estimate_clean_posterior",
    "upm_soft_target_objective",
    "update_confusion_probabilities_",
]
