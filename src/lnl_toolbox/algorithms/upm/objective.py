from __future__ import annotations

"""Paper equations for UPM predicting and updating steps."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_inputs(
    clean_probabilities: Tensor,
    noisy_targets: Tensor,
    psi: Tensor,
    eta: Tensor,
) -> tuple[int, int]:
    if not torch.is_tensor(clean_probabilities) or clean_probabilities.ndim != 2:
        raise ValueError("UPM clean probabilities must have shape [B, C]")
    batch, classes = clean_probabilities.shape
    if batch == 0 or classes < 2 or not clean_probabilities.is_floating_point():
        raise ValueError("UPM clean probabilities must be floating [B, C], C >= 2")
    if noisy_targets.shape != (batch,) or noisy_targets.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    }:
        raise ValueError("UPM noisy targets must be integer [B]")
    for name, value in (("psi", psi), ("eta", eta)):
        if value.shape != (batch,) or not value.is_floating_point():
            raise ValueError(f"UPM {name} must be floating [B]")
        if value.device != clean_probabilities.device:
            raise ValueError(f"UPM {name} must share the probability device")
        if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
            raise ValueError(f"UPM {name} must be finite and in [0, 1]")
    if noisy_targets.device != clean_probabilities.device:
        raise ValueError("UPM targets and probabilities must share a device")
    if bool((noisy_targets < 0).any()) or bool((noisy_targets >= classes).any()):
        raise ValueError("UPM noisy targets are outside the class range")
    if not bool(torch.isfinite(clean_probabilities).all()) or bool((clean_probabilities < 0).any()):
        raise ValueError("UPM clean probabilities must be finite and non-negative")
    if not torch.allclose(clean_probabilities.sum(1), torch.ones(batch, device=clean_probabilities.device, dtype=clean_probabilities.dtype), atol=1e-6, rtol=1e-5):
        raise ValueError("UPM clean probability rows must sum to one")
    return batch, classes


def predict_true_posterior(
    clean_probabilities: Tensor,
    noisy_targets: Tensor,
    psi: Tensor,
    eta: Tensor,
) -> Tensor:
    """Compute detached true-label posterior ``q`` from paper Eq. (8)."""

    clean = clean_probabilities.detach()
    psi = psi.detach()
    eta = eta.detach()
    batch, classes = _validate_inputs(clean, noisy_targets, psi, eta)
    one_hot = F.one_hot(noisy_targets.to(torch.long), classes).to(clean.dtype)
    factor = (1.0 - eta[:, None]) * one_hot + eta[:, None] * psi[:, None]
    unnormalized = clean * factor
    denominator = unnormalized.sum(1, keepdim=True)
    if not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any()):
        raise ValueError("UPM Eq. (8) normalization must be finite and positive")
    result = (unnormalized / denominator).detach()
    if result.shape != (batch, classes) or not bool(torch.isfinite(result).all()):
        raise ValueError("UPM Eq. (8) produced an invalid posterior")
    return result


def update_confusing_probability(
    eta: Tensor,
    q: Tensor,
    noisy_targets: Tensor,
    psi: Tensor,
    *,
    learning_rate: float,
    epsilon: float = 1e-4,
) -> Tensor:
    """Apply explicit projected gradient ascent from paper Eq. (11)-(12)."""

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("UPM eta learning rate must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("UPM eta epsilon must be finite and positive")
    q = q.detach()
    eta = eta.detach()
    psi = psi.detach()
    batch, classes = _validate_inputs(q, noisy_targets, psi, eta)
    if q.requires_grad:
        raise ValueError("UPM q must be detached")
    one_hot = F.one_hot(noisy_targets.to(torch.long), classes).to(q.dtype)
    bracket = torch.ones_like(q) + (psi * eta - eta - 1.0)[:, None] * one_hot
    numerator = (bracket * q).sum(1)
    with torch.no_grad():
        updated = (eta + learning_rate * numerator / (eta + epsilon)).clamp(0.0, 1.0)
    if updated.shape != (batch,) or not bool(torch.isfinite(updated).all()):
        raise ValueError("UPM Eq. (11) produced invalid eta")
    return updated.detach()


def soft_target_cross_entropy(logits: Tensor, q: Tensor) -> Tensor:
    """Return paper Eq. (10)/(13) as per-sample soft-target CE [B]."""

    if not torch.is_tensor(logits) or logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("UPM logits must be floating [B, C]")
    if q.shape != logits.shape or not q.is_floating_point() or q.device != logits.device:
        raise ValueError("UPM q must be floating [B, C] on the logits device")
    if q.requires_grad or not bool(torch.isfinite(q).all()):
        raise ValueError("UPM q must be detached and finite")
    return -(q * F.log_softmax(logits, dim=1)).sum(1)


__all__ = [
    "predict_true_posterior", "soft_target_cross_entropy",
    "update_confusing_probability",
]
