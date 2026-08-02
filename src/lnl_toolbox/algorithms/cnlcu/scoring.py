from __future__ import annotations

"""CNLCU-S uncertainty-aware lower-bound score."""

import math

import torch
from torch import Tensor


def cnlcu_soft_score(
    robust_mean: Tensor,
    history_length: Tensor,
    effective_selected_count: Tensor,
    sigma_squared: float,
) -> tuple[Tensor, Tensor]:
    """Compute paper Eq. (7) without the released code's extra ReLU."""

    if not torch.is_tensor(robust_mean) or robust_mean.ndim != 1:
        raise ValueError("CNLCU robust_mean must have shape [B]")
    if not robust_mean.is_floating_point():
        raise TypeError("CNLCU robust_mean must be floating point")
    for name, value in (
        ("history_length", history_length),
        ("effective_selected_count", effective_selected_count),
    ):
        if not torch.is_tensor(value) or value.shape != robust_mean.shape:
            raise ValueError(f"CNLCU {name} must align with robust_mean as [B]")
        if value.device != robust_mean.device:
            raise ValueError(f"CNLCU {name} must share the robust_mean device")
    sigma = float(sigma_squared)
    if not math.isfinite(sigma) or not 0.0 < sigma < 1.0:
        raise ValueError("CNLCU sigma_squared must be finite and in (0,1)")
    if not bool(torch.isfinite(robust_mean).all()):
        raise ValueError("CNLCU robust_mean must be finite")
    t = history_length.to(robust_mean.dtype)
    count = effective_selected_count.to(robust_mean.dtype)
    if bool((t <= 0).any()) or bool((count <= 0).any()):
        raise ValueError("CNLCU history length and selected count must be positive")
    denominator = count - sigma
    if bool((denominator <= 0).any()):
        raise ValueError("CNLCU Eq. (7) denominator must be strictly positive")
    bonus = sigma * (t + sigma * torch.log(2.0 * t) / t.square()) / denominator
    score = robust_mean - bonus
    if not bool(torch.isfinite(bonus).all()) or not bool(torch.isfinite(score).all()):
        raise ValueError("CNLCU confidence bonus and score must be finite")
    return score, bonus


__all__ = ["cnlcu_soft_score"]
