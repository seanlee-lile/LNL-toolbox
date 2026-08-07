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


def cnlcu_hard_score(
    robust_mean: Tensor,
    observation_count: Tensor,
    outlier_count: Tensor,
    retained_count: Tensor,
    effective_selected_count: Tensor,
    tau_min: float,
    loss_upper_bound: float,
) -> tuple[Tensor, Tensor]:
    """Compute paper Eq. (8) without the released code's extra ReLU."""

    if not torch.is_tensor(robust_mean) or robust_mean.ndim != 1:
        raise ValueError("CNLCU-H robust_mean must have shape [B]")
    if not robust_mean.is_floating_point():
        raise TypeError("CNLCU-H robust_mean must be floating point")
    tensors = {
        "observation_count": observation_count,
        "outlier_count": outlier_count,
        "retained_count": retained_count,
        "effective_selected_count": effective_selected_count,
    }
    for name, value in tensors.items():
        if not torch.is_tensor(value) or value.shape != robust_mean.shape:
            raise ValueError(f"CNLCU-H {name} must align with robust_mean as [B]")
        if value.device != robust_mean.device:
            raise ValueError(f"CNLCU-H {name} must share the robust_mean device")
    tau, bound = float(tau_min), float(loss_upper_bound)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("CNLCU-H tau_min must be finite and positive")
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("CNLCU-H loss_upper_bound must be finite and positive")
    if not bool(torch.isfinite(robust_mean).all()):
        raise ValueError("CNLCU-H robust_mean must be finite")
    t = observation_count.to(torch.float64)
    t_o = outlier_count.to(torch.float64)
    retained = retained_count.to(torch.float64)
    count = effective_selected_count.to(torch.float64)
    if bool((t <= 0).any()) or bool((t_o < 0).any()) or bool((t_o >= t).any()):
        raise ValueError("CNLCU-H requires 0 <= outlier_count < observation_count")
    if not torch.equal(retained_count, observation_count - outlier_count):
        raise ValueError("CNLCU-H retained_count must equal t - t_o")
    if bool((retained <= 0).any()) or bool((count <= 0).any()):
        raise ValueError("CNLCU-H retained and selected counts must be positive")
    factor = (
        2.0
        * math.sqrt(2.0 * tau)
        * bound
        * (t + math.sqrt(2.0) * t_o)
        / (retained * torch.sqrt(t))
    )
    bonus = factor * torch.sqrt(torch.log(4.0 * t) / count)
    score = robust_mean.to(torch.float64) - bonus
    if not bool(torch.isfinite(bonus).all()) or not bool(torch.isfinite(score).all()):
        raise ValueError("CNLCU-H confidence bonus and score must be finite")
    return score, bonus


__all__ = ["cnlcu_hard_score", "cnlcu_soft_score"]
