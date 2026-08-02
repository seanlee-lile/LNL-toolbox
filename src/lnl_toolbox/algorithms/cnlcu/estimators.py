from __future__ import annotations

"""Robust loss estimators for the CNLCU soft and hard variants."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .outliers import lof_retained_mask


def soft_influence(losses: Tensor) -> Tensor:
    """Apply CNLCU Eq. (2), psi(l)=log(1+l+l^2/2)."""

    if not torch.is_tensor(losses) or not losses.is_floating_point():
        raise TypeError("CNLCU soft losses must be a floating-point tensor")
    if not bool(torch.isfinite(losses).all()):
        raise ValueError("CNLCU soft losses must be finite")
    if bool((losses < 0).any()):
        raise ValueError("CNLCU soft losses must be non-negative")
    transformed = torch.log1p(losses + losses.square() / 2.0)
    if not bool(torch.isfinite(transformed).all()):
        raise ValueError("CNLCU soft influence values must be finite")
    return transformed


def soft_robust_mean(
    loss_history: Tensor,
    observed: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return Eq. (3) and each sample's window observation count."""

    if not torch.is_tensor(loss_history) or loss_history.ndim != 2:
        raise ValueError("CNLCU loss history must have shape [B,W]")
    if not loss_history.is_floating_point():
        raise TypeError("CNLCU loss history must use a floating-point dtype")
    if not torch.is_tensor(observed) or observed.shape != loss_history.shape:
        raise ValueError("CNLCU observed mask must align with history as [B,W]")
    if observed.dtype != torch.bool:
        raise TypeError("CNLCU observed mask must use torch.bool")
    if observed.device != loss_history.device:
        raise ValueError("CNLCU history and observed mask must share a device")
    selected = loss_history[observed]
    if selected.numel() and not bool(torch.isfinite(selected).all()):
        raise ValueError("observed CNLCU history values must be finite")
    lengths = observed.sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("every CNLCU sample must have at least one history value")
    transformed = soft_influence(loss_history)
    means = (transformed * observed).sum(dim=1) / lengths.to(loss_history.dtype)
    if not bool(torch.isfinite(means).all()):
        raise ValueError("CNLCU soft robust means must be finite")
    return means, lengths


@dataclass(frozen=True)
class HardEstimate:
    robust_mean: Tensor
    observation_count: Tensor
    outlier_count: Tensor
    retained_count: Tensor
    retained_mask: Tensor


class HardRobustLossEstimator:
    """Implement Eq. (4) with corrected released-code-inspired LOF labels."""

    def __init__(
        self,
        *,
        n_neighbors: int,
        contamination: float,
        minimum_observations: int,
    ) -> None:
        self.n_neighbors = int(n_neighbors)
        self.contamination = float(contamination)
        self.minimum_observations = int(minimum_observations)

    def estimate(self, loss_history: Tensor, observed: Tensor) -> HardEstimate:
        if not torch.is_tensor(loss_history) or loss_history.ndim != 2:
            raise ValueError("CNLCU-H loss history must have shape [B,W]")
        if not loss_history.is_floating_point():
            raise TypeError("CNLCU-H loss history must be floating point")
        if loss_history.requires_grad:
            raise ValueError("CNLCU-H loss history must be detached")
        if not torch.is_tensor(observed) or observed.shape != loss_history.shape:
            raise ValueError("CNLCU-H observed mask must align with history")
        if observed.dtype != torch.bool or observed.device != loss_history.device:
            raise TypeError("CNLCU-H observed mask must be bool on the history device")
        retained = lof_retained_mask(
            loss_history,
            observed,
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            minimum_observations=self.minimum_observations,
        )
        observation_count = observed.sum(dim=1).to(torch.int64)
        retained_count = retained.sum(dim=1).to(torch.int64)
        outlier_count = observation_count - retained_count
        if bool((observation_count <= 0).any()):
            raise ValueError("CNLCU-H observation count must be positive")
        if bool((outlier_count < 0).any()) or bool((retained_count <= 0).any()):
            raise ValueError("CNLCU-H retained and outlier counts are invalid")
        if not torch.equal(retained_count, observation_count - outlier_count):
            raise RuntimeError("CNLCU-H retained-count identity failed")
        if bool((retained & ~observed).any()):
            raise RuntimeError("CNLCU-H retained mask included padding")
        retained_values = loss_history[retained]
        if not bool(torch.isfinite(retained_values).all()) or bool(
            (retained_values < 0).any()
        ):
            raise ValueError("retained CNLCU-H losses must be finite and non-negative")
        total = (loss_history.to(torch.float64) * retained).sum(dim=1)
        robust_mean = total / retained_count.to(torch.float64)
        if not bool(torch.isfinite(robust_mean).all()):
            raise ValueError("CNLCU-H robust means must be finite")
        return HardEstimate(
            robust_mean=robust_mean,
            observation_count=observation_count,
            outlier_count=outlier_count,
            retained_count=retained_count,
            retained_mask=retained,
        )


__all__ = [
    "HardEstimate",
    "HardRobustLossEstimator",
    "soft_influence",
    "soft_robust_mean",
]
