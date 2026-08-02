from __future__ import annotations

"""Paper-exact soft robust loss estimator for CNLCU-S."""

import torch
from torch import Tensor


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


__all__ = ["soft_influence", "soft_robust_mean"]
