from __future__ import annotations

"""Finite-step implementation of LEND label dilution Eq. (3)."""

import math
import torch
from torch import Tensor


def dilute_labels(noisy_onehot: Tensor, graph: Tensor, *, alpha: float,
                  steps: int) -> Tensor:
    if not isinstance(noisy_onehot, Tensor) or noisy_onehot.ndim != 2:
        raise ValueError("LEND noisy one-hot labels must have shape [B,C]")
    if not torch.is_floating_point(noisy_onehot) or noisy_onehot.requires_grad:
        raise ValueError("LEND noisy one-hot labels must be detached floating values")
    if graph.shape != (noisy_onehot.shape[0], noisy_onehot.shape[0]):
        raise ValueError("LEND graph and noisy labels must align")
    if graph.device != noisy_onehot.device or graph.dtype != noisy_onehot.dtype:
        raise ValueError("LEND graph and noisy labels must share dtype and device")
    if not bool(torch.isfinite(noisy_onehot).all()) or bool((noisy_onehot < 0).any()):
        raise ValueError("LEND noisy one-hot labels must be finite and non-negative")
    if not math.isfinite(alpha) or not 0 < alpha < 1 or steps < 1:
        raise ValueError("LEND dilution requires alpha in (0,1) and steps >= 1")
    current = noisy_onehot.detach()
    for _ in range(steps):
        current = alpha * (graph @ current) + (1.0 - alpha) * current
    if not bool(torch.isfinite(current).all()):
        raise ValueError("LEND dilution produced non-finite values")
    return current.detach()


__all__ = ["dilute_labels"]
