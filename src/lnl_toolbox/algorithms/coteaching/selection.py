from __future__ import annotations

"""Paper-method selection primitives for the Co-teaching workflow."""

import math
from numbers import Integral, Real

import torch
from torch import Tensor


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def determine_keep_count(batch_size: int, remember_rate: float) -> int:
    """Use the released Co-teaching implementation's floor convention."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be an integer")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(remember_rate, bool) or not isinstance(remember_rate, Real):
        raise TypeError("remember_rate must be a real number")
    rate = float(remember_rate)
    if not math.isfinite(rate) or not 0.0 < rate <= 1.0:
        raise ValueError("remember_rate must be finite and in (0, 1]")
    return max(1, math.floor(int(batch_size) * rate))


def stable_small_loss_mask(
    losses: Tensor,
    sample_indices: Tensor,
    keep_count: int,
) -> Tensor:
    """Select exact top-k small losses with stable global-index tie breaking."""

    if not isinstance(losses, Tensor) or losses.ndim != 1 or losses.numel() == 0:
        raise ValueError("Co-teaching losses must be a non-empty [B] tensor")
    if not torch.is_floating_point(losses):
        raise ValueError("Co-teaching losses must use a floating-point dtype")
    if losses.requires_grad:
        raise ValueError("Co-teaching selection losses must be detached")
    if not bool(torch.isfinite(losses).all().item()):
        raise ValueError("Co-teaching selection losses must be finite")
    if not isinstance(sample_indices, Tensor) or sample_indices.shape != losses.shape:
        raise ValueError("sample_indices must align with Co-teaching losses as [B]")
    if sample_indices.dtype not in _INTEGER_DTYPES:
        raise ValueError("sample_indices must use an integer dtype")
    if sample_indices.device != losses.device:
        raise ValueError("sample_indices and Co-teaching losses must share a device")
    if torch.unique(sample_indices).numel() != sample_indices.numel():
        raise ValueError("sample_indices must be unique within the batch")
    if isinstance(keep_count, bool) or not isinstance(keep_count, Integral):
        raise TypeError("keep_count must be an integer")
    if not 1 <= int(keep_count) <= losses.numel():
        raise ValueError("keep_count must be in [1, batch_size]")

    index_order = torch.argsort(sample_indices, stable=True)
    loss_order = torch.argsort(losses[index_order], stable=True)
    chosen = index_order[loss_order[: int(keep_count)]]
    mask = torch.zeros_like(losses, dtype=torch.bool)
    mask[chosen] = True
    return mask
