from __future__ import annotations

"""LEND Eq. (6) hard agreement rule."""

import torch
from torch import Tensor


def select_lend_samples(noisy_targets: Tensor, diluted_history: Tensor) -> Tensor:
    if not isinstance(noisy_targets, Tensor) or noisy_targets.ndim != 1:
        raise ValueError("LEND noisy targets must have shape [B]")
    if noisy_targets.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError("LEND noisy targets must use integer dtype")
    if not isinstance(diluted_history, Tensor) or diluted_history.ndim != 2:
        raise ValueError("LEND diluted history must have shape [B,C]")
    if diluted_history.shape[0] != noisy_targets.shape[0] or diluted_history.device != noisy_targets.device:
        raise ValueError("LEND targets and diluted history must align")
    if diluted_history.requires_grad or not bool(torch.isfinite(diluted_history).all()):
        raise ValueError("LEND diluted history must be detached and finite")
    if bool((noisy_targets < 0).any()) or bool((noisy_targets >= diluted_history.shape[1]).any()):
        raise ValueError("LEND noisy target is outside the class range")
    return noisy_targets == diluted_history.argmax(dim=1)


__all__ = ["select_lend_samples"]
