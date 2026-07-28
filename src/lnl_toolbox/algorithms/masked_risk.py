from __future__ import annotations

"""Generic class-exclusion risks with explicit candidate masks."""

import torch
from torch import Tensor


def candidate_masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    excluded_classes: Tensor,
) -> Tensor:
    """Return per-sample CE after excluding configured denominator classes."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, C]")
    batch_size, num_classes = logits.shape
    if targets.shape != (batch_size,) or targets.dtype != torch.long:
        raise ValueError("targets must use torch.long with shape [B]")
    if excluded_classes.shape != logits.shape:
        raise ValueError("excluded_classes must have shape [B, C]")
    if excluded_classes.dtype != torch.bool:
        raise ValueError("excluded_classes must use torch.bool")
    if excluded_classes.device != logits.device:
        raise ValueError("excluded_classes must be on the logits device")
    if bool(excluded_classes.gather(1, targets[:, None]).any()):
        raise ValueError("the supervised target class cannot be excluded")
    if bool(excluded_classes.all(dim=1).any()):
        raise ValueError("every sample must retain at least one candidate class")
    masked_logits = logits.masked_fill(excluded_classes, float("-inf"))
    target_logits = masked_logits.gather(1, targets[:, None]).squeeze(1)
    return torch.logsumexp(masked_logits, dim=1) - target_logits


__all__ = ["candidate_masked_cross_entropy"]
