from __future__ import annotations

"""Trainable, per-sample classification objectives for LNL experiments."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_inputs(logits: Tensor, targets: Tensor) -> None:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [batch, classes], got {tuple(logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(
            f"targets must have shape [{logits.shape[0]}], got {tuple(targets.shape)}"
        )
    if targets.dtype != torch.long:
        raise TypeError(f"targets must use torch.long, got {targets.dtype}")


def validate_per_sample_loss(values: Tensor, batch_size: int) -> Tensor:
    """Enforce the toolbox loss contract and return the validated tensor."""

    if not torch.is_tensor(values):
        raise TypeError("A toolbox loss must return a torch.Tensor")
    if values.shape != (batch_size,):
        raise ValueError(
            f"A toolbox loss must return shape [{batch_size}], got {tuple(values.shape)}"
        )
    return values


def _target_log_probability(logits: Tensor, targets: Tensor) -> Tensor:
    _validate_inputs(logits, targets)
    return F.log_softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)


class CrossEntropyLoss(nn.Module):
    """Cross entropy with an explicit per-sample output contract."""

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        _validate_inputs(logits, targets)
        return F.cross_entropy(logits, targets, reduction="none")


class GeneralizedCrossEntropyLoss(nn.Module):
    """Generalized cross entropy interpolating between CE and MAE-like risk."""

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        if not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1")
        self.q = float(q)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_p_y = _target_log_probability(logits, targets)
        return -torch.expm1(self.q * log_p_y) / self.q


class NormalizedCrossEntropyLoss(nn.Module):
    """Cross entropy normalized over every hypothetical class target."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        self.eps = float(eps)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        _validate_inputs(logits, targets)
        all_ce = -F.log_softmax(logits, dim=1)
        target_ce = all_ce.gather(1, targets[:, None]).squeeze(1)
        return target_ce / all_ce.sum(dim=1).clamp_min(self.eps)


class MeanAbsoluteErrorLoss(nn.Module):
    """Classification MAE expressed through the observed-label probability."""

    def __init__(self, scale: float = 2.0) -> None:
        super().__init__()
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        self.scale = float(scale)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        p_y = _target_log_probability(logits, targets).exp()
        return self.scale * (1.0 - p_y)


class ReverseCrossEntropyLoss(nn.Module):
    """Stable one-hot RCE using ``-log_zero * (1 - p_y)``."""

    def __init__(self, log_zero: float = -4.0) -> None:
        super().__init__()
        if not math.isfinite(log_zero) or log_zero >= 0.0:
            raise ValueError("log_zero must be a finite negative value")
        self.log_zero = float(log_zero)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        p_y = _target_log_probability(logits, targets).exp()
        return -self.log_zero * (1.0 - p_y)


class ActivePassiveLoss(nn.Module):
    """Paper-faithful P0 composition of NCE with MAE or RCE."""

    def __init__(
        self,
        active: nn.Module,
        passive: nn.Module,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if (
            not math.isfinite(alpha)
            or not math.isfinite(beta)
            or alpha <= 0.0
            or beta <= 0.0
        ):
            raise ValueError("alpha and beta must be finite and strictly positive")
        if not isinstance(active, NormalizedCrossEntropyLoss):
            raise TypeError("P0 APL active loss must be NormalizedCrossEntropyLoss")
        if not isinstance(passive, (MeanAbsoluteErrorLoss, ReverseCrossEntropyLoss)):
            raise TypeError(
                "P0 APL passive loss must be MeanAbsoluteErrorLoss or ReverseCrossEntropyLoss"
            )
        self.active = active
        self.passive = passive
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        batch_size = int(targets.shape[0])
        active = validate_per_sample_loss(self.active(logits, targets), batch_size)
        passive = validate_per_sample_loss(self.passive(logits, targets), batch_size)
        return self.alpha * active + self.beta * passive
