from __future__ import annotations

"""VolMinNet Eq. (7) under the Toolbox clean-to-noisy row convention."""

import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def volminnet_objective(
    logits: Tensor,
    noisy_targets: Tensor,
    transition: Tensor,
    *,
    lambda_volume: float,
) -> tuple[Tensor, Mapping[str, float]]:
    if logits.ndim != 2:
        raise ValueError("VolMinNet logits must have shape [B, C]")
    if noisy_targets.shape != (logits.shape[0],):
        raise ValueError("VolMinNet targets must have shape [B]")
    if transition.shape != (logits.shape[1], logits.shape[1]):
        raise ValueError("VolMinNet transition must have shape [C, C]")
    if not torch.is_floating_point(transition) or not bool(torch.isfinite(transition).all().item()):
        raise ValueError("VolMinNet transition must be finite and floating-point")
    if bool((transition <= 0.0).any().item()):
        raise ValueError("VolMinNet transition must be strictly positive")
    if not torch.allclose(
        transition.sum(dim=1),
        torch.ones(logits.shape[1], dtype=transition.dtype, device=transition.device),
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("VolMinNet transition rows must sum to one")
    if not math.isfinite(lambda_volume) or lambda_volume <= 0.0:
        raise ValueError("lambda_volume must be finite and positive")
    if noisy_targets.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise TypeError("VolMinNet targets must use an integer dtype")
    if noisy_targets.numel() and (
        int(noisy_targets.min().item()) < 0
        or int(noisy_targets.max().item()) >= logits.shape[1]
    ):
        raise ValueError("VolMinNet targets are outside the class range")

    sign, logabsdet = torch.linalg.slogdet(transition)
    if (
        not bool(torch.isfinite(sign).item())
        or float(sign.detach().item()) <= 0.0
        or not bool(torch.isfinite(logabsdet).item())
    ):
        raise ValueError("VolMinNet transition determinant must be finite and positive")
    clean_log_probability = F.log_softmax(logits, dim=1)
    noisy_log_probability = torch.logsumexp(
        clean_log_probability[:, :, None] + torch.log(transition)[None, :, :],
        dim=1,
    )
    classification_loss = F.nll_loss(
        noisy_log_probability, noisy_targets.to(dtype=torch.long), reduction="mean"
    )
    objective = classification_loss + float(lambda_volume) * logabsdet
    if not bool(torch.isfinite(objective).item()):
        raise ValueError("VolMinNet objective is non-finite")
    return objective, {
        "classification_loss": float(classification_loss.detach().item()),
        "volume_logdet": float(logabsdet.detach().item()),
        "objective": float(objective.detach().item()),
    }
