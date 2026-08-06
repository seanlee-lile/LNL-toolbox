from __future__ import annotations

"""Paper-level LEND label dilution and selection primitives.

No author-maintained implementation was found for LEND; this module follows
the paper's feature-neighbor diffusion description and keeps the graph local
to the supplied batch.
"""

import torch
from torch import Tensor

from .base import SelectionResult


def dilute_labels(
    features: Tensor,
    noisy_targets: Tensor,
    *,
    neighbors: int = 10,
    gamma: float = 1.0,
    diffusion_alpha: float = 0.99,
    diffusion_steps: int = 10,
    num_classes: int | None = None,
) -> Tensor:
    if features.ndim != 2 or noisy_targets.ndim != 1 or features.shape[0] != noisy_targets.numel():
        raise ValueError("LEND features/targets must have shapes [B,D] and [B]")
    n = features.shape[0]
    if n < 2 or not 1 <= int(neighbors) < n:
        raise ValueError("LEND neighbors must be in [1, B-1]")
    if gamma <= 0.0 or not 0.0 <= diffusion_alpha <= 1.0 or diffusion_steps <= 0:
        raise ValueError("invalid LEND diffusion parameters")
    normalized = F_normalize(features.detach())
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-float("inf"))
    values, indices = torch.topk(similarity, int(neighbors), dim=1)
    weights = torch.softmax(float(gamma) * values, dim=1)
    width = int(num_classes) if num_classes is not None else int(noisy_targets.max().item()) + 1
    if width <= int(noisy_targets.max().item()):
        raise ValueError("LEND num_classes is smaller than an observed target")
    one_hot = torch.nn.functional.one_hot(noisy_targets.to(torch.long), num_classes=width).to(features.dtype)
    diluted = one_hot
    for _ in range(int(diffusion_steps)):
        propagated = (weights[..., None] * diluted[indices]).sum(dim=1)
        diluted = float(diffusion_alpha) * propagated + (1.0 - float(diffusion_alpha)) * one_hot
    return diluted / diluted.sum(dim=1, keepdim=True).clamp_min(1e-8)


def F_normalize(value: Tensor) -> Tensor:
    return value / value.norm(dim=1, keepdim=True).clamp_min(1e-8)


class LENDSelector:
    def __init__(self, *, neighbors: int = 10, gamma: float = 1.0, diffusion_alpha: float = 0.99, diffusion_steps: int = 10, num_classes: int | None = None) -> None:
        self.neighbors = int(neighbors)
        self.gamma = float(gamma)
        self.diffusion_alpha = float(diffusion_alpha)
        self.diffusion_steps = int(diffusion_steps)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.last_soft_labels: Tensor | None = None

    def select(self, *, features: Tensor, noisy_targets: Tensor, soft_labels: Tensor | None = None) -> SelectionResult:
        diluted = dilute_labels(features, noisy_targets, neighbors=self.neighbors, gamma=self.gamma, diffusion_alpha=self.diffusion_alpha, diffusion_steps=self.diffusion_steps, num_classes=self.num_classes)
        self.last_soft_labels = diluted.detach()
        selected = diluted.argmax(dim=1).eq(noisy_targets.to(torch.long))
        return SelectionResult(selected_mask=selected, metrics={"selected_ratio": float(selected.float().mean().item())})


__all__ = ["LENDSelector", "dilute_labels"]
