from __future__ import annotations

"""Asymmetric co-learning primitives for CA2C."""

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def cross_guidance(
    p_logits: Tensor, n_logits: Tensor, candidate_k: int
) -> tuple[Tensor, Tensor]:
    if p_logits.shape != n_logits.shape or p_logits.ndim != 2:
        raise ValueError("CA2C logits must share shape [B,C]")
    classes = p_logits.shape[1]
    if not 0 < candidate_k < classes:
        raise ValueError("CA2C candidate_k must satisfy 0 < K < C")
    candidate = torch.zeros_like(n_logits, dtype=torch.bool)
    n_order = torch.argsort(
        n_logits.detach(), dim=1, descending=True, stable=True
    )[:, :candidate_k]
    candidate.scatter_(1, n_order, True)
    # The official implementation marks P's top-K classes and then takes the
    # complement over all C classes.  Constructing the bottom C-K directly is
    # not equivalent under tied logits.
    complement = torch.ones_like(p_logits, dtype=torch.bool)
    p_order = torch.argsort(
        p_logits.detach(), dim=1, descending=True, stable=True
    )[:, :candidate_k]
    complement.scatter_(1, p_order, False)
    return candidate, complement


@dataclass
class CandidateMemory:
    global_indices: Tensor
    counts: Tensor

    @classmethod
    def create(cls, global_indices: Tensor, num_classes: int) -> "CandidateMemory":
        indices = global_indices.detach().cpu().long().clone()
        if indices.ndim != 1 or indices.numel() == 0 or torch.unique(indices).numel() != indices.numel():
            raise ValueError("CA2C memory indices must be non-empty and unique")
        if isinstance(num_classes, bool) or int(num_classes) != num_classes or int(num_classes) <= 1:
            raise ValueError("CA2C memory requires at least two classes")
        order = torch.argsort(indices, stable=True)
        return cls(indices[order], torch.zeros(indices.numel(), int(num_classes)))

    def _positions(self, indices: Tensor) -> Tensor:
        requested = indices.detach().cpu().long()
        positions = torch.searchsorted(self.global_indices, requested)
        safe = positions.clamp_max(self.global_indices.numel() - 1)
        if bool((positions >= self.global_indices.numel()).any()) or not torch.equal(self.global_indices[safe], requested):
            raise KeyError("CA2C memory is missing a global sample index")
        return positions

    def update_(self, indices: Tensor, candidate_mask: Tensor) -> None:
        if indices.ndim != 1:
            raise ValueError("CA2C update indices must be one-dimensional")
        positions = self._positions(indices)
        if candidate_mask.shape != (indices.numel(), self.counts.shape[1]):
            raise ValueError("CA2C candidate mask dimensions differ from memory")
        if candidate_mask.dtype != torch.bool:
            raise TypeError("CA2C candidate masks must be boolean")
        self.counts[positions] += candidate_mask.detach().cpu().to(self.counts.dtype)

    def targets(self, indices: Tensor) -> Tensor:
        values = self.counts[self._positions(indices)].to(indices.device)
        total = values.sum(dim=1, keepdim=True)
        if bool((total <= 0).any()):
            raise ValueError("CA2C candidate memory contains an empty row")
        return values / total

    def state_dict(self) -> dict[str, Tensor]:
        return {"global_indices": self.global_indices.clone(), "counts": self.counts.clone()}

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for value in (self.global_indices, self.counts):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CandidateMemory":
        if not isinstance(state, Mapping) or "global_indices" not in state or "counts" not in state:
            raise ValueError("CA2C checkpoint is missing candidate memory state")
        indices = torch.as_tensor(state["global_indices"]).long().clone()
        counts = torch.as_tensor(state["counts"]).float().clone()
        if indices.ndim != 1 or counts.ndim != 2 or counts.shape[0] != indices.numel():
            raise ValueError("CA2C candidate memory state has invalid dimensions")
        if counts.shape[1] <= 1 or not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
            raise ValueError("CA2C candidate memory counts are invalid")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("CA2C candidate memory state contains duplicate indices")
        order = torch.argsort(indices, stable=True)
        return cls(indices[order], counts[order])


def partial_label_objective(
    logits: Tensor,
    soft_targets: Tensor,
    hard_weight: float,
    *,
    confidence: Tensor | None = None,
) -> Tensor:
    if logits.shape != soft_targets.shape:
        raise ValueError("CA2C logits and soft targets must align")
    if confidence is not None and confidence.shape != (logits.shape[0],):
        raise ValueError("CA2C confidence must have shape [B]")
    weight = float(hard_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("CA2C hard_weight must be in [0,1]")
    soft = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    hard = F.cross_entropy(logits, soft_targets.argmax(dim=1), reduction="none")
    losses = weight * hard + (1.0 - weight) * soft
    if confidence is not None:
        if not bool(torch.isfinite(confidence).all()) or bool((confidence < 0).any()):
            raise ValueError("CA2C confidence must be finite and non-negative")
        losses = losses * confidence.to(losses)
    return losses.mean()


def negative_label_objective(logits: Tensor, complementary_mask: Tensor) -> Tensor:
    if logits.shape != complementary_mask.shape:
        raise ValueError("CA2C complementary mask must align with logits")
    count = complementary_mask.sum()
    if int(count.item()) == 0:
        return logits.sum() * 0.0
    probability = torch.softmax(logits, dim=1)
    losses = -torch.log1p(-probability.clamp(max=1.0 - 1e-7))
    # Official CA2C sums the complementary classes for each sample before
    # taking the batch mean.  Averaging over selected entries changes the
    # relative weight when different samples have different mask sizes.
    return (losses * complementary_mask.to(losses.dtype)).sum(dim=1).mean()


__all__ = [
    "CandidateMemory", "cross_guidance", "negative_label_objective",
    "partial_label_objective",
]
