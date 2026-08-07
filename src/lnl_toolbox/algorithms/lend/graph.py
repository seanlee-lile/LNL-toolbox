from __future__ import annotations

"""Batch-local feature graph from LEND Eq. (1)."""

import torch
from torch import Tensor
import torch.nn.functional as functional


def _validate(features: Tensor, indices: Tensor, k: int, gamma: float) -> None:
    if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("LEND features must have non-empty floating shape [B,D] with B >= 2")
    if not torch.is_floating_point(features) or not bool(torch.isfinite(features).all()):
        raise ValueError("LEND features must be finite floating values")
    if features.requires_grad:
        raise ValueError("LEND graph features must be detached")
    if not isinstance(indices, Tensor) or indices.ndim != 1 or indices.shape[0] != features.shape[0]:
        raise ValueError("LEND stable sample indices must align as [B]")
    if indices.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError("LEND stable sample indices must use integer dtype")
    if indices.device != features.device or torch.unique(indices).numel() != indices.numel():
        raise ValueError("LEND stable sample indices must be unique and on the feature device")
    if not 1 <= k < features.shape[0]:
        raise ValueError("LEND graph requires 1 <= k < batch size")
    if not torch.isfinite(torch.tensor(gamma)) or gamma <= 0:
        raise ValueError("LEND gamma must be finite and positive")


def build_lend_similarity(features: Tensor, sample_indices: Tensor, *, k: int,
                          gamma: float, metric: str,
                          normalize_features: bool) -> Tensor:
    """Return directed non-negative adjacency A with deterministic kNN ties."""

    features = features.detach()
    sample_indices = sample_indices.detach()
    _validate(features, sample_indices, k, gamma)
    metric = str(metric).lower()
    if metric not in {"inner_product", "cosine", "euclidean"}:
        raise ValueError("unsupported LEND neighbor metric")
    vectors = functional.normalize(features, dim=1) if normalize_features else features
    inner = vectors @ vectors.transpose(0, 1)
    if metric == "inner_product":
        ranking = inner
    elif metric == "cosine":
        normalized = functional.normalize(features, dim=1)
        ranking = normalized @ normalized.transpose(0, 1)
    else:
        ranking = -torch.cdist(features, features)
    ranking = ranking.clone()
    ranking.fill_diagonal_(float("-inf"))
    adjacency = torch.zeros_like(inner)
    # First establish the stable-index order, then use stable score sorting so
    # equal similarities are resolved by global identity rather than position.
    identity_order = torch.argsort(sample_indices, stable=True)
    for row in range(features.shape[0]):
        ordered_scores = ranking[row, identity_order]
        by_score = torch.argsort(ordered_scores, descending=True, stable=True)
        neighbors = identity_order[by_score[:k]]
        adjacency[row, neighbors] = inner[row, neighbors].clamp_min(0).pow(gamma)
    adjacency.fill_diagonal_(0)
    if not bool(torch.isfinite(adjacency).all()) or bool((adjacency < 0).any()):
        raise ValueError("LEND adjacency must be finite and non-negative")
    return adjacency.detach()


def normalize_lend_graph(adjacency: Tensor, *, zero_degree_policy: str = "error") -> Tensor:
    """Apply W=D^-1/2(A^T A)D^-1/2 without extra row normalization."""

    if not isinstance(adjacency, Tensor) or adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("LEND adjacency must be square [B,B]")
    if not torch.is_floating_point(adjacency) or adjacency.requires_grad:
        raise ValueError("LEND adjacency must be detached floating values")
    if not bool(torch.isfinite(adjacency).all()) or bool((adjacency < 0).any()):
        raise ValueError("LEND adjacency must be finite and non-negative")
    if zero_degree_policy != "error":
        raise ValueError("unsupported LEND zero-degree policy")
    product = adjacency.transpose(0, 1) @ adjacency
    degree = product.sum(dim=1)
    if bool((degree <= 0).any()):
        raise ValueError("LEND normalized graph has a non-positive degree")
    inverse = degree.rsqrt()
    graph = inverse[:, None] * product * inverse[None, :]
    if not torch.allclose(graph, graph.transpose(0, 1), atol=1e-6, rtol=1e-5):
        raise ValueError("LEND normalized graph must be symmetric")
    return graph.detach()


__all__ = ["build_lend_similarity", "normalize_lend_graph"]
