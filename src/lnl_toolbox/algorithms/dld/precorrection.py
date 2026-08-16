from __future__ import annotations

"""Train-only label pre-correction for paper-oriented DLD."""

from dataclasses import dataclass
import warnings

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


PARTITION_CLEAN = 0
PARTITION_NOISY = 1
PARTITION_HARD = 2


def _load_gmm():
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError(
            "DLD pre-correction requires the optional training dependency; "
            'install with python -m pip install -e ".[train]"'
        ) from exc
    return GaussianMixture, ConvergenceWarning


def _indices(value: Tensor, size: int, owner: str) -> Tensor:
    if not torch.is_tensor(value) or value.shape != (size,) or value.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    }:
        raise ValueError(f"{owner} must be an integer tensor [N]")
    if torch.unique(value).numel() != size:
        raise ValueError(f"{owner} must be unique")
    return value


@dataclass(frozen=True)
class NeighborDistributionResult:
    probabilities: Tensor
    neighbor_indices: Tensor
    neighbor_values: Tensor
    weights: Tensor
    unnormalized_weights: Tensor

    @property
    def distances(self) -> Tensor:
        """Legacy alias for callers of the original distance-only API."""
        return self.neighbor_values


def weighted_neighbor_distribution(
    query_features: Tensor,
    reference_features: Tensor,
    reference_targets: Tensor,
    query_indices: Tensor,
    reference_indices: Tensor,
    *,
    num_classes: int,
    k: int,
    metric: str,
    delta: float,
    self_neighbor: str = "include",
    query_chunk_size: int | None = None,
) -> NeighborDistributionResult:
    if query_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("DLD KNN features must have shape [N, D]")
    if query_features.shape[1] != reference_features.shape[1]:
        raise ValueError("DLD query and reference feature widths differ")
    if query_features.device != reference_features.device:
        raise ValueError("DLD KNN features must share a device")
    if not torch.is_floating_point(query_features) or not torch.is_floating_point(reference_features):
        raise ValueError("DLD KNN features must be floating tensors")
    if not bool(torch.isfinite(query_features).all()) or not bool(torch.isfinite(reference_features).all()):
        raise ValueError("DLD KNN features must be finite")
    n, m = query_features.shape[0], reference_features.shape[0]
    _indices(query_indices, n, "DLD query_indices")
    _indices(reference_indices, m, "DLD reference_indices")
    if reference_targets.shape != (m,) or reference_targets.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    }:
        raise ValueError("DLD reference_targets must be integer [M]")
    if reference_targets.device != query_features.device or reference_indices.device != query_features.device or query_indices.device != query_features.device:
        raise ValueError("DLD KNN inputs must share a device")
    if num_classes < 2 or bool((reference_targets < 0).any()) or bool((reference_targets >= num_classes).any()):
        raise ValueError("DLD reference targets are outside the class range")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0 or k >= m:
        raise ValueError("DLD k must satisfy 0 < k < reference sample count")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("DLD KNN delta must be finite and positive")
    metric = str(metric).lower()
    if metric in {"cosine_distance", "cosine_similarity"}:
        reference = F.normalize(reference_features, dim=1)
    elif metric != "euclidean":
        raise ValueError(
            "DLD metric must be cosine_similarity, cosine_distance, or euclidean"
        )
    if self_neighbor not in {"include", "exclude"}:
        raise ValueError("DLD self_neighbor must be include or exclude")
    if query_chunk_size is None:
        query_chunk_size = n
    if (
        isinstance(query_chunk_size, bool)
        or not isinstance(query_chunk_size, int)
        or query_chunk_size <= 0
    ):
        raise ValueError("DLD query_chunk_size must be a positive integer")

    # Stable global identity is the deterministic tie-break, never the current
    # row position. Each query chunk uses the exact dense reference set and the
    # same two stable sorts as the original implementation. Chunking therefore
    # changes peak allocation only; it does not approximate the KNN search.
    reference_order = torch.argsort(reference_indices, stable=True)
    probability_chunks: list[Tensor] = []
    position_chunks: list[Tensor] = []
    value_chunks: list[Tensor] = []
    weight_chunks: list[Tensor] = []
    unnormalized_weight_chunks: list[Tensor] = []
    for start in range(0, n, query_chunk_size):
        stop = min(start + query_chunk_size, n)
        current_query = query_features[start:stop]
        if metric in {"cosine_distance", "cosine_similarity"}:
            current = F.normalize(current_query, dim=1)
            similarities = current @ reference.T
            values = (
                similarities
                if metric == "cosine_similarity"
                else (1.0 - similarities).clamp_min(0.0)
            )
        else:
            values = torch.cdist(current_query, reference_features)
        if self_neighbor == "exclude":
            same = query_indices[start:stop, None] == reference_indices[None, :]
            excluded_value = (
                float("-inf") if metric == "cosine_similarity" else float("inf")
            )
            values = values.masked_fill(same, excluded_value)
            if bool((torch.isfinite(values).sum(dim=1) < k).any()):
                raise ValueError("DLD has too few non-self neighbors")
        ordered_values = values[:, reference_order]
        value_order = torch.argsort(
            ordered_values,
            dim=1,
            descending=metric == "cosine_similarity",
            stable=True,
        )[:, :k]
        positions = reference_order[value_order]
        selected_values = values.gather(1, positions)
        if not bool(torch.isfinite(selected_values).all()):
            raise ValueError("DLD selected neighbor values are non-finite")
        denominators = selected_values + float(delta)
        if not bool(torch.isfinite(denominators).all()) or bool(
            (denominators <= 0).any()
        ):
            raise ValueError("DLD neighbor weight denominator must be finite and positive")
        unnormalized_weights = 1.0 / denominators
        if not bool(torch.isfinite(unnormalized_weights).all()) or bool(
            (unnormalized_weights <= 0).any()
        ):
            raise ValueError("DLD neighbor weights must be finite and positive")
        weight_sums = unnormalized_weights.sum(dim=1, keepdim=True)
        if not bool(torch.isfinite(weight_sums).all()) or bool((weight_sums <= 0).any()):
            raise ValueError("DLD neighbor weight sums must be finite and positive")
        weights = unnormalized_weights / weight_sums
        neighbor_targets = reference_targets[positions]
        one_hot = F.one_hot(
            neighbor_targets.to(torch.int64), num_classes=num_classes
        ).to(dtype=query_features.dtype)
        probability_chunks.append((one_hot * weights[:, :, None]).sum(dim=1))
        position_chunks.append(positions)
        value_chunks.append(selected_values)
        weight_chunks.append(weights)
        unnormalized_weight_chunks.append(unnormalized_weights)
    probabilities = torch.cat(probability_chunks, dim=0)
    positions = torch.cat(position_chunks, dim=0)
    selected_values = torch.cat(value_chunks, dim=0)
    weights = torch.cat(weight_chunks, dim=0)
    unnormalized_weights = torch.cat(unnormalized_weight_chunks, dim=0)
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError("DLD neighbor distribution is invalid")
    if not torch.allclose(
        probabilities.sum(dim=1), torch.ones(n, device=probabilities.device, dtype=probabilities.dtype), atol=1e-6, rtol=0
    ):
        raise ValueError("DLD neighbor distribution rows must sum to one")
    return NeighborDistributionResult(
        probabilities.detach(),
        reference_indices[positions].detach(),
        selected_values.detach(),
        weights.detach(),
        unnormalized_weights.detach(),
    )


def kl_ps_to_pw(p_w: Tensor, p_s: Tensor) -> Tensor:
    if p_w.shape != p_s.shape or p_w.ndim != 2:
        raise ValueError("DLD p_w and p_s must share shape [N, C]")
    if not bool(torch.isfinite(p_w).all()) or not bool(torch.isfinite(p_s).all()):
        raise ValueError("DLD view distributions must be finite")
    if bool((p_w < 0).any()) or bool((p_s < 0).any()):
        raise ValueError("DLD view distributions must be non-negative")
    for value in (p_w, p_s):
        if not torch.allclose(value.sum(1), torch.ones(value.shape[0], device=value.device, dtype=value.dtype), atol=1e-6, rtol=0):
            raise ValueError("DLD view distribution rows must sum to one")
    # No second softmax is applied. The tiny floor only makes log(0) explicit;
    # positive p_s mass over zero p_w remains a large, finite divergence.
    floor = torch.finfo(p_w.dtype).tiny
    terms = torch.where(
        p_s > 0,
        p_s * (torch.log(p_s.clamp_min(floor)) - torch.log(p_w.clamp_min(floor))),
        torch.zeros_like(p_s),
    )
    result = terms.sum(dim=1)
    if not bool(torch.isfinite(result).all()) or bool((result < -1e-6).any()):
        raise ValueError("DLD KL divergence is invalid")
    return result.clamp_min(0).detach()


@dataclass(frozen=True)
class DLDPartitionResult:
    divergence: Tensor
    partition: Tensor
    p_ws: Tensor
    low_mean: float
    high_mean: float


def partition_samples(
    p_w: Tensor,
    p_s: Tensor,
    noisy_targets: Tensor,
    *,
    random_state: int = 0,
    minimum_mean_separation: float = 1e-6,
) -> DLDPartitionResult:
    divergence = kl_ps_to_pw(p_w, p_s)
    if noisy_targets.shape != (p_w.shape[0],) or noisy_targets.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    }:
        raise ValueError("DLD noisy_targets must be integer [N]")
    if noisy_targets.device != p_w.device:
        raise ValueError("DLD targets and distributions must share a device")
    if p_w.shape[0] < 2:
        raise ValueError("DLD GMM requires at least two samples")
    GaussianMixture, ConvergenceWarning = _load_gmm()
    values = divergence.detach().cpu().to(torch.float64).numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=int(random_state), n_init=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gmm.fit(values)
    if not bool(gmm.converged_):
        raise RuntimeError("DLD divergence GMM did not converge")
    means = np.asarray(gmm.means_, dtype=np.float64).reshape(-1)
    if means.shape != (2,) or not np.isfinite(means).all():
        raise ValueError("DLD divergence GMM produced invalid means")
    low = int(np.argmin(means)); high = 1 - low
    if abs(float(means[high] - means[low])) <= float(minimum_mean_separation):
        raise ValueError("DLD divergence GMM component means are not distinguishable")
    assignments = torch.as_tensor(gmm.predict(values), device=p_w.device)
    p_ws = ((p_w + p_s) * 0.5).detach()
    predicted = p_ws.argmax(dim=1)
    partition = torch.full_like(noisy_targets, PARTITION_HARD, dtype=torch.int64)
    low_mask = assignments == low
    partition[low_mask & (predicted == noisy_targets)] = PARTITION_CLEAN
    partition[low_mask & (predicted != noisy_targets)] = PARTITION_NOISY
    return DLDPartitionResult(
        divergence,
        partition.detach(),
        p_ws,
        float(means[low]),
        float(means[high]),
    )


def construct_y0(
    p_ws: Tensor, noisy_targets: Tensor, partition: Tensor
) -> Tensor:
    if p_ws.ndim != 2 or noisy_targets.shape != (p_ws.shape[0],) or partition.shape != noisy_targets.shape:
        raise ValueError("DLD y0 inputs are misaligned")
    classes = p_ws.shape[1]
    noisy_one_hot = F.one_hot(noisy_targets.to(torch.int64), classes).to(p_ws.dtype)
    corrected = F.one_hot(p_ws.argmax(1), classes).to(p_ws.dtype)
    result = torch.empty_like(p_ws)
    result[partition == PARTITION_CLEAN] = noisy_one_hot[partition == PARTITION_CLEAN]
    result[partition == PARTITION_NOISY] = corrected[partition == PARTITION_NOISY]
    result[partition == PARTITION_HARD] = p_ws[partition == PARTITION_HARD]
    if bool(~torch.isin(partition, torch.tensor([0, 1, 2], device=partition.device)).any()):
        raise ValueError("DLD partition contains an unknown value")
    return result.detach()


def construct_yn(
    p_w: Tensor, p_s: Tensor, noisy_targets: Tensor, partition: Tensor
) -> Tensor:
    if p_w.shape != p_s.shape or p_w.ndim != 2 or noisy_targets.shape != (p_w.shape[0],):
        raise ValueError("DLD yn inputs are misaligned")
    result = torch.zeros_like(p_w)
    noisy_mask = partition == PARTITION_NOISY
    result[noisy_mask] = F.one_hot(
        noisy_targets[noisy_mask].to(torch.int64), p_w.shape[1]
    ).to(p_w.dtype)
    hard_mask = partition == PARTITION_HARD
    difference = (p_w[hard_mask] - p_s[hard_mask]).abs()
    denominator = difference.sum(dim=1, keepdim=True)
    if denominator.numel() and (
        not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any())
    ):
        raise ValueError("DLD hard yn normalization denominator must be finite and positive")
    result[hard_mask] = difference / denominator
    if not bool(torch.isfinite(result).all()):
        raise ValueError("DLD yn is non-finite")
    return result.detach()


__all__ = [
    "DLDPartitionResult",
    "NeighborDistributionResult",
    "PARTITION_CLEAN",
    "PARTITION_HARD",
    "PARTITION_NOISY",
    "construct_y0",
    "construct_yn",
    "kl_ps_to_pw",
    "partition_samples",
    "weighted_neighbor_distribution",
]
