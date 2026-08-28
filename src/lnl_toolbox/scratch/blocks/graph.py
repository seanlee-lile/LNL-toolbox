"""Scratch-native graph operations used by neighborhood-based recipes."""

from __future__ import annotations

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Graph operations require PyTorch; install the `train` extra.") from exc
    return torch, F


@block(
    id="pairwise_similarity",
    name="Pairwise Similarity",
    category="Graph",
    description="Build a detached pairwise similarity matrix from feature rows.",
    params={"features": {"type": "slot", "default": "features"}, "metric": {"type": "enum", "options": ["inner_product", "cosine", "euclidean"], "default": "inner_product"}, "normalize_features": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "similarity"}},
    requires=("features",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
)
def pairwise_similarity(ctx: ScratchContext, features: str = "features", metric: str = "inner_product", normalize_features: bool = False, save_as: str = "similarity") -> None:
    torch, F = _torch()
    values = ctx[features].detach()
    if bool(normalize_features) or str(metric) == "cosine":
        values = F.normalize(values, dim=1)
    if str(metric) in {"inner_product", "cosine"}:
        result = values @ values.T
    elif str(metric) == "euclidean":
        result = -torch.cdist(values, values).square()
    else:
        raise ValueError(f"unsupported similarity metric: {metric}")
    ctx[save_as] = result


@block(
    id="topk_neighborhood",
    name="Top-k Neighborhood",
    category="Graph",
    description="Select stable top-k neighbour indices from a ranking metric. Edge weights are constructed by a separate operation.",
    params={"similarity": {"type": "slot", "default": "similarity"}, "stable_sample_indices": {"type": "slot", "default": "indices"}, "k": {"type": "int", "default": 8, "min": 1}, "save_as": {"type": "slot", "default": "neighbor_indices"}},
    requires=("similarity",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
)
def topk_neighborhood(ctx: ScratchContext, similarity: str = "similarity", stable_sample_indices: str = "indices", k: int = 8, save_as: str = "neighbor_indices") -> None:
    torch, _ = _torch()
    values = ctx[similarity].detach()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("topk_neighborhood expects a square ranking matrix")
    n = int(values.shape[0])
    sample_indices = ctx.get(stable_sample_indices)
    if sample_indices is None:
        sample_indices = torch.arange(n, device=values.device)
    sample_indices = torch.as_tensor(sample_indices, device=values.device).reshape(-1)
    if sample_indices.numel() != n or len(set(sample_indices.detach().cpu().tolist())) != n:
        raise ValueError("stable_sample_indices must be unique and aligned with similarity rows")
    count = min(int(k), max(n - 1, 0))
    selected = torch.empty((n, count), dtype=torch.long, device=values.device)
    for row in range(n):
        candidates = [column for column in range(n) if column != row]
        candidates.sort(key=lambda column: (-float(values[row, column]), int(sample_indices[column])))
        if count:
            selected[row] = torch.as_tensor(candidates[:count], dtype=torch.long, device=values.device)
    ctx[save_as] = selected


@block(
    id="neighbor_edge_weights",
    name="Neighbour Edge Weights",
    category="Graph",
    description="Construct inner-product edge weights for selected neighbours; ranking and weighting remain separate operations.",
    params={"features": {"type": "slot", "default": "features"}, "neighbor_indices": {"type": "slot", "default": "neighbor_indices"}, "gamma": {"type": "float", "default": 1.0, "min": 0.0001}, "normalize_features": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "adjacency"}},
    requires=("features", "neighbor_indices"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
)
def neighbor_edge_weights(ctx: ScratchContext, features: str = "features", neighbor_indices: str = "neighbor_indices", gamma: float = 1.0, normalize_features: bool = False, save_as: str = "adjacency") -> None:
    torch, F = _torch()
    values = ctx[features].detach()
    if values.ndim != 2:
        raise ValueError("neighbor edge weights expect a [N,D] feature matrix")
    if bool(normalize_features):
        values = F.normalize(values, dim=1)
    neighbors = torch.as_tensor(ctx[neighbor_indices], device=values.device, dtype=torch.long)
    if neighbors.ndim != 2 or neighbors.shape[0] != values.shape[0]:
        raise ValueError("neighbor_indices must align with feature rows")
    selected = values[neighbors]
    source = values[:, None, :]
    weights = (source * selected).sum(dim=-1).clamp_min(0).pow(float(gamma))
    adjacency = torch.zeros((values.shape[0], values.shape[0]), dtype=values.dtype, device=values.device)
    rows = torch.arange(values.shape[0], device=values.device)[:, None].expand_as(neighbors)
    adjacency[rows, neighbors] = weights
    ctx[save_as] = adjacency


@block(
    id="normalize_graph",
    name="Normalize Graph",
    category="Graph",
    description="Construct the normalized A-transpose-A propagation graph.",
    params={"adjacency": {"type": "slot", "default": "adjacency"}, "save_as": {"type": "slot", "default": "graph"}},
    requires=("adjacency",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
)
def normalize_graph(ctx: ScratchContext, adjacency: str = "adjacency", save_as: str = "graph") -> None:
    values = ctx[adjacency]
    gram = values.T @ values
    degree = gram.sum(dim=1).clamp_min(1e-12)
    inv_sqrt = degree.rsqrt()
    ctx[save_as] = inv_sqrt[:, None] * gram * inv_sqrt[None, :]


@block(
    id="propagate_labels",
    name="Propagate Labels",
    category="Graph",
    description="Apply finite-step graph diffusion to one-hot observed labels.",
    params={"graph": {"type": "slot", "default": "graph"}, "labels": {"type": "slot", "default": "labels"}, "num_classes": {"type": "int", "default": 10, "min": 2}, "alpha": {"type": "float", "default": 0.99, "min": 0.0001, "max": 0.9999}, "steps": {"type": "int", "default": 10, "min": 1}, "save_as": {"type": "slot", "default": "diluted_labels"}},
    requires=("graph", "labels"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
)
def propagate_labels(ctx: ScratchContext, graph: str = "graph", labels: str = "labels", num_classes: int = 10, alpha: float = 0.99, steps: int = 10, save_as: str = "diluted_labels") -> None:
    torch, F = _torch()
    values = F.one_hot(ctx[labels].long(), int(num_classes)).to(ctx[graph].dtype)
    operator = ctx[graph].to(values)
    for _ in range(int(steps)):
        values = float(alpha) * (operator @ values) + (1.0 - float(alpha)) * values
    ctx[save_as] = values
