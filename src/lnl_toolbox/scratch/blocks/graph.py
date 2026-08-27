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
    description="Keep a stable top-k neighbor adjacency from a similarity matrix.",
    params={"similarity": {"type": "slot", "default": "similarity"}, "k": {"type": "int", "default": 8, "min": 1}, "gamma": {"type": "float", "default": 1.0, "min": 0.0001}, "save_as": {"type": "slot", "default": "adjacency"}},
    requires=("similarity",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
)
def topk_neighborhood(ctx: ScratchContext, similarity: str = "similarity", k: int = 8, gamma: float = 1.0, save_as: str = "adjacency") -> None:
    torch, _ = _torch()
    values = ctx[similarity].detach().clone()
    values.fill_diagonal_(-torch.inf)
    count = min(int(k), max(int(values.shape[1]) - 1, 1))
    order = torch.argsort(values, dim=1, descending=True, stable=True)[:, :count]
    adjacency = torch.zeros_like(values)
    weights = values.gather(1, order).clamp_min(0).pow(float(gamma))
    adjacency.scatter_(1, order, weights)
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
