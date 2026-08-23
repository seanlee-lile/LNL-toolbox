"""Co-teaching's peer selection and cross-update preparation."""

from __future__ import annotations

from typing import Any

from ...context import ScratchContext
from ...registry import block


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Co-teaching blocks require PyTorch; install the `train` extra.") from exc
    return torch


@block(
    id="peer_exchange",
    name="Co-teaching Peer Exchange",
    category="Paper Specific",
    description="Each peer selects its own small-loss examples for the other peer.",
    params={
        "loss_a": {"type": "slot", "default": "loss_a_per_sample"},
        "loss_b": {"type": "slot", "default": "loss_b_per_sample"},
        "keep_rate": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0},
        "selected_a_as": {"type": "slot", "default": "selected_a"},
        "selected_b_as": {"type": "slot", "default": "selected_b"},
    },
    requires=("loss_a", "loss_b"),
    provides=("selected_a_as", "selected_b_as"),
)
def peer_exchange(
    ctx: ScratchContext,
    loss_a: str = "loss_a_per_sample",
    loss_b: str = "loss_b_per_sample",
    keep_rate: float = 0.8,
    selected_a_as: str = "selected_a",
    selected_b_as: str = "selected_b",
) -> None:
    torch = _torch()
    values_a, values_b = ctx[loss_a].reshape(-1), ctx[loss_b].reshape(-1)
    if values_a.shape != values_b.shape:
        raise ValueError("peer losses must have the same batch shape")
    count = max(1, min(values_a.numel(), int(torch.ceil(torch.tensor(values_a.numel() * float(keep_rate))).item())))
    ctx[selected_a_as] = torch.argsort(values_a, stable=True)[:count]
    ctx[selected_b_as] = torch.argsort(values_b, stable=True)[:count]


@block(
    id="select_by_indices",
    name="Select Loss by Peer Indices",
    category="Paper Specific",
    description="Reduce a peer's loss using the indices selected by the other peer.",
    params={"loss": {"type": "slot", "required": True}, "indices": {"type": "slot", "required": True}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("loss", "indices"),
    provides=("save_as",),
)
def select_by_indices(ctx: ScratchContext, loss: str, indices: str, save_as: str = "loss") -> None:
    ctx[save_as] = ctx[loss].reshape(-1)[ctx[indices]].mean()
