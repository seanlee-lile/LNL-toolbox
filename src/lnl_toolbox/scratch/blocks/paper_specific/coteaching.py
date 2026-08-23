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
    id="remember_rate_formula",
    name="Remember-rate Formula",
    category="Paper Specific",
    description="Compute Co-teaching's epoch-wise remember rate from the noise rate and warm-up schedule.",
    params={
        "epoch": {"type": "slot", "default": "epoch"},
        "noise_rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "warmup_epochs": {"type": "int", "default": 10, "min": 0},
        "save_as": {"type": "slot", "default": "remember_rate"},
    },
    requires=("epoch",),
    provides=("save_as",),
    placement=("epoch",), stage="train", ui_group="⑩ 论文专用",
    formula="R(epoch) = 1 - min(epoch / T_k × noise_rate, noise_rate)",
    formula_ref="Co-teaching legacy remember_rate() and paper schedule",
    paper="Co-teaching",
)
def remember_rate_formula(
    ctx: ScratchContext,
    epoch: str = "epoch",
    noise_rate: float = 0.2,
    warmup_epochs: int = 10,
    save_as: str = "remember_rate",
) -> None:
    if int(warmup_epochs) <= 0:
        progress = 1.0
    else:
        progress = min(max(float(ctx[epoch]), 0.0) / int(warmup_epochs), 1.0)
    ctx[save_as] = 1.0 - progress * float(noise_rate)


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
    placement=("batch",), stage="train", ui_group="⑩ 论文专用", beginner_visible=False,
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
    placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="L_A = mean(loss_A[selected_B])", formula_ref="Co-teaching cross-update step", paper="Co-teaching",
)
def select_by_indices(ctx: ScratchContext, loss: str, indices: str, save_as: str = "loss") -> None:
    ctx[save_as] = ctx[loss].reshape(-1)[ctx[indices]].mean()


@block(
    id="cross_select_loss_a_from_b",
    name="[A ← B] Cross-selected Loss",
    category="Paper Specific",
    description="Use B's selected indices to compute A's update loss.",
    params={
        "loss": {"type": "slot", "default": "loss_a_per_sample"},
        "indices": {"type": "slot", "default": "selected_b"},
        "save_as": {"type": "slot", "default": "loss_a"},
    },
    requires=("loss", "indices"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="L_A = mean(loss_A[selected_B])", formula_ref="Co-teaching peer cross-update", paper="Co-teaching",
)
def cross_select_loss_a_from_b(ctx: ScratchContext, loss: str = "loss_a_per_sample", indices: str = "selected_b", save_as: str = "loss_a") -> None:
    select_by_indices(ctx, loss=loss, indices=indices, save_as=save_as)


@block(
    id="cross_select_loss_b_from_a",
    name="[B ← A] Cross-selected Loss",
    category="Paper Specific",
    description="Use A's selected indices to compute B's update loss.",
    params={
        "loss": {"type": "slot", "default": "loss_b_per_sample"},
        "indices": {"type": "slot", "default": "selected_a"},
        "save_as": {"type": "slot", "default": "loss_b"},
    },
    requires=("loss", "indices"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="L_B = mean(loss_B[selected_A])", formula_ref="Co-teaching peer cross-update", paper="Co-teaching",
)
def cross_select_loss_b_from_a(ctx: ScratchContext, loss: str = "loss_b_per_sample", indices: str = "selected_a", save_as: str = "loss_b") -> None:
    select_by_indices(ctx, loss=loss, indices=indices, save_as=save_as)
