"""Small Scratch-native state and accumulator operations."""

from __future__ import annotations

from typing import Any

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("State operations require PyTorch; install the `train` extra.") from exc
    return torch


@block(
    id="create_indexed_history",
    name="Create Indexed History",
    category="State",
    description="Create a stable-index tensor history for persistent per-example values.",
    params={"size": {"type": "int", "default": 0, "min": 0}, "prepared_data": {"type": "value", "default": None}, "width": {"type": "int", "default": 1, "min": 1}, "save_as": {"type": "slot", "default": "indexed_history"}},
    provides=("save_as",), placement=("top",), stage="setup", ui_group="④ 状态更新",
)
def create_indexed_history(ctx: ScratchContext, size: int = 0, prepared_data: Any = None, width: int = 1, save_as: str = "indexed_history") -> None:
    torch = _torch()
    if isinstance(prepared_data, str) and prepared_data in ctx:
        prepared = ctx[prepared_data]
        indices = getattr(prepared, "train_indices", None)
        if indices is not None and len(indices):
            size = int(max(indices)) + 1
    shape = (int(size), int(width))
    ctx[save_as] = {"values": torch.zeros(shape), "seen": torch.zeros(int(size), dtype=torch.bool), "last_epoch": torch.full((int(size),), -1, dtype=torch.long)}


@block(
    id="indexed_history",
    name="Indexed History Update",
    category="State",
    description="Write detached per-example values into a stable-index history.",
    params={"state": {"type": "slot", "default": "indexed_history"}, "indices": {"type": "slot", "default": "indices"}, "values": {"type": "slot", "default": "values"}, "save_as": {"type": "slot", "default": "history_values"}},
    requires=("state", "indices", "values"), provides=("save_as",), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def indexed_history(ctx: ScratchContext, state: str = "indexed_history", indices: str = "indices", values: str = "values", save_as: str = "history_values") -> None:
    state_value = ctx[state]
    rows = ctx[indices].detach().long().cpu()
    incoming_value = ctx[values].detach().float()
    incoming = incoming_value.reshape(incoming_value.shape[0], -1).cpu()
    if int(state_value["values"].shape[1]) != int(incoming.shape[1]):
        raise ValueError("indexed history value width does not match state width")
    state_value["values"][rows] = incoming
    state_value["seen"][rows] = True
    ctx[save_as] = state_value["values"][rows].squeeze(-1).to(ctx[values].device)


@block(
    id="indexed_ema",
    name="Indexed EMA",
    category="State",
    description="Update persistent values by stable index with an epoch-aware exponential moving average.",
    params={"state": {"type": "slot", "default": "indexed_history"}, "indices": {"type": "slot", "default": "indices"}, "values": {"type": "slot", "default": "values"}, "beta": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0}, "epoch": {"type": "slot", "default": "epoch"}, "save_as": {"type": "slot", "default": "history_values"}},
    requires=("state", "indices", "values"), provides=("save_as",), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def indexed_ema(ctx: ScratchContext, state: str = "indexed_history", indices: str = "indices", values: str = "values", beta: float = 0.9, epoch: str = "epoch", save_as: str = "history_values") -> None:
    state_value = ctx[state]
    rows = ctx[indices].detach().long().cpu()
    incoming = ctx[values].detach().float().cpu()
    current_epoch = int(ctx.get(epoch, 0))
    if incoming.ndim == 1:
        incoming = incoming[:, None]
    if int(state_value["values"].shape[1]) != int(incoming.shape[1]):
        raise ValueError("indexed_ema value width does not match state width")
    for row, value in zip(rows.tolist(), incoming):
        if row < 0 or row >= int(state_value["values"].shape[0]):
            raise IndexError(f"indexed history row out of range: {row}")
        if bool(state_value["seen"][row]) and int(state_value["last_epoch"][row]) == current_epoch:
            continue
        if bool(state_value["seen"][row]):
            state_value["values"][row].mul_(float(beta)).add_(value, alpha=1.0 - float(beta))
        else:
            state_value["values"][row].copy_(value)
            state_value["seen"][row] = True
        state_value["last_epoch"][row] = current_epoch
    ctx[save_as] = state_value["values"][rows].to(ctx[values].device)


@block(
    id="agreement_mask",
    name="Agreement Mask",
    category="Sample Selection",
    description="Select rows whose hard label agrees with the argmax of a soft history.",
    params={"labels": {"type": "slot", "default": "labels"}, "history": {"type": "slot", "default": "history_values"}, "save_as": {"type": "slot", "default": "selected_mask"}},
    requires=("labels", "history"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑦ 样本选择",
)
def agreement_mask(ctx: ScratchContext, labels: str = "labels", history: str = "history_values", save_as: str = "selected_mask") -> None:
    ctx[save_as] = ctx[history].detach().argmax(dim=-1).eq(ctx[labels].long())


@block(
    id="masked_mean",
    name="Masked Mean",
    category="Loss",
    description="Reduce selected values with an explicit selected-count or batch-size denominator.",
    params={"values": {"type": "slot", "default": "loss_per_sample"}, "mask": {"type": "slot", "default": "selected_mask"}, "denominator": {"type": "enum", "options": ["selected", "batch"], "default": "selected"}, "empty": {"type": "enum", "options": ["zero", "error"], "default": "zero"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("values", "mask"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def masked_mean(ctx: ScratchContext, values: str = "loss_per_sample", mask: str = "selected_mask", denominator: str = "selected", empty: str = "zero", save_as: str = "loss") -> None:
    tensor = ctx[values]
    selected = tensor[ctx[mask].bool()]
    if selected.numel() == 0:
        if str(empty) == "error":
            raise ValueError("masked_mean received an empty selection")
        ctx[save_as] = tensor.sum() * 0.0
        return
    divisor = selected.numel() if str(denominator) == "selected" else tensor.numel()
    ctx[save_as] = selected.sum() / max(int(divisor), 1)


@block(
    id="create_grouped_accumulator",
    name="Create Grouped Accumulator",
    category="State",
    description="Create class/group-wise sums and counts for cross-batch statistics.",
    params={"groups": {"type": "int", "default": 10, "min": 1}, "width": {"type": "int", "default": 1, "min": 1}, "save_as": {"type": "slot", "default": "grouped_state"}},
    provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="④ 状态更新",
)
def create_grouped_accumulator(ctx: ScratchContext, groups: int = 10, width: int = 1, save_as: str = "grouped_state") -> None:
    torch = _torch()
    ctx[save_as] = {"sums": torch.zeros((int(groups), int(width))), "counts": torch.zeros(int(groups))}


@block(
    id="grouped_accumulator",
    name="Grouped Accumulator",
    category="State",
    description="Accumulate detached values by integer group label.",
    params={"state": {"type": "slot", "default": "grouped_state"}, "groups": {"type": "slot", "default": "groups"}, "values": {"type": "slot", "default": "values"}},
    requires=("state", "groups", "values"), provides=(), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def grouped_accumulator(ctx: ScratchContext, state: str = "grouped_state", groups: str = "groups", values: str = "values") -> None:
    state_value = ctx[state]
    labels = ctx[groups].detach().long().reshape(-1).cpu()
    tensor = ctx[values].detach().float().reshape(labels.numel(), -1).cpu()
    for group in labels.unique().tolist():
        mask = labels == int(group)
        state_value["sums"][int(group)] += tensor[mask].sum(dim=0)
        state_value["counts"][int(group)] += mask.sum()


@block(
    id="finalize_grouped_accumulator",
    name="Finalize Grouped Accumulator",
    category="State",
    description="Publish group-wise means without fabricating values for unseen groups.",
    params={"state": {"type": "slot", "default": "grouped_state"}, "save_as": {"type": "slot", "default": "grouped_means"}},
    requires=("state",), provides=("save_as",), placement=("epoch", "top"), stage="train", ui_group="④ 状态更新",
)
def finalize_grouped_accumulator(ctx: ScratchContext, state: str = "grouped_state", save_as: str = "grouped_means") -> None:
    state_value = ctx[state]
    means = state_value["sums"].clone()
    seen = state_value["counts"] > 0
    means[seen] = means[seen] / state_value["counts"][seen, None]
    ctx[save_as] = means
