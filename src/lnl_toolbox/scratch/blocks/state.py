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


class _ScratchModelEMA:
    """Model-agnostic EMA state kept with the other Scratch state blocks."""

    def __init__(self, model: Any, momentum: float, update_buffers: bool = False) -> None:
        import copy
        self.model = copy.deepcopy(model)
        self.momentum = float(momentum)
        self.update_buffers = bool(update_buffers)
        self.model.eval()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def update(self, model: Any) -> None:
        torch = _torch()
        with torch.no_grad():
            for target, source in zip(self.model.parameters(), model.parameters()):
                target.mul_(self.momentum).add_(source.detach(), alpha=1.0 - self.momentum)
            if self.update_buffers:
                for target, source in zip(self.model.buffers(), model.buffers()):
                    target.copy_(source)


@block(
    id="create_model_ema",
    name="Create Model EMA",
    category="State",
    description="Create an independent exponential moving-average copy for one model slot.",
    params={"model": {"type": "slot", "default": "model"}, "momentum": {"type": "float", "default": 0.999, "min": 0.0, "max": 1.0}, "update_buffers": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "ema_model"}},
    requires=("model",), provides=("save_as",), placement=("top",), stage="setup", ui_group="④ 状态更新",
)
def create_model_ema(ctx: ScratchContext, model: str = "model", momentum: float = 0.999, update_buffers: bool = False, save_as: str = "ema_model") -> None:
    ctx[save_as] = _ScratchModelEMA(ctx[model], float(momentum), bool(update_buffers))


@block(
    id="update_model_ema",
    name="Update Model EMA",
    category="State",
    description="Update one model EMA from its corresponding live model; parameter and buffer mutation is explicit.",
    params={"model": {"type": "slot", "default": "model"}, "ema": {"type": "slot", "default": "ema_model"}},
    requires=("model", "ema"), provides=(), placement=("batch", "epoch"), stage="train", ui_group="④ 状态更新",
)
def update_model_ema(ctx: ScratchContext, model: str = "model", ema: str = "ema_model") -> None:
    value = ctx[ema]
    if not hasattr(value, "update") or not hasattr(value, "model"):
        raise TypeError("update_model_ema requires a Scratch model EMA container")
    value.update(ctx[model])


@block(
    id="create_indexed_state",
    name="Create Indexed State",
    category="State",
    description="Create a paper-independent state table keyed by stable sample indices.",
    params={"indices": {"type": "slot", "default": "indices"}, "prepared_data": {"type": "value", "default": None}, "size": {"type": "int", "default": 0, "min": 0}, "width": {"type": "int", "default": 1, "min": 1}, "initial_value": {"type": "float", "default": 0.0}, "dtype": {"type": "value", "default": "float32"}, "save_as": {"type": "slot", "default": "state"}},
    provides=("save_as",), placement=("top",), stage="setup", ui_group="④ 状态更新",
)
def create_indexed_state(ctx: ScratchContext, indices: str = "indices", size: int = 0,
                         prepared_data: Any = None, width: int = 1, initial_value: float = 0.0,
                         dtype: Any = "float32", save_as: str = "state") -> None:
    torch = _torch()
    if isinstance(prepared_data, str) and prepared_data in ctx:
        prepared = ctx[prepared_data]
        train_indices = getattr(prepared, "train_indices", None)
        if train_indices is not None and len(train_indices):
            size = max(int(size), int(max(train_indices)) + 1)
    if indices in ctx:
        values = torch.as_tensor(ctx[indices]).reshape(-1)
        if values.numel():
            size = max(int(size), int(values.max().item()) + 1)
    if isinstance(dtype, str):
        try:
            dtype = getattr(torch, dtype.replace("torch.", ""))
        except AttributeError as exc:
            raise ValueError(f"unknown indexed state dtype: {dtype}") from exc
    if not isinstance(dtype, torch.dtype):
        raise TypeError("indexed state dtype must be a torch.dtype or dtype name")
    fill = bool(initial_value) if dtype == torch.bool else initial_value
    ctx[save_as] = {"values": torch.full((int(size), int(width)), fill, dtype=dtype),
                    "seen": torch.zeros(int(size), dtype=torch.bool),
                    "last_epoch": torch.full((int(size),), -1, dtype=torch.long)}


@block(
    id="indexed_read",
    name="Indexed Read",
    category="State",
    description="Read state rows by stable local/sample indices.",
    params={"state": {"type": "slot", "default": "state"}, "indices": {"type": "slot", "default": "indices"}, "save_as": {"type": "slot", "default": "state_values"}},
    requires=("state", "indices"), provides=("save_as",), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def indexed_read(ctx: ScratchContext, state: str = "state", indices: str = "indices", save_as: str = "state_values") -> None:
    table = ctx[state]
    rows = _torch().as_tensor(ctx[indices], dtype=_torch().long).reshape(-1)
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= int(table["values"].shape[0])):
        raise IndexError("indexed_read index out of range")
    ctx[save_as] = table["values"][rows].clone()


@block(
    id="indexed_write",
    name="Indexed Write",
    category="State",
    description="Write values into a generic indexed state and publish the same mutated state.",
    params={"state": {"type": "slot", "default": "state"}, "indices": {"type": "slot", "default": "indices"}, "values": {"type": "slot", "default": "values"}, "save_as": {"type": "slot", "default": "state"}},
    requires=("state", "indices", "values"), provides=("save_as",), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def indexed_write(ctx: ScratchContext, state: str = "state", indices: str = "indices", values: str = "values", save_as: str = "state") -> None:
    table = ctx[state]
    rows = _torch().as_tensor(ctx[indices], dtype=_torch().long).reshape(-1).cpu()
    incoming = ctx[values].detach().reshape(rows.numel(), -1).cpu()
    if incoming.shape[1] != table["values"].shape[1]:
        raise ValueError("indexed_write value width does not match state")
    table["values"][rows] = incoming
    table["seen"][rows] = True
    if "last_epoch" in table:
        table["last_epoch"][rows] = int(ctx.get("epoch", -1))
    ctx[save_as] = table


@block(
    id="indexed_ema",
    name="Indexed EMA",
    category="State",
    description="Update persistent values by stable index with an explicit duplicate policy.",
    params={"state": {"type": "slot", "default": "state"}, "indices": {"type": "slot", "default": "indices"}, "values": {"type": "slot", "default": "values"}, "beta": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0}, "momentum": {"type": "value", "default": None}, "epoch": {"type": "slot", "default": "epoch"}, "duplicate_policy": {"type": "enum", "options": ["update", "skip", "error"], "default": "update"}, "save_as": {"type": "slot", "default": "history_values"}},
    requires=("state", "indices", "values"), provides=("save_as",), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def indexed_ema(ctx: ScratchContext, state: str = "state", indices: str = "indices", values: str = "values", beta: float = 0.9, momentum: float | None = None, epoch: str = "epoch", duplicate_policy: str = "update", save_as: str = "history_values") -> None:
    state_value = ctx[state]
    rows = ctx[indices].detach().long().cpu()
    incoming = ctx[values].detach().float().cpu()
    current_epoch = int(ctx.get(epoch, 0))
    beta = float(beta if momentum is None else momentum)
    duplicate_policy = str(duplicate_policy)
    # ``duplicate_policy`` is a data parameter, not a hidden LEND/CAL rule.
    # Keep the public operation's default as ordinary EMA updates; callers
    # that require one observation per epoch must opt into skip/error.
    if duplicate_policy not in {"update", "skip", "error"}:
        raise ValueError("duplicate_policy must be 'update', 'skip' or 'error'")
    if incoming.ndim == 1:
        incoming = incoming[:, None]
    if int(state_value["values"].shape[1]) != int(incoming.shape[1]):
        raise ValueError("indexed_ema value width does not match state width")
    for row, value in zip(rows.tolist(), incoming):
        if row < 0 or row >= int(state_value["values"].shape[0]):
            raise IndexError(f"indexed history row out of range: {row}")
        duplicate = bool(state_value["seen"][row]) and int(state_value["last_epoch"][row]) == current_epoch
        if duplicate and duplicate_policy == "error":
            raise ValueError(f"indexed_ema received duplicate row {row} in epoch {current_epoch}")
        if duplicate and duplicate_policy == "skip":
            continue
        if bool(state_value["seen"][row]):
            state_value["values"][row].mul_(float(beta)).add_(value, alpha=1.0 - float(beta))
        else:
            state_value["values"][row].copy_(value)
            state_value["seen"][row] = True
        state_value["last_epoch"][row] = current_epoch
    ctx[save_as] = state_value["values"][rows].to(ctx[values].device)


# Compatibility import alias.  The registered operation is defined in the
# selection module so the physical module matches its Sample Selection
# semantics.
from .selection import agreement_mask


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


@block(
    id="grouped_accumulate",
    name="Grouped Accumulate",
    category="State",
    description="Accumulate detached values by group label using the public grouped-state contract.",
    params={"state": {"type": "slot", "default": "grouped_state"}, "groups": {"type": "slot", "default": "groups"}, "values": {"type": "slot", "default": "values"}},
    requires=("state", "groups", "values"), provides=(), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def grouped_accumulate(ctx: ScratchContext, state: str = "grouped_state", groups: str = "groups", values: str = "values") -> None:
    state_value = ctx[state]
    labels = ctx[groups].detach().long().reshape(-1).cpu()
    tensor = ctx[values].detach().float().reshape(labels.numel(), -1).cpu()
    for group in labels.unique().tolist():
        mask = labels == int(group)
        state_value["sums"][int(group)] += tensor[mask].sum(dim=0)
        state_value["counts"][int(group)] += mask.sum()


@block(
    id="reset_grouped_accumulator",
    name="Reset Grouped Accumulator",
    category="State",
    description="Clear sums and counts in an existing grouped accumulator.",
    params={"state": {"type": "slot", "default": "grouped_state"}},
    requires=("state",), provides=(), placement=("epoch", "top"), stage="train", ui_group="④ 状态更新",
)
def reset_grouped_accumulator(ctx: ScratchContext, state: str = "grouped_state") -> None:
    value = ctx[state]
    value["sums"].zero_(); value["counts"].zero_()


@block(
    id="create_indexed_window",
    name="Create Indexed Window",
    category="State",
    description="Create a fixed-size per-index history window; rows are addressed by stable sample indices.",
    params={"indices": {"type": "slot", "default": "indices"}, "prepared_data": {"type": "value", "default": None}, "size": {"type": "int", "default": 0, "min": 0}, "window_size": {"type": "int", "default": 5, "min": 1}, "width": {"type": "int", "default": 1, "min": 1}, "epoch_indexed": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "indexed_window"}},
    provides=("save_as",), placement=("top",), stage="setup", ui_group="④ 状态更新",
)
def create_indexed_window(ctx: ScratchContext, indices: str = "indices", prepared_data: Any = None, size: int = 0, window_size: int = 5, width: int = 1, epoch_indexed: bool = False, save_as: str = "indexed_window") -> None:
    torch = _torch(); size = int(size)
    if isinstance(prepared_data, str) and prepared_data in ctx:
        prepared = ctx[prepared_data]
        train_indices = getattr(prepared, "train_indices", None)
        if train_indices is not None and len(train_indices):
            size = max(size, int(max(train_indices)) + 1)
    if indices in ctx:
        values = torch.as_tensor(ctx[indices]).reshape(-1)
        if values.numel(): size = max(size, int(values.max().item()) + 1)
    ctx[save_as] = {
        "values": torch.zeros((size, int(window_size), int(width))),
        "observed": torch.zeros((size, int(window_size)), dtype=torch.bool),
        "counts": torch.zeros(size, dtype=torch.long),
        "cursor": torch.zeros(size, dtype=torch.long),
        "window_size": int(window_size),
        "epoch": 0,
        "active_slot": 0,
        "epoch_indexed": bool(epoch_indexed),
    }


@block(
    id="append_indexed_window",
    name="Append Indexed Window",
    category="State",
    description="Append detached per-index values to a bounded rolling window.",
    params={"state": {"type": "slot", "default": "indexed_window"}, "indices": {"type": "slot", "default": "indices"}, "values": {"type": "slot", "default": "values"}},
    requires=("state", "indices", "values"), provides=(), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def append_indexed_window(ctx: ScratchContext, state: str = "indexed_window", indices: str = "indices", values: str = "values") -> None:
    table = ctx[state]; rows = _torch().as_tensor(ctx[indices]).long().reshape(-1).cpu(); incoming = ctx[values].detach().float().reshape(rows.numel(), -1).cpu()
    if incoming.shape[1] != int(table["values"].shape[2]): raise ValueError("indexed window value width does not match state")
    for row, value in zip(rows.tolist(), incoming):
        if row < 0 or row >= int(table["values"].shape[0]): raise IndexError("indexed window row out of range")
        position = int(table["active_slot"]) if table.get("epoch_indexed") else int(table["cursor"][row])
        table["values"][row, position] = value
        if not table.get("epoch_indexed"):
            table["cursor"][row] = (position + 1) % int(table["window_size"])
        table["counts"][row] = min(int(table["counts"][row]) + 1, int(table["window_size"]))
        table["observed"][row, position] = True


@block(
    id="read_indexed_window",
    name="Read Indexed Window",
    category="State",
    description="Read rolling history rows and counts without creating paper-specific state objects.",
    params={"state": {"type": "slot", "default": "indexed_window"}, "indices": {"type": "slot", "default": "indices"}, "values_as": {"type": "slot", "default": "window_values"}, "counts_as": {"type": "slot", "default": "window_counts"}, "observed_as": {"type": "slot", "default": "window_observed"}},
    requires=("state", "indices"), provides=("values_as", "counts_as", "observed_as"), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def read_indexed_window(ctx: ScratchContext, state: str = "indexed_window", indices: str = "indices", values_as: str = "window_values", counts_as: str = "window_counts", observed_as: str = "window_observed") -> None:
    table = ctx[state]; rows = _torch().as_tensor(ctx[indices]).long().reshape(-1).cpu(); ctx[values_as] = table["values"][rows].clone(); ctx[counts_as] = table["counts"][rows].clone(); ctx[observed_as] = table["observed"][rows].clone()


@block(
    id="reset_indexed_window",
    name="Reset Indexed Window",
    category="State",
    description="Reset selected rows of a fixed-size indexed history window.",
    params={"state": {"type": "slot", "default": "indexed_window"}, "indices": {"type": "slot", "default": "indices"}},
    requires=("state", "indices"), provides=(), placement=("epoch", "top"), stage="train", ui_group="④ 状态更新",
)
def reset_indexed_window(ctx: ScratchContext, state: str = "indexed_window", indices: str = "indices") -> None:
    table = ctx[state]
    rows = _torch().as_tensor(ctx[indices]).long().reshape(-1).cpu()
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= int(table["values"].shape[0])):
        raise IndexError("indexed window row out of range")
    table["values"][rows] = 0
    table["observed"][rows] = False
    table["counts"][rows] = 0
    table["cursor"][rows] = 0


@block(
    id="advance_indexed_window_epoch",
    name="Advance Indexed Window Epoch",
    category="State",
    description="Advance a fixed-window history to a new epoch and clear only the active slot.",
    params={"state": {"type": "slot", "default": "indexed_window"},
            "epoch": {"type": "slot", "default": "epoch"}},
    requires=("state", "epoch"), provides=(), placement=("epoch",), stage="train", ui_group="④ 状态更新",
)
def advance_indexed_window_epoch(ctx: ScratchContext, state: str = "indexed_window", epoch: str = "epoch") -> None:
    table = ctx[state]
    if "window_size" not in table or "values" not in table:
        raise TypeError("advance_indexed_window_epoch requires an indexed window state")
    slot = int(ctx[epoch]) % int(table["window_size"])
    table["values"][:, slot] = 0
    if "observed" in table:
        table["observed"][:, slot] = False
    table["epoch"] = int(ctx[epoch])
    table["active_slot"] = slot


@block(
    id="indexed_increment",
    name="Indexed Increment",
    category="State",
    description="Increment a scalar indexed table at stable sample indices.",
    params={"state": {"type": "slot", "default": "state"},
            "indices": {"type": "slot", "default": "indices"},
            "values": {"type": "slot", "default": "increments"}},
    requires=("state", "indices", "values"), provides=(), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def indexed_increment(ctx: ScratchContext, state: str = "state", indices: str = "indices", values: str = "increments") -> None:
    table = ctx[state]
    rows = _torch().as_tensor(ctx[indices], dtype=_torch().long).reshape(-1).cpu()
    raw = ctx[values] if isinstance(values, str) and values in ctx else values
    increments = _torch().as_tensor(raw).reshape(-1).detach().cpu()
    if increments.numel() == 1 and rows.numel() != 1:
        increments = increments.expand(rows.numel())
    if rows.numel() != increments.numel() or int(table["values"].shape[1]) != 1:
        raise ValueError("indexed_increment requires one scalar value per index")
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= int(table["values"].shape[0])):
        raise IndexError("indexed_increment index out of range")
    table["values"][rows] += increments.to(table["values"].dtype).reshape(-1, 1)
    if "seen" in table:
        table["seen"][rows] = True


@block(
    id="ema_update",
    name="EMA Update",
    category="State",
    description="Compute an exponential moving-average value without owning paper-specific state.",
    params={"previous": {"type": "slot", "default": "previous"},
            "current": {"type": "slot", "default": "current"},
            "momentum": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0},
            "save_as": {"type": "slot", "default": "updated"}},
    requires=("previous", "current"), provides=("save_as",), placement=("batch", "epoch"), stage="train", ui_group="④ 状态更新",
)
def ema_update(ctx: ScratchContext, previous: str = "previous", current: str = "current",
               momentum: float = 0.9, save_as: str = "updated") -> None:
    value = ctx[previous]
    target = ctx[current]
    ctx[save_as] = value * float(momentum) + target * (1.0 - float(momentum))


@block(
    id="robust_window_mean",
    name="Robust Window Mean",
    category="State",
    description="Average observed values in an indexed window, excluding unobserved entries.",
    params={"values": {"type": "slot", "default": "window_values"},
            "observed": {"type": "slot", "default": "window_observed"},
            "save_as": {"type": "slot", "default": "robust_mean"},
            "counts_as": {"type": "slot", "default": "window_counts"}},
    requires=("values", "observed"), provides=("save_as", "counts_as"), placement=("batch",), stage="train", ui_group="④ 状态更新",
)
def robust_window_mean(ctx: ScratchContext, values: str = "window_values", observed: str = "window_observed",
                       save_as: str = "robust_mean", counts_as: str = "window_counts") -> None:
    tensor = ctx[values]
    mask = ctx[observed].bool()
    if tensor.ndim == mask.ndim + 1 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.shape != mask.shape:
        raise ValueError("robust_window_mean values and observed shapes must match")
    counts = mask.sum(dim=-1)
    denominator = counts.clamp_min(1).to(tensor.dtype)
    ctx[save_as] = (tensor * mask.to(tensor.dtype)).sum(dim=-1) / denominator
    ctx[counts_as] = counts
