"""Forward-pass blocks using string slots for multi-model recipes."""

from __future__ import annotations

from typing import Any

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Forward blocks require PyTorch; install the `train` extra.") from exc
    return torch


@block(
    id="forward",
    name="Forward",
    category="Forward",
    description="Run a model on an input slot and save logits.",
    params={
        "model": {"type": "slot", "default": "model"},
        "input": {"type": "slot", "default": "images"},
        "save_as": {"type": "slot", "default": "logits"},
    },
    requires=("model", "input"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="④ 前向与概率",
)
def forward(
    ctx: ScratchContext,
    model: str = "model",
    input: str = "images",
    save_as: str = "logits",
) -> None:
    ctx[save_as] = ctx[model](ctx[input])


@block(
    id="module_forward",
    name="Module Forward",
    category="Forward",
    description="Run one module with an explicit ordered list of input slots.",
    params={"module": {"type": "slot", "default": "module"},
            "inputs": {"type": "value", "default": []},
            "save_as": {"type": "slot", "default": "output"}},
    requires=("module",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="④ 前向与概率",
)
def module_forward(ctx: ScratchContext, module: str = "module", inputs: Any = (), save_as: str = "output") -> None:
    """Call a module without embedding paper-specific predictor semantics."""
    if isinstance(inputs, str):
        inputs = [inputs]
    if inputs is None:
        inputs = []
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("module_forward inputs must be a sequence of slot names")
    arguments = [ctx[item] if isinstance(item, str) and item in ctx else item for item in inputs]
    ctx[save_as] = ctx[module](*arguments)


@block(
    id="forward_feature",
    name="Forward + Feature",
    category="Forward",
    description="Run a model and save both logits and penultimate features when available.",
    params={
        "model": {"type": "slot", "default": "model"},
        "input": {"type": "slot", "default": "images"},
        "logits_as": {"type": "slot", "default": "logits"},
        "features_as": {"type": "slot", "default": "features"},
    },
    requires=("model", "input"),
    provides=("logits_as", "features_as"),
    placement=("batch",), stage="train", ui_group="④ 前向与概率", beginner_visible=False,
)
def forward_feature(
    ctx: ScratchContext,
    model: str = "model",
    input: str = "images",
    logits_as: str = "logits",
    features_as: str = "features",
) -> None:
    output = ctx[model].forward_with_features(ctx[input])
    if hasattr(output, "logits") and hasattr(output, "features"):
        ctx[logits_as], ctx[features_as] = output.logits, output.features
    else:
        ctx[logits_as], ctx[features_as] = output


@block(
    id="softmax",
    name="Softmax / Probability",
    category="Forward",
    description="Convert logits to class probabilities.",
    params={
        "logits": {"type": "slot", "default": "logits"},
        "temperature": {"type": "float", "default": 1.0, "min": 0.000001},
        "detach": {"type": "bool", "default": False},
        "save_as": {"type": "slot", "default": "probabilities"},
    },
    requires=("logits",),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="④ 前向与概率",
    formula="p = softmax(z)", formula_ref="method definition",
)
def softmax(
    ctx: ScratchContext,
    logits: str = "logits",
    temperature: float = 1.0,
    detach: bool = False,
    save_as: str = "probabilities",
) -> None:
    torch = _torch()
    values = ctx[logits].detach() if bool(detach) else ctx[logits]
    ctx[save_as] = torch.softmax(values / float(temperature), dim=-1)
