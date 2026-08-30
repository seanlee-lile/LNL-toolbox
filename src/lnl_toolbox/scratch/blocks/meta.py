"""Scratch-native, composable higher-order parameter operations."""

from __future__ import annotations

from typing import Any

from ..context import ScratchContext
from ..registry import block


def _torch():
    import torch
    return torch


@block(
    id="virtual_parameter_update",
    name="Virtual Parameter Update",
    category="Meta",
    description="Differentiate a scalar loss and expose a functional virtual parameter state.",
    params={"model": {"type": "slot", "default": "model"}, "loss": {"type": "slot", "default": "virtual_loss"}, "learning_rate": {"type": "float", "default": 1.0, "min": 0.0}, "create_graph": {"type": "bool", "default": True}, "save_as": {"type": "slot", "default": "virtual_state"}},
    requires=("model", "loss"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="theta'=theta-alpha grad_theta L", formula_ref="functional virtual parameter update",
)
def virtual_parameter_update(ctx: ScratchContext, model: str = "model", loss: str = "virtual_loss", learning_rate: float = 1.0, create_graph: bool = True, save_as: str = "virtual_state") -> None:
    torch = _torch(); network = ctx[model]
    named = dict(network.named_parameters())
    if not named: raise ValueError("virtual_parameter_update requires a model with parameters")
    gradients = torch.autograd.grad(ctx[loss], tuple(named.values()), create_graph=bool(create_graph), retain_graph=True, allow_unused=False)
    virtual = {name: parameter - float(learning_rate) * gradient for (name, parameter), gradient in zip(named.items(), gradients)}
    virtual.update(dict(network.named_buffers()))
    ctx[save_as] = {"parameters": named, "gradients": gradients, "state": virtual, "create_graph": bool(create_graph)}


@block(
    id="functional_forward_with_state",
    name="Functional Forward With State",
    category="Meta",
    description="Evaluate a model with an explicit virtual parameter/buffer state.",
    params={"model": {"type": "slot", "default": "model"}, "state": {"type": "slot", "default": "virtual_state"}, "inputs": {"type": "slot", "default": "images"}, "save_as": {"type": "slot", "default": "logits"}},
    requires=("model", "state", "inputs"), provides=("save_as",), placement=("batch",), stage="train", ui_group="④ 前向与概率",
)
def functional_forward_with_state(ctx: ScratchContext, model: str = "model", state: str = "virtual_state", inputs: str = "images", save_as: str = "logits") -> None:
    from torch.func import functional_call
    holder = ctx[state]
    ctx[save_as] = functional_call(ctx[model], holder["state"], (ctx[inputs],), strict=True)


@block(
    id="functional_forward",
    name="Functional Forward",
    category="Meta",
    description="Evaluate a module with an explicit parameter/buffer mapping.",
    params={"model": {"type": "slot", "default": "model"},
            "state": {"type": "slot", "default": "parameters"},
            "inputs": {"type": "slot", "default": "images"},
            "save_as": {"type": "slot", "default": "logits"}},
    requires=("model", "state", "inputs"), provides=("save_as",), placement=("batch",), stage="train", ui_group="④ 前向与概率",
)
def functional_forward(ctx: ScratchContext, model: str = "model", state: str = "parameters",
                       inputs: str = "images", save_as: str = "logits") -> None:
    from torch.func import functional_call
    mapping = ctx[state]
    if isinstance(mapping, dict) and "state" in mapping and "parameters" in mapping:
        mapping = mapping["state"]
    if not isinstance(mapping, dict):
        raise TypeError("functional_forward state must be a parameter/buffer mapping")
    ctx[save_as] = functional_call(ctx[model], mapping, (ctx[inputs],), strict=True)


@block(
    id="gradient_wrt",
    name="Gradient With Respect To",
    category="Meta",
    description="Differentiate a scalar loss with respect to an explicit differentiable input slot.",
    params={"loss": {"type": "slot", "default": "loss"}, "input": {"type": "slot", "default": "input"}, "create_graph": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "gradient"}},
    requires=("loss", "input"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
)
def gradient_wrt(ctx: ScratchContext, loss: str = "loss", input: str = "input", create_graph: bool = False, save_as: str = "gradient") -> None:
    torch = _torch(); value = ctx[input]
    if not getattr(value, "requires_grad", False): raise ValueError("gradient_wrt input must require gradients")
    result = torch.autograd.grad(ctx[loss], value, create_graph=bool(create_graph), retain_graph=True, allow_unused=False)[0]
    if not torch.isfinite(result).all(): raise ValueError("gradient_wrt produced a non-finite gradient")
    ctx[save_as] = result
