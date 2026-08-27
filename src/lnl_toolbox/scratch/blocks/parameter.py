"""Scratch-native parameter criticality operations."""

from __future__ import annotations

from ..context import ScratchContext
from ..registry import block


def _torch():
    import torch
    return torch


@block(
    id="parameter_criticality_mask",
    name="Parameter Criticality Mask",
    category="Parameter Update",
    description="Select trainable scalar parameters with the largest detached |gradient times parameter| scores.",
    params={"model": {"type": "slot", "default": "model"}, "noise_rate": {"type": "float", "default": 0.4, "min": 0.0, "max": 0.999999}, "save_as": {"type": "slot", "default": "critical_masks"}},
    requires=("model",), provides=("save_as",), placement=("batch", "epoch"), stage="train", ui_group="⑤ 损失公式",
    formula="s_i=|g_i theta_i|; S=top-ceil((1-tau)m)(s)", formula_ref="critical parameter selection",
)
def parameter_criticality_mask(ctx: ScratchContext, model: str = "model", noise_rate: float = 0.4, save_as: str = "critical_masks") -> None:
    import math
    torch = _torch()
    tau = float(noise_rate)
    if not 0.0 <= tau < 1.0:
        raise ValueError("noise_rate must satisfy 0 <= noise_rate < 1")
    values = sorted(((str(name), parameter) for name, parameter in ctx[model].named_parameters() if parameter.requires_grad and parameter.grad is not None), key=lambda item: item[0])
    if not values:
        raise ValueError("parameter criticality requires trainable parameters with gradients")
    scores = torch.cat([(parameter.grad.detach() * parameter.detach()).abs().reshape(-1).to(torch.float64) for _, parameter in values])
    count = int(math.ceil((1.0 - tau) * int(scores.numel())))
    order = torch.argsort(scores, descending=True, stable=True)
    selected = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    selected[order[:count]] = True
    masks = {}
    offset = 0
    for name, parameter in values:
        size = int(parameter.numel())
        masks[name] = selected[offset:offset + size].reshape(parameter.shape)
        offset += size
    ctx[save_as] = masks


@block(
    id="masked_gradient_update",
    name="Masked Gradient Update",
    category="Parameter Update",
    description="Apply a parameter mask, global gradient scale, and optional L1 term before the optimizer step.",
    params={"model": {"type":"slot", "default":"model"}, "masks": {"type":"slot", "default":"critical_masks"}, "scale": {"type":"float", "default":0.6, "min":0.0}, "l1_decay": {"type":"float", "default":0.001, "min":0.0}},
    requires=("model", "masks"), provides=(), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="g_i <- scale*1[i in S]g_i + lambda sign(theta_i)", formula_ref="masked parameter gradient update",
)
def masked_gradient_update(ctx: ScratchContext, model: str = "model", masks: str = "critical_masks", scale: float = 0.6, l1_decay: float = 0.001) -> None:
    if float(scale) < 0.0 or float(l1_decay) < 0.0:
        raise ValueError("scale and l1_decay must be non-negative")
    torch = _torch()
    with torch.no_grad():
        for name, parameter in ctx[model].named_parameters():
            if parameter.grad is None or name not in ctx[masks]:
                continue
            parameter.grad.mul_(ctx[masks][name].to(device=parameter.grad.device, dtype=parameter.grad.dtype))
            parameter.grad.mul_(float(scale))
            if float(l1_decay):
                parameter.grad.add_(parameter.sign(), alpha=float(l1_decay))
