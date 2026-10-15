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
    description="Select critical parameters using the paper's exact top-k or the released code's 2D/4D threshold masks.",
    params={"model": {"type": "slot", "default": "model"}, "noise_rate": {"type": "float", "default": 0.4, "min": 0.0, "max": 0.999999}, "compatibility_mode": {"type": "enum", "default": "paper", "options": ["paper", "official_code"]}, "save_as": {"type": "slot", "default": "critical_masks"}},
    requires=("model",), provides=("save_as",), placement=("batch", "epoch"), stage="train", ui_group="⑤ 损失公式",
    formula="s_i=|g_i theta_i|; S=top-ceil((1-tau)m)(s)", formula_ref="critical parameter selection", formula_kind="special",
)
def parameter_criticality_mask(ctx: ScratchContext, model: str = "model", noise_rate: float = 0.4, save_as: str = "critical_masks", compatibility_mode: str = "paper") -> None:
    from lnl_toolbox.algorithms.cdr import critical_parameter_masks, official_code_parameter_masks
    if compatibility_mode == "paper":
        selected = critical_parameter_masks(ctx[model].named_parameters(), noise_rate)
    elif compatibility_mode == "official_code":
        selected = official_code_parameter_masks(ctx[model].named_parameters(), noise_rate)
    else:
        raise ValueError("compatibility_mode must be 'paper' or 'official_code'")
    ctx[save_as] = dict(selected.masks)


@block(
    id="masked_gradient_update",
    name="Masked Gradient Update",
    category="Parameter Update",
    description="Apply a parameter mask, global gradient scale, and optional L1 term before the optimizer step.",
    params={"model": {"type":"slot", "default":"model"}, "masks": {"type":"slot", "default":"critical_masks"}, "scale": {"type":"float", "default":0.6, "min":0.0}, "l1_decay": {"type":"float", "default":0.001, "min":0.0}},
    requires=("model", "masks"), provides=(), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="g_i <- scale*1[i in S]g_i + lambda sign(theta_i)", formula_ref="masked parameter gradient update", formula_kind="special",
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


@block(
    id="parameter_squared_norm",
    name="Parameter Squared Norm",
    category="Parameter Update",
    description="Reduce a module's trainable parameters to one differentiable squared-norm scalar.",
    params={"model": {"type": "slot", "default": "model"}, "save_as": {"type": "slot", "default": "parameter_norm"}},
    requires=("model",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
)
def parameter_squared_norm(ctx: ScratchContext, model: str = "model", save_as: str = "parameter_norm") -> None:
    parameters = [parameter for parameter in ctx[model].parameters() if parameter.requires_grad]
    if not parameters: raise ValueError("parameter_squared_norm requires trainable parameters")
    ctx[save_as] = sum((parameter.square().sum() for parameter in parameters))
