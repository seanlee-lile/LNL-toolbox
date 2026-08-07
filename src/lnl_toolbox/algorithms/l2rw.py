from __future__ import annotations

"""Differentiable one-step example reweighting from Ren et al. (ICML 2018)."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.func import functional_call

from lnl_toolbox.treatments.weights import WeightResult


def meta_gradient(
    model: nn.Module,
    training_inputs: Tensor,
    noisy_targets: Tensor,
    trusted_inputs: Tensor,
    trusted_targets: Tensor,
    *,
    virtual_learning_rate: float,
    weight_decay: float = 0.0,
    implementation: str = "paper",
) -> Tensor:
    alpha = float(virtual_learning_rate)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("virtual_learning_rate must be finite and positive")
    decay = float(weight_decay)
    if not math.isfinite(decay) or decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    implementation = str(implementation).strip().lower()
    if implementation not in {"paper", "official"}:
        raise ValueError("implementation must be 'paper' or 'official'")
    if training_inputs.shape[0] == 0 or trusted_inputs.shape[0] == 0:
        raise ValueError("L2RW training and trusted batches must be non-empty")
    if training_inputs.ndim < 2 or trusted_inputs.ndim < 2:
        raise ValueError("L2RW inputs must include a batch dimension and features")
    if noisy_targets.shape != (training_inputs.shape[0],):
        raise ValueError("noisy_targets must align with training_inputs")
    if trusted_targets.shape != (trusted_inputs.shape[0],):
        raise ValueError("trusted_targets must align with trusted_inputs")
    if noisy_targets.device != training_inputs.device or trusted_targets.device != trusted_inputs.device:
        raise ValueError("L2RW inputs and targets must share devices")
    if noisy_targets.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    } or trusted_targets.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    }:
        raise TypeError("L2RW targets must use integer dtypes")
    parameters = dict(model.named_parameters())
    if not parameters:
        raise ValueError("L2RW model has no trainable parameters")
    buffers = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }
    train_logits = functional_call(
        model, {**parameters, **buffers}, (training_inputs,), strict=True
    )
    if train_logits.ndim != 2 or train_logits.shape[0] != training_inputs.shape[0]:
        raise ValueError("L2RW model must return logits with shape [B,C]")
    train_losses = F.cross_entropy(train_logits, noisy_targets.long(), reduction="none")
    epsilon = torch.zeros_like(train_losses, requires_grad=True)
    virtual_loss = torch.sum(epsilon * train_losses)
    gradients = torch.autograd.grad(
        virtual_loss, tuple(parameters.values()), create_graph=True
    )
    if implementation == "official":
        if abs(alpha - 1.0) > 1e-12:
            raise ValueError("official L2RW reweight_autodiff has no virtual update")
        state = {**parameters, **buffers}
    else:
        virtual_parameters = {
            name: parameter - alpha * gradient
            for (name, parameter), gradient in zip(parameters.items(), gradients)
        }
        state = {**virtual_parameters, **buffers}
    trusted_logits = functional_call(model, state, (trusted_inputs,), strict=True)
    if trusted_logits.ndim != 2 or trusted_logits.shape[0] != trusted_inputs.shape[0]:
        raise ValueError("L2RW model must return trusted logits with shape [B,C]")
    trusted_loss = F.cross_entropy(trusted_logits, trusted_targets.long())
    if decay:
        trusted_loss = trusted_loss + 0.5 * decay * sum(
            parameter.square().sum() for parameter in parameters.values()
        )
    if implementation == "official":
        trusted_gradients = torch.autograd.grad(
            trusted_loss, tuple(parameters.values()), retain_graph=True
        )
        gradient = torch.autograd.grad(
            gradients, epsilon, grad_outputs=trusted_gradients, only_inputs=True
        )[0]
    else:
        gradient = torch.autograd.grad(trusted_loss, epsilon, only_inputs=True)[0]
    if not bool(torch.isfinite(gradient).all().item()):
        raise ValueError("L2RW meta-gradient is non-finite")
    return gradient


def meta_reweight(
    model: nn.Module,
    training_inputs: Tensor,
    noisy_targets: Tensor,
    trusted_inputs: Tensor,
    trusted_targets: Tensor,
    *,
    virtual_learning_rate: float,
    weight_decay: float = 0.0,
    implementation: str = "paper",
) -> WeightResult:
    gradient = meta_gradient(
        model,
        training_inputs,
        noisy_targets,
        trusted_inputs,
        trusted_targets,
        virtual_learning_rate=virtual_learning_rate,
        weight_decay=weight_decay,
        implementation=implementation,
    )
    raw = torch.relu(
        gradient if str(implementation).strip().lower() == "official" else -gradient
    ).detach()
    total = raw.sum()
    weights = raw / total if bool(total > 0) else torch.zeros_like(raw)
    return WeightResult(
        weights,
        {
            "positive_weight_count": float(weights.gt(0).sum().item()),
            "weight_sum": float(weights.sum().item()),
            "meta_gradient_norm": float(gradient.detach().norm().item()),
        },
    )


__all__ = ["meta_gradient", "meta_reweight"]
