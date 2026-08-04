from __future__ import annotations

"""Pure DLD forward-state and objective functions."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .schedules import DirectionalDiffusionSchedule


def construct_direction(y0: Tensor, yn: Tensor) -> Tensor:
    if not torch.is_tensor(y0) or not torch.is_tensor(yn) or y0.shape != yn.shape:
        raise ValueError("DLD y0 and yn must be matching tensors")
    if y0.ndim != 2 or not torch.is_floating_point(y0) or yn.device != y0.device:
        raise ValueError("DLD y0 and yn must be floating [B, C] tensors on one device")
    if not bool(torch.isfinite(y0).all()) or not bool(torch.isfinite(yn).all()):
        raise ValueError("DLD y0 and yn must be finite")
    return yn - y0


def sample_forward_state(
    y0: Tensor,
    yd: Tensor,
    timestep: Tensor,
    epsilon: Tensor,
    schedule: DirectionalDiffusionSchedule,
) -> Tensor:
    if y0.ndim != 2 or yd.shape != y0.shape or epsilon.shape != y0.shape:
        raise ValueError("DLD y0, yd and epsilon must share shape [B, C]")
    if timestep.shape != (y0.shape[0],) or timestep.dtype != torch.int64:
        raise ValueError("DLD timestep must be int64 with shape [B]")
    if timestep.device != y0.device or yd.device != y0.device or epsilon.device != y0.device:
        raise ValueError("DLD forward tensors must share a device")
    if bool((timestep < 0).any()) or bool((timestep >= schedule.timesteps).any()):
        raise ValueError("DLD timestep is outside the schedule")
    for value in (y0, yd, epsilon):
        if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
            raise ValueError("DLD forward tensors must be finite and floating")
    alpha = schedule.alpha_bar.to(y0.device, y0.dtype).gather(0, timestep)[:, None]
    beta = schedule.beta_bar.to(y0.device, y0.dtype).gather(0, timestep)[:, None]
    result = y0 + alpha * yd + beta * epsilon
    if not bool(torch.isfinite(result).all()):
        raise ValueError("DLD forward state is non-finite")
    return result


@dataclass(frozen=True)
class DLDObjectiveResult:
    direction_per_sample: Tensor
    noise_per_sample: Tensor
    direction_loss: Tensor
    noise_loss: Tensor


def dld_objective(
    predicted_direction: Tensor,
    target_direction: Tensor,
    predicted_noise: Tensor,
    target_noise: Tensor,
) -> DLDObjectiveResult:
    if predicted_direction.shape != target_direction.shape or predicted_noise.shape != target_noise.shape:
        raise ValueError("DLD prediction and target shapes must match")
    if predicted_direction.ndim != 2 or predicted_noise.shape != predicted_direction.shape:
        raise ValueError("DLD objective tensors must share shape [B, C]")
    values = (predicted_direction, target_direction, predicted_noise, target_noise)
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("DLD objective tensors must be finite")
    direction = (predicted_direction - target_direction).square().mean(dim=1)
    noise = (predicted_noise - target_noise).square().mean(dim=1)
    return DLDObjectiveResult(direction, noise, direction.mean(), noise.mean())


__all__ = [
    "DLDObjectiveResult",
    "construct_direction",
    "dld_objective",
    "sample_forward_state",
]
