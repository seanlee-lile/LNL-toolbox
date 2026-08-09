from __future__ import annotations

"""Deterministic accelerated reverse sampling for paper-oriented DLD.

The paper-oriented v1 policy uses the paper's difference update for cumulative
directional/random coefficients, while deliberately starting from an all-zero
label vector. The zero initialization is an engineering policy, not the paper's
forward endpoint formula.
"""

import numpy as np
import torch
from torch import Tensor, nn

from .schedules import DirectionalDiffusionSchedule


def accelerated_timesteps(total_timesteps: int, inference_steps: int) -> tuple[tuple[int, int], ...]:
    if total_timesteps <= 0 or inference_steps <= 0 or inference_steps > total_timesteps:
        raise ValueError("DLD inference steps must satisfy 0 < S <= T")
    points = np.linspace(-1, total_timesteps - 1, inference_steps + 1).astype(np.int64)
    if np.unique(points).size != points.size:
        raise ValueError("DLD accelerated timestep sequence contains duplicates")
    ordered = points[::-1].tolist()
    return tuple((int(a), int(b)) for a, b in zip(ordered[:-1], ordered[1:]))


@torch.inference_mode()
def sample_labels(
    direction_model: nn.Module,
    noise_model: nn.Module,
    condition_features: Tensor,
    schedule: DirectionalDiffusionSchedule,
    *,
    inference_steps: int = 5,
    y_n: Tensor | None = None,
) -> Tensor:
    if condition_features.ndim != 2 or not torch.is_floating_point(condition_features):
        raise ValueError("DLD sampling condition_features must be floating [B, D]")
    if not bool(torch.isfinite(condition_features).all()):
        raise ValueError("DLD sampling features must be finite")
    batch = condition_features.shape[0]
    classes = int(getattr(direction_model, "num_classes", 0))
    if classes < 2 or int(getattr(noise_model, "num_classes", 0)) != classes:
        raise ValueError("DLD sampling requires compatible predictor class counts")
    if y_n is None:
        y_n = torch.zeros(batch, classes, device=condition_features.device, dtype=condition_features.dtype)
    if y_n.shape != (batch, classes) or y_n.device != condition_features.device:
        raise ValueError("DLD sampling y_n must have shape [B, C] on the feature device")
    direction_training = direction_model.training
    noise_training = noise_model.training
    direction_model.eval(); noise_model.eval()
    try:
        y_t = torch.zeros_like(y_n)
        for current, following in accelerated_timesteps(schedule.timesteps, inference_steps):
            timestep = torch.full((batch,), current, dtype=torch.int64, device=y_t.device)
            predicted_direction = direction_model(y_t, y_n, condition_features, timestep)
            predicted_noise = noise_model(y_t, y_n, condition_features, timestep)
            next_alpha = 0.0 if following < 0 else float(schedule.alpha_bar[following])
            next_beta = 0.0 if following < 0 else float(schedule.beta_bar[following])
            delta_alpha = float(schedule.alpha_bar[current]) - next_alpha
            delta_beta = float(schedule.beta_bar[current]) - next_beta
            y_t = y_t - delta_alpha * predicted_direction - delta_beta * predicted_noise
            if not bool(torch.isfinite(y_t).all()):
                raise ValueError("DLD reverse sample became non-finite")
        return y_t
    finally:
        direction_model.train(direction_training)
        noise_model.train(noise_training)


__all__ = ["accelerated_timesteps", "sample_labels"]
