from __future__ import annotations

"""Directional and random schedules used by paper-oriented DLD."""

from dataclasses import dataclass
import hashlib

import torch
from torch import Tensor


@dataclass(frozen=True)
class DirectionalDiffusionSchedule:
    alpha: Tensor
    beta: Tensor
    alpha_bar: Tensor
    beta_bar: Tensor

    @classmethod
    def average(
        cls,
        timesteps: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        tolerance: float = 1e-6,
    ) -> "DirectionalDiffusionSchedule":
        if isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps <= 0:
            raise ValueError("DLD timesteps must be a positive integer")
        alpha = torch.full((timesteps,), 1.0 / timesteps, device=device, dtype=dtype)
        beta = torch.full(
            (timesteps,), 1.0 / (timesteps ** 0.5), device=device, dtype=dtype
        )
        result = cls(alpha, beta, alpha.cumsum(0), beta.square().cumsum(0).sqrt())
        result.validate(tolerance=tolerance)
        return result

    @property
    def timesteps(self) -> int:
        return int(self.alpha.numel())

    @property
    def identity_hash(self) -> str:
        digest = hashlib.sha256()
        for value in (self.alpha, self.beta, self.alpha_bar, self.beta_bar):
            cpu = value.detach().cpu().contiguous()
            digest.update(str(tuple(cpu.shape)).encode())
            digest.update(cpu.numpy().tobytes())
        return digest.hexdigest()

    def validate(self, *, tolerance: float = 1e-6) -> None:
        shapes = {tuple(value.shape) for value in (self.alpha, self.beta, self.alpha_bar, self.beta_bar)}
        if len(shapes) != 1 or next(iter(shapes))[0] <= 0:
            raise ValueError("DLD schedule values must share non-empty shape [T]")
        for value in (self.alpha, self.beta, self.alpha_bar, self.beta_bar):
            if value.ndim != 1 or not torch.is_floating_point(value):
                raise ValueError("DLD schedule values must be floating tensors [T]")
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError("DLD schedule values must be finite and non-negative")
        if abs(float(self.alpha_bar[-1]) - 1.0) > tolerance:
            raise ValueError("DLD directional schedule endpoint must equal one")
        if abs(float(self.beta_bar[-1]) - 1.0) > tolerance:
            raise ValueError("DLD random schedule endpoint must equal one")

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "alpha": self.alpha.detach().cpu(),
            "beta": self.beta.detach().cpu(),
            "alpha_bar": self.alpha_bar.detach().cpu(),
            "beta_bar": self.beta_bar.detach().cpu(),
        }


__all__ = ["DirectionalDiffusionSchedule"]
