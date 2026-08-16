from __future__ import annotations

"""Two-predictor optimization unit for DLD diffusion training."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lnl_toolbox.training.model_ema import ModelEMA

from .artifacts import DLDPreCorrectionArtifact
from .objective import dld_objective, sample_forward_state
from .schedules import DirectionalDiffusionSchedule


class DLDAlgorithm:
    def __init__(
        self,
        *,
        direction_model: nn.Module,
        noise_model: nn.Module,
        direction_optimizer: torch.optim.Optimizer,
        noise_optimizer: torch.optim.Optimizer,
        direction_scheduler: Any,
        noise_scheduler: Any,
        schedule: DirectionalDiffusionSchedule,
        artifact: DLDPreCorrectionArtifact,
        device: torch.device | str,
        ema_decay: float | None,
    ) -> None:
        self.device = torch.device(device)
        self.direction_model = direction_model.to(self.device)
        self.noise_model = noise_model.to(self.device)
        self.direction_optimizer = direction_optimizer
        self.noise_optimizer = noise_optimizer
        self.direction_scheduler = direction_scheduler
        self.noise_scheduler = noise_scheduler
        self.schedule = schedule
        self.artifact = artifact
        direction_parameters = {id(value) for value in self.direction_model.parameters()}
        noise_parameters = {id(value) for value in self.noise_model.parameters()}
        if not direction_parameters or not noise_parameters or direction_parameters & noise_parameters:
            raise ValueError("DLD predictor parameter sets must be non-empty and disjoint")
        if {id(value) for group in direction_optimizer.param_groups for value in group["params"]} != direction_parameters:
            raise ValueError("DLD direction optimizer is not bound exactly to direction_model")
        if {id(value) for group in noise_optimizer.param_groups for value in group["params"]} != noise_parameters:
            raise ValueError("DLD noise optimizer is not bound exactly to noise_model")
        self.direction_ema = None if ema_decay is None else ModelEMA(self.direction_model, ema_decay)
        self.noise_ema = None if ema_decay is None else ModelEMA(self.noise_model, ema_decay)
        self._position = {int(index): position for position, index in enumerate(artifact.global_indices)}

    def _lookup(self, sample_indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if sample_indices.ndim != 1 or sample_indices.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        } or torch.unique(sample_indices).numel() != sample_indices.numel():
            raise ValueError("DLD batch sample indices must be unique integer [B]")
        positions: list[int] = []
        for value in sample_indices.detach().cpu().tolist():
            if int(value) not in self._position:
                raise KeyError(f"DLD artifact is missing stable sample index {int(value)}")
            positions.append(self._position[int(value)])
        y0 = torch.as_tensor(self.artifact.y0[positions], device=self.device, dtype=torch.float32)
        yn = torch.as_tensor(self.artifact.yn[positions], device=self.device, dtype=torch.float32)
        features = torch.as_tensor(
            self.artifact.condition_features[positions], device=self.device, dtype=torch.float32
        )
        return y0, yn, features

    def train_step(self, sample_indices: Tensor) -> dict[str, float]:
        y0, yn, features = self._lookup(sample_indices)
        yd = yn - y0
        batch = y0.shape[0]
        timestep = torch.randint(0, self.schedule.timesteps, (batch,), device=self.device)
        epsilon = torch.randn_like(y0)
        y_t = sample_forward_state(y0, yd, timestep, epsilon, self.schedule)
        self.direction_model.train(); self.noise_model.train()
        predicted_direction = self.direction_model(y_t, yn, features, timestep)
        predicted_noise = self.noise_model(y_t, yn, features, timestep)
        objective = dld_objective(predicted_direction, yd, predicted_noise, epsilon)

        self.direction_optimizer.zero_grad(set_to_none=True)
        objective.direction_loss.backward()
        direction_norm = self._gradient_norm(self.direction_model)
        self.direction_optimizer.step()
        if self.direction_ema is not None:
            self.direction_ema.update(self.direction_model)

        self.noise_optimizer.zero_grad(set_to_none=True)
        objective.noise_loss.backward()
        noise_norm = self._gradient_norm(self.noise_model)
        self.noise_optimizer.step()
        if self.noise_ema is not None:
            self.noise_ema.update(self.noise_model)
        return {
            "direction_loss": float(objective.direction_loss.detach()),
            "noise_loss": float(objective.noise_loss.detach()),
            "direction_gradient_norm": direction_norm,
            "noise_gradient_norm": noise_norm,
            "direction_parameter_norm": self._parameter_norm(self.direction_model),
            "noise_parameter_norm": self._parameter_norm(self.noise_model),
            "predicted_direction_rms": self._tensor_rms(predicted_direction),
            "predicted_noise_rms": self._tensor_rms(predicted_noise),
            "target_direction_rms": self._tensor_rms(yd),
            "target_noise_rms": self._tensor_rms(epsilon),
            "samples": float(batch),
        }

    @staticmethod
    def _gradient_norm(model: nn.Module) -> float:
        values = [parameter.grad.detach().square().sum() for parameter in model.parameters() if parameter.grad is not None]
        if not values:
            raise RuntimeError("DLD predictor received no gradients")
        result = float(torch.stack(values).sum().sqrt())
        if not np.isfinite(result):
            raise ValueError("DLD gradient norm is non-finite")
        return result

    @staticmethod
    def _parameter_norm(model: nn.Module) -> float:
        values = [parameter.detach().square().sum() for parameter in model.parameters()]
        result = float(torch.stack(values).sum().sqrt())
        if not np.isfinite(result):
            raise ValueError("DLD parameter norm is non-finite")
        return result

    @staticmethod
    def _tensor_rms(value: Tensor) -> float:
        result = float(value.detach().square().mean().sqrt())
        if not np.isfinite(result):
            raise ValueError("DLD predictor telemetry is non-finite")
        return result

    def prediction_models(self) -> tuple[nn.Module, nn.Module]:
        if self.direction_ema is not None and self.noise_ema is not None:
            return self.direction_ema.model, self.noise_ema.model
        return self.direction_model, self.noise_model

    def state_dict(self) -> dict[str, Any]:
        return {
            "direction_model": self.direction_model.state_dict(),
            "noise_model": self.noise_model.state_dict(),
            "direction_optimizer": self.direction_optimizer.state_dict(),
            "noise_optimizer": self.noise_optimizer.state_dict(),
            "direction_scheduler": None if self.direction_scheduler is None else self.direction_scheduler.state_dict(),
            "noise_scheduler": None if self.noise_scheduler is None else self.noise_scheduler.state_dict(),
            "direction_ema": None if self.direction_ema is None else self.direction_ema.state_dict(),
            "noise_ema": None if self.noise_ema is None else self.noise_ema.state_dict(),
            "schedule_identity_hash": self.schedule.identity_hash,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schedule_identity_hash") != self.schedule.identity_hash:
            raise ValueError("DLD checkpoint schedule identity mismatch")
        self.direction_model.load_state_dict(value["direction_model"])
        self.noise_model.load_state_dict(value["noise_model"])
        self.direction_optimizer.load_state_dict(value["direction_optimizer"])
        self.noise_optimizer.load_state_dict(value["noise_optimizer"])
        for scheduler, name in ((self.direction_scheduler, "direction_scheduler"), (self.noise_scheduler, "noise_scheduler")):
            saved = value.get(name)
            if scheduler is None and saved is not None:
                raise ValueError(f"DLD checkpoint unexpectedly contains {name}")
            if scheduler is not None:
                if saved is None:
                    raise ValueError(f"DLD checkpoint is missing {name}")
                scheduler.load_state_dict(saved)
        for ema, name in ((self.direction_ema, "direction_ema"), (self.noise_ema, "noise_ema")):
            saved = value.get(name)
            if ema is None and saved is not None:
                raise ValueError(f"DLD checkpoint unexpectedly contains {name}")
            if ema is not None:
                if saved is None:
                    raise ValueError(f"DLD checkpoint is missing {name}")
                ema.load_state_dict(saved)


__all__ = ["DLDAlgorithm"]
