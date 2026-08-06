from __future__ import annotations

"""MentorNet curriculum features and reusable continuous-weight providers."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from lnl_toolbox.models.mentornet import build_mentor_model
from lnl_toolbox.training.mentor_artifacts import MentorArtifact
from lnl_toolbox.treatments.weights import (
    SupervisedWeightInput,
    WeightResult,
)


@dataclass
class MovingPercentileState:
    percentile: float = 0.75
    decay: float = 0.95
    value: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.percentile < 1.0:
            raise ValueError("percentile must satisfy 0 < percentile < 1")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must satisfy 0 <= decay < 1")

    def update(self, losses: torch.Tensor) -> float:
        current = float(torch.quantile(losses.detach(), self.percentile).item())
        self.value = (
            current
            if self.value is None
            else self.decay * self.value + (1.0 - self.decay) * current
        )
        return self.value

    def state_dict(self) -> dict[str, Any]:
        return {
            "percentile": self.percentile,
            "decay": self.decay,
            "value": self.value,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state.get("percentile", -1)) != self.percentile:
            raise ValueError("moving percentile configuration mismatch")
        if float(state.get("decay", -1)) != self.decay:
            raise ValueError("moving percentile decay mismatch")
        value = state.get("value")
        self.value = None if value is None else float(value)


class MentorNetWeightProvider:
    """Consume a frozen MentorArtifact during ordinary StudentNet training."""

    def __init__(
        self,
        artifact_path: str,
        total_epochs: int,
        *,
        percentile: float = 0.75,
        decay: float = 0.95,
        burn_in_fraction: float = 0.2,
        burn_in_epoch: int | None = None,
        fixed_epoch_after_burn_in: bool = False,
        fixed_label: int | None = 0,
        dropout_schedule: Sequence[Sequence[float]] = (),
        seed: int = 0,
        zero_weight_policy: str = "error",
    ) -> None:
        self.artifact_path = str(Path(artifact_path))
        self.artifact = MentorArtifact.load(self.artifact_path)
        self.model = build_mentor_model(dict(self.artifact.architecture))
        self.model.load_state_dict(dict(self.artifact.model_state))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.total_epochs = int(total_epochs)
        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        self.burn_in_fraction = float(burn_in_fraction)
        if not 0.0 <= self.burn_in_fraction <= 1.0:
            raise ValueError("burn_in_fraction must be within [0, 1]")
        self.burn_in_epoch = None if burn_in_epoch is None else int(burn_in_epoch)
        if self.burn_in_epoch is not None and not 0 <= self.burn_in_epoch <= 100:
            raise ValueError("burn_in_epoch must be within [0, 100]")
        self.fixed_epoch_after_burn_in = bool(fixed_epoch_after_burn_in)
        self.fixed_label = fixed_label
        self.moving = MovingPercentileState(percentile, decay)
        self.dropout_schedule = tuple(
            (float(rate), int(epochs)) for rate, epochs in dropout_schedule
        )
        if any(not 0 <= rate <= 1 or epochs <= 0 for rate, epochs in self.dropout_schedule):
            raise ValueError("dropout schedule entries must be [rate, positive_epochs]")
        self.generator = torch.Generator().manual_seed(int(seed))
        self.zero_weight_policy = str(zero_weight_policy)
        if self.zero_weight_policy not in {"error", "all"}:
            raise ValueError("zero_weight_policy must be 'error' or 'all'")

    def _dropout_rate(self, epoch: int) -> float:
        boundary = 0
        for rate, duration in self.dropout_schedule:
            boundary += duration
            if epoch < boundary:
                return rate
        return 0.0

    def compute(self, weight_input: SupervisedWeightInput) -> WeightResult:
        losses = weight_input.per_sample_loss.detach()
        epoch = int(weight_input.metadata.get("epoch", 0))
        moving = self.moving.update(losses)
        configured_burn_in = self.burn_in_epoch
        if configured_burn_in is None:
            burn_in = epoch / self.total_epochs < self.burn_in_fraction
            mentor_epoch = epoch
        else:
            mentor_epoch = min(epoch, configured_burn_in)
            burn_in = mentor_epoch < max(0, configured_burn_in - 1)
        if burn_in:
            weights = torch.ones_like(losses)
        else:
            labels = (
                weight_input.noisy_targets.detach()
                if self.fixed_label is None
                else torch.full_like(
                    weight_input.noisy_targets, int(self.fixed_label)
                )
            )
            epoch_percentage = min(99, mentor_epoch if self.fixed_epoch_after_burn_in
                                   else int(100 * epoch / self.total_epochs))
            epochs = torch.full_like(labels, epoch_percentage)
            self.model.to(losses.device)
            with torch.no_grad():
                weights = self.model(
                    losses,
                    losses - moving,
                    labels,
                    epochs,
                )
        rate = self._dropout_rate(mentor_epoch)
        if rate:
            keep = torch.rand(
                weights.shape,
                generator=self.generator,
                device="cpu",
            ).to(weights.device) >= rate
            weights = weights * keep
        if not bool((weights > 0).any()):
            if self.zero_weight_policy == "error":
                raise ValueError("MentorNet produced an all-zero batch")
            weights = torch.ones_like(weights)
        weights = weights.clamp(0, 1).detach()
        return WeightResult(
            weights,
            {
                "weight_mean": float(weights.mean().item()),
                "zero_weight_ratio": float((weights == 0).float().mean().item()),
                "moving_percentile": float(moving),
            },
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "artifact_hash": self.artifact.artifact_hash,
            "total_epochs": self.total_epochs,
            "burn_in_epoch": self.burn_in_epoch,
            "fixed_epoch_after_burn_in": self.fixed_epoch_after_burn_in,
            "moving": self.moving.state_dict(),
            "rng_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("artifact_hash") != self.artifact.artifact_hash:
            raise ValueError("MentorArtifact identity mismatch")
        if "total_epochs" in state and int(state["total_epochs"]) != self.total_epochs:
            raise ValueError("MentorNet total_epochs mismatch")
        if "burn_in_epoch" in state and state["burn_in_epoch"] != self.burn_in_epoch:
            raise ValueError("MentorNet burn_in_epoch mismatch")
        if (
            "fixed_epoch_after_burn_in" in state
            and bool(state["fixed_epoch_after_burn_in"]) != self.fixed_epoch_after_burn_in
        ):
            raise ValueError("MentorNet fixed-epoch configuration mismatch")
        self.moving.load_state_dict(state["moving"])
        self.generator.set_state(torch.as_tensor(state["rng_state"]).cpu())
