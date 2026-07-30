from __future__ import annotations

"""Paper Eq. (1) minimum-volume transition-training primitives for PCSE.

The strictly diagonally dominant parameterization is a numerical-safety
implementation choice.  It is intentionally narrower than the paper's general
nonsingular row-stochastic transition assumption.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VolMinDiagnostics:
    determinant: float
    log_determinant: float
    minimum_singular_value: float
    condition_number: float


class DiagonallyDominantTransition(nn.Module):
    """Positive row-stochastic transition with diagonal entries above 0.5."""

    def __init__(
        self,
        num_classes: int,
        *,
        initial_flip_mass: float,
        max_flip_mass: float,
        temperature: float,
        seed: int,
    ) -> None:
        super().__init__()
        if num_classes < 3:
            raise ValueError("PCSE VolMin requires at least three classes")
        if not 0.0 < max_flip_mass < 0.5:
            raise ValueError("max_flip_mass must be in (0, 0.5)")
        if not 0.0 < initial_flip_mass < max_flip_mass:
            raise ValueError(
                "initial_flip_mass must be in (0, max_flip_mass)"
            )
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("transition temperature must be finite and positive")
        self.num_classes = int(num_classes)
        self.max_flip_mass = float(max_flip_mass)
        self.temperature = float(temperature)
        self.seed = int(seed)

        ratio = initial_flip_mass / max_flip_mass
        initial_logit = math.log(ratio / (1.0 - ratio))
        self.flip_logits = nn.Parameter(
            torch.full((num_classes,), initial_logit, dtype=torch.float64)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        off_diagonal = 1e-3 * torch.randn(
            num_classes,
            num_classes,
            generator=generator,
            dtype=torch.float64,
        )
        off_diagonal.fill_diagonal_(0.0)
        self.off_diagonal_logits = nn.Parameter(off_diagonal)

    def matrix(self, *, dtype: torch.dtype | None = None) -> Tensor:
        flip_mass = self.max_flip_mass * torch.sigmoid(self.flip_logits)
        mask = torch.eye(
            self.num_classes,
            dtype=torch.bool,
            device=self.off_diagonal_logits.device,
        )
        logits = self.off_diagonal_logits / self.temperature
        logits = logits.masked_fill(mask, float("-inf"))
        off_diagonal = torch.softmax(logits, dim=1).masked_fill(mask, 0.0)
        transition = off_diagonal * flip_mass[:, None]
        transition = transition + torch.diag(1.0 - flip_mass)
        if dtype is not None:
            transition = transition.to(dtype=dtype)
        return transition

    def forward(self) -> Tensor:
        return self.matrix()


def validate_trainable_transition(
    transition: Tensor,
    *,
    determinant_tolerance: float,
    condition_limit: float,
) -> tuple[Tensor, VolMinDiagnostics]:
    """Validate a differentiable transition without detaching its logdet."""

    if not torch.is_tensor(transition) or transition.ndim != 2:
        raise ValueError("VolMin transition must have shape [C, C]")
    rows, columns = transition.shape
    if rows < 3 or rows != columns:
        raise ValueError("VolMin transition must be square with C >= 3")
    if not torch.is_floating_point(transition):
        raise TypeError("VolMin transition must use a floating-point dtype")
    if not bool(torch.isfinite(transition).all().item()):
        raise ValueError("VolMin transition contains non-finite values")
    if bool((transition < 0.0).any().item()):
        raise ValueError("VolMin transition must be non-negative")
    if not torch.allclose(
        transition.sum(dim=1),
        torch.ones(rows, device=transition.device, dtype=transition.dtype),
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("VolMin transition rows must sum to one")
    if not bool(
        (
            torch.diagonal(transition)
            > transition.sum(dim=1) - torch.diagonal(transition)
        ).all().item()
    ):
        raise ValueError(
            "VolMin transition must remain strictly diagonally dominant"
        )

    sign, logabsdet = torch.linalg.slogdet(transition)
    singular_values = torch.linalg.svdvals(transition)
    smallest = singular_values[-1]
    largest = singular_values[0]
    condition = largest / smallest
    if not bool(torch.isfinite(sign).item()) or float(sign.detach().item()) <= 0.0:
        raise ValueError("VolMin transition determinant must be positive")
    if not bool(torch.isfinite(logabsdet).item()):
        raise ValueError("VolMin transition log determinant must be finite")
    if (
        not bool(torch.isfinite(smallest).item())
        or float(smallest.detach().item()) <= determinant_tolerance
    ):
        raise ValueError("VolMin transition is singular or below tolerance")
    if (
        not bool(torch.isfinite(condition).item())
        or float(condition.detach().item()) > condition_limit
    ):
        raise ValueError("VolMin transition exceeds the condition limit")
    diagnostics = VolMinDiagnostics(
        determinant=float(torch.exp(logabsdet).detach().item()),
        log_determinant=float(logabsdet.detach().item()),
        minimum_singular_value=float(smallest.detach().item()),
        condition_number=float(condition.detach().item()),
    )
    return logabsdet, diagnostics


def volmin_objective(
    logits: Tensor,
    noisy_targets: Tensor,
    transition: Tensor,
    *,
    lambda_volume: float,
    determinant_tolerance: float,
    condition_limit: float,
) -> tuple[Tensor, Mapping[str, float]]:
    """Return paper-direction ``mean NLL + lambda * logdet(T)``."""

    if logits.ndim != 2:
        raise ValueError("VolMin logits must have shape [B, C]")
    if noisy_targets.ndim != 1 or noisy_targets.shape[0] != logits.shape[0]:
        raise ValueError("VolMin targets must have shape [B]")
    if transition.shape != (logits.shape[1], logits.shape[1]):
        raise ValueError("VolMin logits and transition class counts differ")
    if noisy_targets.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("VolMin targets must use an integer dtype")
    if noisy_targets.numel() and (
        int(noisy_targets.min().item()) < 0
        or int(noisy_targets.max().item()) >= logits.shape[1]
    ):
        raise ValueError("VolMin targets are outside the class range")
    if not math.isfinite(lambda_volume) or lambda_volume <= 0.0:
        raise ValueError("lambda_volume must be finite and positive")

    logdet, diagnostics = validate_trainable_transition(
        transition,
        determinant_tolerance=determinant_tolerance,
        condition_limit=condition_limit,
    )
    clean_log_probability = torch.log_softmax(logits, dim=1)
    transition_log_probability = torch.log(transition)
    noisy_log_probability = torch.logsumexp(
        clean_log_probability[:, :, None]
        + transition_log_probability[None, :, :],
        dim=1,
    )
    classification_loss = F.nll_loss(
        noisy_log_probability,
        noisy_targets.to(dtype=torch.long),
        reduction="mean",
    )
    objective = classification_loss + lambda_volume * logdet
    if not bool(torch.isfinite(objective).item()):
        raise ValueError("VolMin objective is non-finite")
    return objective, {
        "classification_loss": float(classification_loss.detach().item()),
        "volume_regularizer": float(logdet.detach().item()),
        "objective": float(objective.detach().item()),
        "determinant": diagnostics.determinant,
        "minimum_singular_value": diagnostics.minimum_singular_value,
        "condition_number": diagnostics.condition_number,
    }


def build_volmin_optimizer(
    model: nn.Module,
    transition_model: DiagonallyDominantTransition,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Build the method-local joint optimizer with explicit parameter groups."""

    name = str(config.get("name", "")).strip().lower()
    if name != "adamw":
        raise ValueError("PCSE paper_volmin currently requires optimizer adamw")
    model_lr = float(config["model_lr"])
    transition_lr = float(config["transition_lr"])
    weight_decay = float(config.get("weight_decay", 0.0))
    if (
        not math.isfinite(model_lr)
        or model_lr <= 0.0
        or not math.isfinite(transition_lr)
        or transition_lr <= 0.0
        or not math.isfinite(weight_decay)
        or weight_decay < 0.0
    ):
        raise ValueError("VolMin optimizer values are invalid")
    return torch.optim.AdamW(
        [
            {
                "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
                "lr": model_lr,
                "weight_decay": weight_decay,
                "name": "model",
            },
            {
                "params": list(transition_model.parameters()),
                "lr": transition_lr,
                "weight_decay": 0.0,
                "name": "transition",
            },
        ]
    )
