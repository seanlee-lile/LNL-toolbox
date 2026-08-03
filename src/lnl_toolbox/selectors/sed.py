from __future__ import annotations

"""SED clean selection and confidence reweighting used by FINE."""

import torch

from .base import SelectionInput, SelectionResult, validate_selection_input


class SEDSelector:
    """Select samples whose detached score is below a configurable threshold."""

    def __init__(self, threshold: float = 0.5, keep_rate: float | None = None) -> None:
        if threshold < 0.0:
            raise ValueError("SED threshold must be non-negative")
        if keep_rate is not None and not 0.0 < keep_rate <= 1.0:
            raise ValueError("SED keep_rate must be in (0, 1]")
        self.threshold = float(threshold)
        self.keep_rate = None if keep_rate is None else float(keep_rate)

    def select(self, selection_input: SelectionInput) -> SelectionResult:
        batch = validate_selection_input(selection_input)
        scores = selection_input.scores
        mask = scores <= self.threshold
        if self.keep_rate is not None:
            count = max(1, int(torch.ceil(torch.tensor(batch * self.keep_rate)).item()))
            order = torch.argsort(scores, stable=True)
            mask = torch.zeros_like(mask)
            mask[order[:count]] = True
        if not bool(mask.any()):
            mask[torch.argmin(scores)] = True
        return SelectionResult(mask, {"selected_samples": float(mask.sum()), "selected_ratio": float(mask.float().mean())})


class SelfAdaptiveClassSelector:
    """Official SCS class-adaptive threshold over an epoch snapshot."""

    def __init__(
        self,
        num_classes: int,
        momentum: float = 0.999,
        *,
        quantile: float | None = 0.8,
        maximum_threshold: float | None = 0.95,
    ) -> None:
        if num_classes < 2 or not 0.0 <= momentum < 1.0:
            raise ValueError("invalid SCS class count or momentum")
        if quantile is not None and not 0.0 <= quantile <= 1.0:
            raise ValueError("SCS quantile must be in [0, 1]")
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.quantile = quantile
        self.maximum_threshold = maximum_threshold
        self.local = torch.full((num_classes,), 1.0 / num_classes)
        self.global_threshold = torch.tensor(1.0 / num_classes)

    @torch.no_grad()
    def select_epoch(self, probabilities: torch.Tensor, noisy_targets: torch.Tensor) -> torch.Tensor:
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError("SCS probabilities must have shape [N, C]")
        if noisy_targets.shape != (probabilities.shape[0],):
            raise ValueError("SCS targets must have shape [N]")
        self.local = self.local.to(probabilities.device)
        self.global_threshold = self.global_threshold.to(probabilities.device)
        given = probabilities.gather(1, noisy_targets.long()[:, None]).squeeze(1)
        statistic = (
            given.mean()
            if self.quantile is None
            else torch.quantile(given, self.quantile)
        )
        self.global_threshold.mul_(self.momentum).add_(statistic * (1.0 - self.momentum))
        if self.maximum_threshold is not None:
            self.global_threshold.clamp_(0.0, float(self.maximum_threshold))
        self.local.mul_(self.momentum).add_(
            probabilities.mean(dim=0) * (1.0 - self.momentum)
        )
        modulation = self.local / self.local.max().clamp_min(torch.finfo(probabilities.dtype).tiny)
        return given.ge(self.global_threshold * modulation[noisy_targets.long()])

    def state_dict(self) -> dict[str, object]:
        return {"local": self.local.detach().cpu(), "global_threshold": self.global_threshold.detach().cpu()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        local = torch.as_tensor(state["local"], dtype=torch.float32)
        threshold = torch.as_tensor(state["global_threshold"], dtype=torch.float32)
        if local.shape != (self.num_classes,) or threshold.numel() != 1:
            raise ValueError("SCS state shape mismatch")
        self.local = local
        self.global_threshold = threshold.reshape(())


class SelfAdaptiveConfidenceReweighting:
    """Official SCR class-wise Gaussian confidence weight."""

    def __init__(self, num_classes: int, momentum: float = 0.99, n_sigma: float = 2.0) -> None:
        if num_classes < 2 or not 0.0 <= momentum < 1.0 or n_sigma <= 0.0:
            raise ValueError("invalid SCR configuration")
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.n_sigma = float(n_sigma)
        self.mean = torch.full((num_classes,), 1.0 / num_classes)
        self.variance = torch.ones(num_classes)

    @torch.no_grad()
    def weights(self, probabilities: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.to(probabilities.device)
        self.variance = self.variance.to(probabilities.device)
        maximum, predicted = probabilities.max(dim=1)
        current_mean = torch.zeros_like(self.mean)
        current_variance = torch.ones_like(self.variance)
        for class_index in range(self.num_classes):
            values = maximum[predicted == class_index]
            if values.numel() > 1:
                current_mean[class_index] = values.mean()
                current_variance[class_index] = values.var(unbiased=True)
        self.mean.mul_(self.momentum).add_(current_mean * (1.0 - self.momentum))
        self.variance.mul_(self.momentum).add_(current_variance * (1.0 - self.momentum))
        mean = self.mean[predicted]
        variance = self.variance[predicted].clamp_min(torch.finfo(probabilities.dtype).tiny)
        deficit = torch.clamp(maximum - mean, max=0.0)
        return torch.exp(-(deficit.square() / (2.0 * variance / self.n_sigma**2)))

    def state_dict(self) -> dict[str, object]:
        return {"mean": self.mean.detach().cpu(), "variance": self.variance.detach().cpu()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        mean = torch.as_tensor(state["mean"], dtype=torch.float32)
        variance = torch.as_tensor(state["variance"], dtype=torch.float32)
        if mean.shape != (self.num_classes,) or variance.shape != (self.num_classes,):
            raise ValueError("SCR state shape mismatch")
        self.mean, self.variance = mean, variance


SCS = SelfAdaptiveClassSelector
SCR = SelfAdaptiveConfidenceReweighting

__all__ = [
    "SCR",
    "SCS",
    "SEDSelector",
    "SelfAdaptiveClassSelector",
    "SelfAdaptiveConfidenceReweighting",
]
