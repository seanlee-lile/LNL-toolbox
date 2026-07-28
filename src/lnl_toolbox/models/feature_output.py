from __future__ import annotations

"""Model-output contract for objectives that consume learned features."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FeatureOutput:
    """Logits and representation from one model forward pass."""

    logits: Tensor
    features: Tensor

    def __post_init__(self) -> None:
        if self.logits.ndim != 2 or self.features.ndim != 2:
            raise ValueError("FeatureOutput tensors must have shape [B, D]")
        if self.logits.shape[0] != self.features.shape[0]:
            raise ValueError("FeatureOutput logits and features must share batch size")
        if not torch.isfinite(self.logits).all() or not torch.isfinite(self.features).all():
            raise ValueError("FeatureOutput tensors must be finite")


def forward_with_features(model: nn.Module, inputs: Tensor) -> FeatureOutput:
    """Call the opt-in feature interface without changing ordinary forward()."""

    method = getattr(model, "forward_with_features", None)
    if not callable(method):
        raise TypeError(
            f"model {type(model).__name__} does not expose forward_with_features()"
        )
    output: Any = method(inputs)
    if isinstance(output, FeatureOutput):
        return output
    if isinstance(output, dict) and {"logits", "features"}.issubset(output):
        return FeatureOutput(output["logits"], output["features"])
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return FeatureOutput(output[0], output[1])
    raise TypeError("forward_with_features() must return FeatureOutput or (logits, features)")


def classifier_parameters(model: nn.Module) -> tuple[Tensor, Tensor | None]:
    """Find the final linear classifier parameters through a generic model contract."""

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Linear):
        return classifier.weight, classifier.bias
    if isinstance(classifier, nn.Sequential):
        for module in reversed(tuple(classifier)):
            if isinstance(module, nn.Linear):
                return module.weight, module.bias
    raise TypeError(f"model {type(model).__name__} has no discoverable linear classifier")


def validate_objective(value: Tensor) -> Tensor:
    if not torch.is_tensor(value) or value.ndim != 0:
        raise ValueError("objective consumer must return a scalar tensor")
    if not value.requires_grad:
        raise ValueError("objective consumer output must require gradients")
    if not bool(torch.isfinite(value.detach()).item()):
        raise ValueError("objective consumer output must be finite")
    return value


__all__ = ["FeatureOutput", "classifier_parameters", "forward_with_features", "validate_objective"]
