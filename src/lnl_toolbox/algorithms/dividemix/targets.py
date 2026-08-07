from __future__ import annotations

"""Detached target construction for DivideMix co-refinement and co-guessing."""

import torch
from torch import Tensor, nn


def sharpen(probabilities: Tensor, temperature: float) -> Tensor:
    if probabilities.ndim != 2 or not torch.is_floating_point(probabilities):
        raise ValueError("probabilities must be floating point [B,C]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not bool(torch.isfinite(probabilities).all().item()) or bool((probabilities < 0).any().item()):
        raise ValueError("probabilities must be finite and non-negative")
    powered = probabilities.detach().pow(1.0 / float(temperature))
    denominator = powered.sum(1, keepdim=True)
    if bool((denominator <= 0).any().item()):
        raise ValueError("probability rows must have positive mass")
    return (powered / denominator).detach()


@torch.no_grad()
def co_refine(model: nn.Module, views: tuple[Tensor, ...], noisy_targets: Tensor, clean_probability: Tensor, temperature: float) -> Tensor:
    if not views:
        raise ValueError("co-refinement requires at least one augmentation")
    predictions = [torch.softmax(model(view), dim=1) for view in views]
    average = torch.stack(predictions).mean(0)
    if clean_probability.shape != noisy_targets.shape:
        raise ValueError("clean probability and noisy targets must align as [B]")
    if not bool(torch.isfinite(clean_probability).all().item()) or bool(((clean_probability < 0) | (clean_probability > 1)).any().item()):
        raise ValueError("clean probabilities must be finite and in [0,1]")
    if noisy_targets.numel() and (int(noisy_targets.min()) < 0 or int(noisy_targets.max()) >= average.shape[1]):
        raise ValueError("noisy targets are outside the model class range")
    one_hot = torch.nn.functional.one_hot(noisy_targets, average.shape[1]).to(average.dtype)
    blended = clean_probability[:, None] * one_hot + (1.0 - clean_probability[:, None]) * average
    return sharpen(blended, temperature)


@torch.no_grad()
def co_guess(model_a: nn.Module, model_b: nn.Module, views: tuple[Tensor, ...], temperature: float) -> Tensor:
    if not views:
        raise ValueError("co-guessing requires at least one augmentation")
    predictions = [torch.softmax(model(view), dim=1) for view in views for model in (model_a, model_b)]
    return sharpen(torch.stack(predictions).mean(0), temperature)
