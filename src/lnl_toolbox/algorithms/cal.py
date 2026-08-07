from __future__ import annotations

"""CORES2 sample sieve and second-order CAL objective."""

import torch
from torch import Tensor
from torch.nn import functional as F
from collections.abc import Mapping
from bisect import bisect_right


def resolve_confidence_weight(
    epoch: int,
    default: float,
    schedule: Mapping[str, object] | None = None,
) -> float:
    """Resolve the official CAL SegAlpha value for a zero-based epoch.

    The official CIFAR recipe starts the CORES² sieve with ``alpha=0`` and
    changes it by linear interpolation between configured milestones. Keeping this schedule in the
    algorithm module makes it reusable by runners without coupling it to a
    particular experiment lifecycle.
    """
    if epoch < 0:
        raise ValueError("CAL schedule epoch must be non-negative")
    if schedule is None:
        value = float(default)
    else:
        milestones = [int(value) for value in schedule.get("milestones", [])]
        values = [float(value) for value in schedule.get("values", [])]
        if not values or len(values) != len(milestones):
            raise ValueError("CAL confidence schedule requires one value per milestone")
        if any(value < 0 for value in milestones) or milestones != sorted(milestones):
            raise ValueError("CAL confidence schedule milestones must be sorted and non-negative")
        if any(not torch.isfinite(torch.tensor(value)) for value in values):
            raise ValueError("CAL confidence schedule values must be finite")
        if any(right <= left for left, right in zip(milestones, milestones[1:])):
            raise ValueError("CAL confidence schedule milestones must be strictly increasing")
        if epoch <= milestones[0]:
            value = values[0]
        elif epoch >= milestones[-1]:
            value = values[-1]
        else:
            right = bisect_right(milestones, epoch)
            left_epoch, right_epoch = milestones[right - 1], milestones[right]
            left_value, right_value = values[right - 1], values[right]
            fraction = (epoch - left_epoch) / float(right_epoch - left_epoch)
            value = left_value + fraction * (right_value - left_value)
    if not torch.isfinite(torch.tensor(value)) or value < 0:
        raise ValueError("CAL confidence weight must be finite and non-negative")
    return value


def cal_all_class_losses(logits: Tensor, eps: float = 1.0e-5) -> Tensor:
    """Return the official CAL ``-log(softmax(logits) + eps)`` matrix."""
    if logits.ndim != 2:
        raise ValueError("CAL logits must have shape [B,C]")
    if not torch.isfinite(torch.tensor(float(eps))) or eps <= 0.0:
        raise ValueError("CAL epsilon must be finite and positive")
    return -torch.log(F.softmax(logits, dim=1) + float(eps))


def cores2_adjusted_losses(
    logits: Tensor,
    noisy_targets: Tensor,
    noisy_prior: Tensor,
    confidence_weight: float,
) -> Tensor:
    if logits.ndim != 2 or noisy_targets.shape != (logits.shape[0],):
        raise ValueError("CAL logits/targets shapes are invalid")
    prior = noisy_prior.to(device=logits.device, dtype=logits.dtype)
    if prior.shape != (logits.shape[1],) or bool((prior < 0).any()):
        raise ValueError("CAL noisy prior must be non-negative [C]")
    prior = prior / prior.sum().clamp_min(torch.finfo(logits.dtype).tiny)
    base_losses = cal_all_class_losses(logits, eps=1.0e-8)
    peer_losses = cal_all_class_losses(logits, eps=1.0e-5)
    observed = base_losses.gather(1, noisy_targets.long()[:, None]).squeeze(1)
    return observed - float(confidence_weight) * (peer_losses * prior).sum(dim=1)


def cal_transition_indicators(
    proxy_targets: Tensor,
    noisy_targets: Tensor,
    retained_mask: Tensor,
    num_classes: int,
) -> Tensor:
    if proxy_targets.shape != noisy_targets.shape or retained_mask.shape != proxy_targets.shape:
        raise ValueError("CAL proxy/noisy/mask shapes differ")
    result = torch.zeros(
        proxy_targets.shape[0], num_classes, num_classes,
        dtype=torch.float32, device=proxy_targets.device,
    )
    rows = torch.arange(proxy_targets.shape[0], device=proxy_targets.device)[retained_mask]
    result[rows, proxy_targets[retained_mask].long(), noisy_targets[retained_mask].long()] = 1.0
    return result


def cal_covariance_correction(
    all_class_losses: Tensor,
    proxy_targets: Tensor,
    noisy_targets: Tensor,
    retained_mask: Tensor,
    proxy_class_prior: Tensor,
    reference_loss_means: Tensor,
) -> tuple[Tensor, Tensor]:
    if all_class_losses.ndim != 2:
        raise ValueError("CAL all-class losses must have shape [B,C]")
    classes = all_class_losses.shape[1]
    if reference_loss_means.shape != (classes, classes):
        raise ValueError("CAL reference loss means must have shape [C,C]")
    prior = proxy_class_prior.to(all_class_losses)
    if prior.shape != (classes,):
        raise ValueError("CAL proxy class prior must have shape [C]")
    correction = all_class_losses.sum() * 0.0
    detached_means = reference_loss_means.detach().clone().to(all_class_losses)
    for proxy_class in range(classes):
        class_mask = retained_mask & proxy_targets.eq(proxy_class)
        count = int(class_mask.sum().item())
        if count == 0:
            continue
        losses = all_class_losses[class_mask]
        detached_means[proxy_class] = losses.detach().mean(dim=0)
        noisy = noisy_targets[class_mask]
        for noisy_class in range(classes):
            indicator = noisy.eq(noisy_class).to(losses.dtype)
            centered_indicator = indicator - indicator.mean()
            centered_loss = losses[:, noisy_class] - reference_loss_means[proxy_class, noisy_class]
            correction = correction + prior[proxy_class] * (centered_indicator * centered_loss).mean()
    return correction, detached_means


def cal_objective(
    logits: Tensor,
    noisy_targets: Tensor,
    proxy_targets: Tensor,
    retained_mask: Tensor,
    noisy_prior: Tensor,
    proxy_class_prior: Tensor,
    reference_loss_means: Tensor,
    *,
    confidence_weight: float,
) -> tuple[Tensor, Tensor]:
    all_losses = cal_all_class_losses(logits)
    base = cores2_adjusted_losses(
        logits, noisy_targets, noisy_prior, confidence_weight
    ).mean()
    correction, means = cal_covariance_correction(
        all_losses, proxy_targets, noisy_targets, retained_mask,
        proxy_class_prior, reference_loss_means,
    )
    objective = base - correction
    if not bool(torch.isfinite(objective).item()):
        raise ValueError("CAL objective is non-finite")
    return objective, means


__all__ = [
    "cal_all_class_losses", "cal_covariance_correction", "cal_objective",
    "cal_transition_indicators", "cores2_adjusted_losses",
    "resolve_confidence_weight",
]
