from __future__ import annotations

"""Shared classification evaluation used across training algorithms."""

import torch

from lnl_toolbox.losses.torch_losses import validate_per_sample_loss


@torch.inference_mode()
def evaluate_classification(model, loader, loss, device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        logits = model(inputs)
        count = int(targets.numel())
        per_sample_loss = validate_per_sample_loss(loss(logits, targets), count)
        loss_sum += float(per_sample_loss.sum().item())
        correct += int((logits.argmax(1) == targets).sum().item())
        samples += count
    return {"loss": loss_sum / samples, "accuracy": correct / samples, "samples": float(samples)}


@torch.inference_mode()
def evaluate_model_group(models, loader, loss, device) -> dict[str, float]:
    """Evaluate every model and the mean-logit ensemble in one loader pass."""

    models.eval()
    names = tuple(models.models)
    loss_sums = {name: 0.0 for name in names}
    correct = {name: 0 for name in names}
    ensemble_loss_sum = 0.0
    ensemble_correct = 0
    samples = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        logits = models.logits(inputs)
        count = int(targets.numel())
        for name, output in logits.items():
            values = validate_per_sample_loss(loss(output, targets), count)
            loss_sums[name] += float(values.sum().item())
            correct[name] += int((output.argmax(1) == targets).sum().item())
        ensemble = torch.stack(tuple(logits.values()), dim=0).mean(dim=0)
        ensemble_values = validate_per_sample_loss(loss(ensemble, targets), count)
        ensemble_loss_sum += float(ensemble_values.sum().item())
        ensemble_correct += int((ensemble.argmax(1) == targets).sum().item())
        samples += count
    if samples == 0:
        raise ValueError("evaluation loader must contain at least one sample")
    metrics = {
        "loss": ensemble_loss_sum / samples,
        "accuracy": ensemble_correct / samples,
        "samples": float(samples),
    }
    for name in names:
        metrics[f"{name}_loss"] = loss_sums[name] / samples
        metrics[f"{name}_accuracy"] = correct[name] / samples
    return metrics
