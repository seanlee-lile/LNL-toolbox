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
