from __future__ import annotations

import torch


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
        loss_sum += float(loss(logits, targets).item()) * count
        correct += int((logits.argmax(1) == targets).sum().item())
        samples += count
    return {"loss": loss_sum / samples, "accuracy": correct / samples, "samples": float(samples)}
