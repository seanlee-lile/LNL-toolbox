from __future__ import annotations

"""Two-network CA2C workflow with global candidate memory."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
import yaml

from lnl_toolbox.algorithms.ca2c import (
    CandidateMemory,
    cross_guidance,
    negative_label_objective,
    partial_label_objective,
)
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state, read_checkpoint, restore_rng_state
from lnl_toolbox.training.interfaces import RunContext
from lnl_toolbox.training.experiment import build_optimizer, build_scheduler
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.reproduction_data import build_reproduction_model, prepare_noisy_classification


def _confidence_penalty(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)
    return torch.sum(probability * torch.log(probability.clamp_min(1e-8)), dim=1).mean()


@torch.inference_mode()
def _evaluate(models, loader, criterion, device):
    for model in models: model.eval()
    total = correct = 0; loss_sum = 0.0
    for batch in loader:
        targets = batch["target"].to(device); logits = sum(model(batch["input"].to(device)) for model in models) / len(models)
        loss_sum += float(criterion(logits, targets).sum()); total += targets.numel(); correct += int(logits.argmax(1).eq(targets).sum())
    return {"loss": loss_sum / total, "accuracy": correct / total}


def run_ca2c_experiment(config: dict[str, Any], output_dir=None, resume=None, *, context: RunContext | None = None) -> Path:
    config = deepcopy(config); seed = int(config.get("seed", 1)); seed_everything(seed)
    device = resolve_device(str(config.get("trainer", {}).get("device", "auto")))
    run_dir = Path(resume).resolve().parent if resume else Path(output_dir or Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_noisy_classification(config, run_dir, seed); classes = data.num_classes
    p_model = build_reproduction_model(config["model"], config["data"], classes).to(device); n_model = build_reproduction_model(config["model"], config["data"], classes).to(device)
    p_optimizer = build_optimizer(p_model, config["optimizer"]); n_optimizer = build_optimizer(n_model, config["optimizer"])
    epochs = int(config["trainer"]["epochs"]); p_scheduler = build_scheduler(p_optimizer, config.get("scheduler"), epochs); n_scheduler = build_scheduler(n_optimizer, config.get("scheduler"), epochs)
    memory = CandidateMemory.create(torch.as_tensor(data.train_indices), classes); criterion = CrossEntropyLoss().to(device); start = 0; rows = []
    payload = read_checkpoint(resume, device) if resume else None
    if payload is not None:
        if payload.get("method") != "ca2c" or payload.get("config") != config: raise ValueError("CA2C resume identity mismatch")
        p_model.load_state_dict(payload["p_model"]); n_model.load_state_dict(payload["n_model"]); p_optimizer.load_state_dict(payload["p_optimizer"]); n_optimizer.load_state_dict(payload["n_optimizer"])
        if p_scheduler is not None: p_scheduler.load_state_dict(payload["p_scheduler"]); n_scheduler.load_state_dict(payload["n_scheduler"])
        memory = CandidateMemory.from_state_dict(payload["candidate_memory"])
        if payload.get("candidate_memory_hash") not in {None, memory.fingerprint()}:
            raise ValueError("CA2C candidate memory checkpoint hash mismatch")
        start = int(payload["completed_epoch"]) + 1
        rows = list(payload.get("metrics", []))
        restore_rng_state(payload["rng_state"])
    ca2c_config = dict(config.get("ca2c", {}) or {})
    warmup_epochs = int(ca2c_config.get("warmup_epochs", 0))
    k = int(ca2c_config["candidate_k"])
    mixing = float(
        ca2c_config.get(
            "lambda_",
            ca2c_config.get("lambda", ca2c_config.get("hard_weight", 0.99)),
        )
    )
    robust_weight = float(ca2c_config.get("robust_weight", 0.8))
    session = context.session if context is not None and context.state.get("lifecycle_active") else None
    if session is not None:
        session.start_phase("co_learning", total_units=epochs)
    for epoch in range(start, epochs):
        p_model.train(); n_model.train(); total = correct = 0; loss_sum = 0.0
        for batch in data.train_loader:
            inputs, targets, indices = batch["input"].to(device), batch["target"].to(device), batch["index"].to(device); p_logits, n_logits = p_model(inputs), n_model(inputs)
            if epoch < warmup_epochs:
                p_loss = criterion(p_logits, targets).mean() + _confidence_penalty(p_logits)
                n_loss = criterion(n_logits, targets).mean() + _confidence_penalty(n_logits)
                p_candidates = torch.zeros_like(p_logits, dtype=torch.bool)
                n_candidates = torch.zeros_like(n_logits, dtype=torch.bool)
                p_candidates.scatter_(1, p_logits.detach().topk(k, dim=1).indices, True)
                n_candidates.scatter_(1, n_logits.detach().topk(k, dim=1).indices, True)
                memory.update_(indices, p_candidates); memory.update_(indices, n_candidates)
            else:
                candidates, complements = cross_guidance(p_logits, n_logits, k); memory.update_(indices, candidates); soft_targets = memory.targets(indices)
                confidence = soft_targets.max(dim=1).values
                p_base = partial_label_objective(
                    p_logits,
                    soft_targets,
                    mixing,
                    confidence=confidence,
                )
                probability = torch.softmax(n_logits, dim=1)
                n_base = negative_label_objective(n_logits, complements)
                strong = batch.get("strong_input", batch["input"]).to(device)
                p_strong = p_model(strong); n_strong = n_model(strong)
                p_consistency = F.cross_entropy(p_strong, p_logits.detach().argmax(1))
                n_consistency = F.cross_entropy(n_strong, n_logits.detach().argmax(1))
                p_loss = robust_weight * p_base + (1.0 - robust_weight) * p_consistency
                n_loss = robust_weight * n_base + (1.0 - robust_weight) * n_consistency
            p_optimizer.zero_grad(set_to_none=True); p_loss.backward(); p_optimizer.step(); n_optimizer.zero_grad(set_to_none=True); n_loss.backward(); n_optimizer.step()
            ensemble = (p_logits.detach() + n_logits.detach()) / 2; total += targets.numel(); correct += int(ensemble.argmax(1).eq(targets).sum()); loss_sum += float((p_loss.detach() + n_loss.detach()) / 2) * targets.numel()
        validation = _evaluate((p_model, n_model), data.validation_loader, criterion, device); test = _evaluate((p_model, n_model), data.test_loader, criterion, device)
        row = standardize_epoch_row({
            "epoch": epoch + 1,
            "phase": "warmup" if epoch < warmup_epochs else "robust",
            "train_loss": loss_sum / total,
            "train_accuracy": correct / total,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "learning_rate": p_optimizer.param_groups[0]["lr"],
            "method": "ca2c",
            "candidate_memory_hash": memory.fingerprint(),
        })
        rows.append(row)
        if session is not None:
            session.log_epoch(
                epoch + 1,
                phase=str(row.get("phase", "train")),
                **{key: value for key, value in row.items()
                   if key not in {"event", "epoch", "phase", "seq"}},
            )
        print(
            f"CA2C epoch {epoch + 1}/{epochs} phase={row['phase']} "
            f"loss={row['train_loss']:.5f} val={row['validation_accuracy']:.4f} "
            f"test={row['test_accuracy']:.4f}",
            flush=True,
        )
        if p_scheduler is not None: p_scheduler.step(); n_scheduler.step()
        atomic_save({
            "method": "ca2c",
            "config": config,
            "phase": "warmup" if epoch < warmup_epochs else "robust",
            "p_model": p_model.state_dict(),
            "n_model": n_model.state_dict(),
            "p_optimizer": p_optimizer.state_dict(),
            "n_optimizer": n_optimizer.state_dict(),
            "p_scheduler": None if p_scheduler is None else p_scheduler.state_dict(),
            "n_scheduler": None if n_scheduler is None else n_scheduler.state_dict(),
            "candidate_memory": memory.state_dict(),
            "candidate_memory_hash": memory.fingerprint(),
            "completed_epoch": epoch,
            "metrics": rows,
            "rng_state": capture_rng_state(),
        }, run_dir / "last.pt")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if session is None:
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        session.end_phase("co_learning", completed_units=max(0, epochs - start))
        session.emit("final", phase="evaluation", method="ca2c",
                     completed_epochs=epochs,
                     test_accuracy=rows[-1].get("test_accuracy") if rows else None)
    if rows: write_training_curves_svg(rows, run_dir / "training_curves.svg")
    return run_dir


__all__ = ["run_ca2c_experiment"]
