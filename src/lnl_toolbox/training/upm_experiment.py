from __future__ import annotations

"""UPM's warm-up, posterior snapshot and alternating training lifecycle."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lnl_toolbox.algorithms.upm import estimate_clean_posterior, update_confusion_probabilities_, upm_soft_target_objective
from lnl_toolbox.data.multiclass_synthetic import MulticlassTensorDataset, generate_synthetic_multiclass
from lnl_toolbox.noise.estimators import PosteriorSnapshot
from lnl_toolbox.noise.generators import generate_symmetric
from lnl_toolbox.noise.upm import UPMNoiseState
from lnl_toolbox.runtime import seed_everything
from lnl_toolbox.training.interfaces import RunContext
from lnl_toolbox.training.reproduction_data import build_reproduction_model, prepare_noisy_classification


class _UPMMLP(nn.Module):
    def __init__(self, dimension: int, width: int, classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, width), nn.ReLU(), nn.Linear(width, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _dir(config: Mapping[str, Any], output_dir: str | Path | None, resume: str | Path | None) -> Path:
    path = Path(resume).resolve().parent if resume else (Path(output_dir).expanduser().resolve() if output_dir else Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _data(config: Mapping[str, Any], run_dir: Path) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, int, int, np.ndarray]:
    d = config.get("data", {}); classes, dimension = int(d.get("num_classes", 3)), int(d.get("dimension", 6)); n = int(d.get("train_size", 90)); val_n = int(d.get("validation_size", 30)); test_n = int(d.get("test_size", 30)); seed = int(config.get("seed", 1))
    if str(d.get("name", "synthetic_multiclass")).lower() in {"cifar10", "cifar100"}:
        prepared = prepare_noisy_classification(config, run_dir, seed)
        return prepared.train_loader, prepared.snapshot_loader, prepared.validation_loader, prepared.test_loader, 0, prepared.num_classes, prepared.noisy_targets
    train = generate_synthetic_multiclass(n, dimension, classes, seed, start_index=0, split="train")
    val = generate_synthetic_multiclass(val_n, dimension, classes, seed + 1, start_index=n, split="validation")
    test = generate_synthetic_multiclass(test_n, dimension, classes, seed + 2, start_index=n + val_n, split="test")
    noise = config.get("noise", {}); manifest = generate_symmetric(train.labels, classes, float(noise.get("rate", 0.2)), int(noise.get("seed", seed + 10)), "synthetic_multiclass", sampling="per_class", rng="default_rng")
    snapshot = DataLoader(MulticlassTensorDataset(train, manifest.noisy_targets), batch_size=int(config.get("loader", {}).get("batch_size", 30)), shuffle=False)
    return DataLoader(MulticlassTensorDataset(train, manifest.noisy_targets), batch_size=int(config.get("loader", {}).get("batch_size", 30)), shuffle=True), snapshot, DataLoader(MulticlassTensorDataset(val), batch_size=30), DataLoader(MulticlassTensorDataset(test), batch_size=30), dimension, classes, manifest.noisy_targets


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input"].to(device)); correct += int(logits.argmax(1).eq(batch["target"].to(device)).sum()); total += logits.shape[0]
    return correct / max(total, 1)


def _snapshot(model: nn.Module, loader: DataLoader, device: torch.device, classes: int) -> PosteriorSnapshot:
    rows: dict[int, tuple[np.ndarray, int]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(model(batch["input"].to(device)), dim=1).cpu().numpy()
            for index, probability, target in zip(batch["index"].tolist(), probs, batch["target"].tolist()):
                rows[int(index)] = (probability, int(target))
    indices = np.asarray(sorted(rows), dtype=np.int64)
    probabilities = np.stack([rows[int(i)][0] for i in indices])
    targets = np.asarray([rows[int(i)][1] for i in indices], dtype=np.int64)
    return PosteriorSnapshot(probabilities, targets, indices, "synthetic_multiclass", "train")


def run_upm_experiment(config: dict[str, Any], output_dir: str | Path | None = None, resume: str | Path | None = None, *, context: RunContext | None = None) -> Path:
    run_dir = _dir(config, output_dir, resume); seed_everything(int(config.get("seed", 1))); train_loader, snapshot_loader, val_loader, test_loader, dimension, classes, noisy_targets = _data(config, run_dir)
    device = torch.device(str(config.get("trainer", {}).get("device", "cpu"))); model_cfg = config.get("model", {}); warm = (_UPMMLP(dimension, int(model_cfg.get("hidden_width", 16)), classes) if dimension else build_reproduction_model(model_cfg, config["data"], classes)).to(device); warm_opt = torch.optim.SGD(warm.parameters(), lr=float(config.get("optimizer", {}).get("lr", 0.05)), momentum=0.9)
    pre_epochs = int(config.get("warmup", {}).get("epochs", 1))
    for _ in range(pre_epochs):
        warm.train()
        for batch in train_loader:
            loss = nn.functional.cross_entropy(warm(batch["input"].to(device)), batch["target"].to(device)); warm_opt.zero_grad(); loss.backward(); warm_opt.step()
    snapshot = _snapshot(warm, snapshot_loader, device, classes)
    state = UPMNoiseState.from_snapshot(snapshot, torch.as_tensor(noisy_targets), eta_init=float(config.get("upm", {}).get("eta_init", 0.01)))
    model = ((_UPMMLP(dimension, int(model_cfg.get("hidden_width", 16)), classes) if dimension else build_reproduction_model(model_cfg, config["data"], classes)).to(device)); optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get("optimizer", {}).get("lr", 0.05)), momentum=0.9)
    scheduler_cfg = config.get("scheduler", {}) or {}; scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(value) for value in scheduler_cfg.get("milestones", [])], gamma=float(scheduler_cfg.get("gamma", 0.1))) if scheduler_cfg.get("milestones") else None
    epochs = int(config.get("trainer", {}).get("epochs", 1)); start = 0; checkpoint = run_dir / "last.pt"
    if resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"]); state.load_state_dict(payload["upm_state"]); start = int(payload["epoch"])
    upm_cfg = config.get("upm", {}); interval = max(1, int(upm_cfg.get("eta_update_interval", 1))); metrics = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    session = context.session if context is not None and context.state.get("lifecycle_active") else None
    if session is not None:
        session.start_phase("alternating_training", total_units=epochs)
    with metrics:
        for epoch in range(start, epochs):
            model.train(); total = 0.0
            for batch in train_loader:
                inputs, targets, indices = batch["input"].to(device), batch["target"].to(device), batch["index"]
                logits = model(inputs); psi, eta = state.lookup(indices); q = estimate_clean_posterior(logits, targets, psi.to(device), eta.to(device)); loss = upm_soft_target_objective(logits, q); optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss.detach()) * inputs.shape[0]
                if (epoch + 1) >= int(upm_cfg.get("eta_update_start_epoch", 1)) and (epoch + 1) % interval == 0:
                    update_confusion_probabilities_(state, indices, q, targets, learning_rate=float(upm_cfg.get("eta_lr", 0.7)))
            if scheduler is not None:
                scheduler.step()
            record = {"epoch": epoch + 1, "train_loss": total / len(train_loader.dataset), "validation_accuracy": _accuracy(model, val_loader, device), "test_accuracy": _accuracy(model, test_loader, device), "eta_mean": float(state.confusion_probability.mean())}
            if session is not None:
                session.log_epoch(
                    epoch + 1,
                    phase="alternating_training",
                    **{key: value for key, value in record.items() if key != "epoch"},
                )
            else:
                metrics.write(json.dumps(record) + "\n"); metrics.flush()
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "upm_state": state.state_dict(), "config": config}, checkpoint)
    if session is not None:
        session.end_phase("alternating_training", completed_units=max(0, epochs - start))
        session.emit("final", phase="evaluation", method="upm", completed_epochs=epochs,
                     test_accuracy=record.get("test_accuracy") if epochs else None)
    return run_dir


__all__ = ["run_upm_experiment"]
