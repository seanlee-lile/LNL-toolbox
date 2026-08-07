from __future__ import annotations

"""LEND feature-dilution lifecycle (paper-equation implementation)."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from lnl_toolbox.data.multiclass_synthetic import MulticlassTensorDataset, generate_synthetic_multiclass
from lnl_toolbox.noise.generators import generate_symmetric
from lnl_toolbox.runtime import seed_everything
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.training.reproduction_data import build_reproduction_model, prepare_noisy_classification
from lnl_toolbox.selectors.history import IndexedSoftLabelState
from lnl_toolbox.selectors.lend import LENDSelector


class _LENDMLP(nn.Module):
    def __init__(self, dimension: int, width: int, classes: int) -> None:
        super().__init__(); self.encoder = nn.Sequential(nn.Linear(dimension, width), nn.ReLU()); self.classifier = nn.Linear(width, classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x); return features, self.classifier(features)


def _run_dir(config: Mapping[str, Any], output_dir: str | Path | None, resume: str | Path | None) -> Path:
    path = Path(resume).resolve().parent if resume else (Path(output_dir).expanduser().resolve() if output_dir else Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")); path.mkdir(parents=True, exist_ok=True); return path


def run_lend_experiment(config: dict[str, Any], output_dir: str | Path | None = None, resume: str | Path | None = None) -> Path:
    run_dir = _run_dir(config, output_dir, resume); seed_everything(int(config.get("seed", 1))); data = config.get("data", {}); classes, dimension = int(data.get("num_classes", 3)), int(data.get("dimension", 6)); n = int(data.get("train_size", 90)); seed = int(config.get("seed", 1))
    if str(data.get("name", "synthetic_multiclass")).lower() in {"cifar10", "cifar100"}:
        prepared = prepare_noisy_classification(config, run_dir, seed); loader, val_loader, test_loader, classes, n = prepared.train_loader, prepared.validation_loader, prepared.test_loader, prepared.num_classes, int(prepared.train_indices.size); dimension = 0
    else:
        train = generate_synthetic_multiclass(n, dimension, classes, seed, start_index=0, split="train"); val = generate_synthetic_multiclass(int(data.get("validation_size", 30)), dimension, classes, seed + 1, start_index=n, split="validation"); test = generate_synthetic_multiclass(int(data.get("test_size", 30)), dimension, classes, seed + 2, start_index=n + int(data.get("validation_size", 30)), split="test"); noise = config.get("noise", {}); manifest = generate_symmetric(train.labels, classes, float(noise.get("rate", 0.2)), int(noise.get("seed", seed + 10)), "synthetic_multiclass", sampling="per_class", rng="default_rng"); loader = DataLoader(MulticlassTensorDataset(train, manifest.noisy_targets), batch_size=int(config.get("loader", {}).get("batch_size", 30)), shuffle=True); val_loader = DataLoader(MulticlassTensorDataset(val), batch_size=30); test_loader = DataLoader(MulticlassTensorDataset(test), batch_size=30)
    device = torch.device(str(config.get("trainer", {}).get("device", "cpu"))); model = (_LENDMLP(dimension, int(config.get("model", {}).get("hidden_width", 16)), classes) if dimension else build_reproduction_model(config["model"], data, classes)).to(device); optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get("optimizer", {}).get("lr", 0.05)), momentum=0.9); selector_cfg = config.get("selector", {}); selector = LENDSelector(neighbors=int(selector_cfg.get("neighbors", 10)), gamma=float(selector_cfg.get("gamma", 1.0)), diffusion_alpha=float(selector_cfg.get("diffusion_alpha", 0.99)), diffusion_steps=int(selector_cfg.get("diffusion_steps", 10)), num_classes=classes); state = IndexedSoftLabelState(n, classes); checkpoint = run_dir / "last.pt"; start = 0
    if resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"]); state.load_state_dict(payload["lend_state"]); start = int(payload["epoch"])
    epochs = int(config.get("trainer", {}).get("epochs", 1)); metrics = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    with metrics:
        for epoch in range(start, epochs):
            model.train(); total = 0.0; selected_total = 0; count = 0
            for batch in loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                indices = batch["index"]
                if dimension:
                    features, logits = model(inputs)
                else:
                    output = forward_with_features(model, inputs)
                    features, logits = output.features, output.logits
                selection = selector.select(features=features.detach(), noisy_targets=targets)
                selected = selection.selected_mask
                if selector.last_soft_labels is None:
                    raise RuntimeError("LEND selector did not produce diluted labels")
                state.update(indices, selector.last_soft_labels, momentum=float(config.get("selector", {}).get("momentum", 0.9)))
                if selected.any():
                    loss = nn.functional.cross_entropy(logits[selected], targets[selected])
                else:
                    loss = logits.sum() * 0.0
                optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss.detach()) * inputs.shape[0]; selected_total += int(selected.sum()); count += int(inputs.shape[0])
            def acc(eval_loader: DataLoader) -> float:
                model.eval(); correct = total_eval = 0
                with torch.no_grad():
                    for item in eval_loader:
                        out = model(item["input"].to(device))[1] if dimension else forward_with_features(model, item["input"].to(device)).logits; correct += int(out.argmax(1).eq(item["target"].to(device)).sum()); total_eval += out.shape[0]
                return correct / max(total_eval, 1)
            record = {"epoch": epoch + 1, "train_loss": total / max(count, 1), "selected_ratio": selected_total / max(count, 1), "validation_accuracy": acc(val_loader), "test_accuracy": acc(test_loader)}; metrics.write(json.dumps(record) + "\n"); metrics.flush(); torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "lend_state": state.state_dict(), "config": config}, checkpoint)
    return run_dir


__all__ = ["run_lend_experiment"]
