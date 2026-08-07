from __future__ import annotations

"""Independent training entry point for the official DLD workflow."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import math

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from lnl_toolbox.algorithms.dld import (
    DLDPrecorrectionArtifact,
    precorrect_two_views,
    weighted_mse,
)
from lnl_toolbox.models.directional_diffusion import DirectionalDiffusion
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.data.multi_view import build_strong_cifar_transform
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.interfaces import RunContext
from lnl_toolbox.training.experiment import build_optimizer
from lnl_toolbox.training.model_ema import ModelEMA
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.reproduction_data import (
    build_reproduction_model,
    prepare_noisy_classification,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_hash(indices: np.ndarray, weak: np.ndarray, strong: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (indices, weak, strong):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _set_official_learning_rate(
    optimizer: torch.optim.Optimizer,
    progress: float,
    *,
    warmup_epochs: int,
    total_epochs: int,
    base_learning_rate: float,
) -> float:
    """Apply the official DLD linear-warmup/half-cycle cosine schedule."""

    if progress < warmup_epochs:
        learning_rate = base_learning_rate * progress / max(1, warmup_epochs)
    else:
        denominator = max(1, total_epochs - warmup_epochs)
        learning_rate = base_learning_rate * 0.5 * (
            1.0 + math.cos(math.pi * (progress - warmup_epochs) / denominator)
        )
    for group in optimizer.param_groups:
        scale = float(group.get("lr_scale", 1.0))
        group["lr"] = learning_rate * scale
    return learning_rate


def _load_official_noise(data: Any, path: Path, run_dir: Path) -> dict[str, Any]:
    """Replace the observed training targets with an official DLD label file."""

    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("DLD official noise file must contain a one-dimensional label vector")
    indices = np.asarray(data.train_indices, dtype=np.int64)
    if values.size > int(indices.max(initial=0)):
        selected = values[indices]
    elif values.size == indices.size:
        selected = values
    else:
        raise ValueError("DLD official noise labels do not cover the selected training indices")
    if selected.min(initial=0) < 0 or selected.max(initial=0) >= int(data.num_classes):
        raise ValueError("DLD official noise labels are outside the class range")
    selected = selected.astype(np.int64, copy=True)
    data.noisy_targets = selected
    data.manifest.noisy_targets = selected.copy()
    data.manifest.metadata["source"] = str(path)
    data.manifest.metadata["source_sha256"] = _sha256_file(path)
    data.manifest.save(run_dir / "noise_manifest.npz")
    dataset = data.train_loader.dataset
    target_map = {int(index): int(target) for index, target in zip(indices, selected)}
    if hasattr(dataset, "targets_by_index"):
        dataset.targets_by_index = target_map
    elif hasattr(dataset, "_target_by_index"):
        dataset._target_by_index = target_map
    elif hasattr(dataset, "targets") and hasattr(dataset, "indices"):
        for position, index in enumerate(np.asarray(dataset.indices, dtype=np.int64)):
            dataset.targets[position] = int(target_map[int(index)])
    else:
        raise TypeError("DLD cannot replace targets on this dataset adapter")
    metadata = {
        "source": str(path),
        "sha256": _sha256_file(path),
        "num_labels": int(values.size),
        "realized_rate": float(np.mean(selected != np.asarray(data.manifest.clean_targets, dtype=np.int64))),
    }
    (run_dir / "dld_noise_source.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _align_strong_view_normalization(data: Any, config: Mapping[str, Any]) -> None:
    """Apply the configured official CIFAR normalization to the strong view."""

    dataset = data.train_loader.dataset
    if not hasattr(dataset, "strong_transform"):
        return
    normalization = dict(config.get("data", {}).get("normalization", {}) or {})
    mean = normalization.get("mean")
    std = normalization.get("std")
    if mean is None or std is None:
        return
    dataset.strong_transform = build_strong_cifar_transform(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
        magnitude=int(config.get("data", {}).get("strong_magnitude", 10)),
    )


def _collect_train_features(
    feature_model: torch.nn.Module,
    loader,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    feature_model.eval()
    positions = {int(index): position for position, index in enumerate(indices)}
    weak_features: np.ndarray | None = None
    strong_features: np.ndarray | None = None
    with torch.inference_mode():
        for batch in loader:
            weak_input = batch["input"].to(device)
            strong_input = batch.get("strong_input", batch["input"]).to(device)
            weak_output = forward_with_features(feature_model, weak_input)
            strong_output = forward_with_features(feature_model, strong_input)
            weak = weak_output.features.detach().cpu().numpy().astype(np.float32)
            strong = strong_output.features.detach().cpu().numpy().astype(np.float32)
            if weak_features is None:
                weak_features = np.zeros((indices.size, weak.shape[1]), dtype=np.float32)
                strong_features = np.zeros_like(weak_features)
            for row, index in enumerate(batch["index"].detach().cpu().numpy().tolist()):
                try:
                    position = positions[int(index)]
                except KeyError as exc:
                    raise ValueError(f"DLD loader emitted unknown global index {index}") from exc
                weak_features[position] = weak[row]
                strong_features[position] = strong[row]
    if weak_features is None or strong_features is None:
        raise ValueError("DLD training loader produced no batches")
    return weak_features, strong_features


def _load_or_collect_features(
    feature_model: torch.nn.Module,
    loader,
    indices: np.ndarray,
    run_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, str]:
    path = run_dir / "dld_features.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            saved_indices = np.asarray(payload["global_indices"], dtype=np.int64)
            weak = np.asarray(payload["weak_features"], dtype=np.float32)
            strong = np.asarray(payload["strong_features"], dtype=np.float32)
            if not np.array_equal(saved_indices, indices):
                raise ValueError("DLD feature cache indices do not match the current split")
            digest = _feature_hash(saved_indices, weak, strong)
            if str(payload["feature_hash"].item()) != digest:
                raise ValueError("DLD feature cache hash is invalid")
            return weak, strong, digest
    weak, strong = _collect_train_features(feature_model, loader, indices, device)
    digest = _feature_hash(indices, weak, strong)
    np.savez_compressed(
        path,
        global_indices=indices,
        weak_features=weak,
        strong_features=strong,
        feature_hash=np.asarray(digest),
    )
    return weak, strong, digest


@torch.inference_mode()
def _evaluate(
    diffusion: DirectionalDiffusion,
    residual_ema: ModelEMA,
    noise_ema: ModelEMA,
    feature_model: torch.nn.Module,
    loader,
    device: torch.device,
    sampling_timesteps: int,
) -> dict[str, float]:
    feature_model.eval()
    diffusion.eval()
    residual_ema.model.eval()
    noise_ema.model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        output = forward_with_features(feature_model, batch["input"].to(device))
        probabilities = diffusion.sample(
            output.features,
            images=batch["input"].to(device),
            sampling_timesteps=sampling_timesteps,
            residual_model=residual_ema.model,
            noise_model=noise_ema.model,
        )
        targets = batch["target"].to(device, dtype=torch.long)
        loss_sum += float(F.nll_loss(probabilities.clamp_min(1e-8).log(), targets, reduction="sum"))
        correct += int(probabilities.argmax(dim=1).eq(targets).sum())
        total += int(targets.numel())
    if total == 0:
        raise ValueError("DLD evaluation loader produced no samples")
    return {"loss": loss_sum / total, "accuracy": correct / total}


@torch.inference_mode()
def _evaluate_matrix(
    diffusion: DirectionalDiffusion,
    residual_ema: ModelEMA,
    noise_ema: ModelEMA,
    features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    sampling_timesteps: int,
) -> float:
    values = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(targets, dtype=torch.long, device=device)
    correct = 0
    for start in range(0, values.shape[0], 512):
        probabilities = diffusion.sample(
            values[start:start + 512],
            sampling_timesteps=sampling_timesteps,
            residual_model=residual_ema.model,
            noise_model=noise_ema.model,
        )
        correct += int(probabilities.argmax(dim=1).eq(labels[start:start + 512]).sum())
    return correct / max(1, labels.numel())


def _branch_module(*modules: torch.nn.Module) -> torch.nn.Module:
    """Expose one diffusion branch to the shared optimizer factory."""

    return torch.nn.ModuleList(list(modules))


def _symmetric_timesteps(batch: int, total: int, device: torch.device) -> torch.Tensor:
    half = (batch + 1) // 2
    first = torch.randint(0, total, (half,), device=device)
    paired = total - 1 - first
    return torch.cat((first, paired), dim=0)[:batch]


def _load_artifact_or_build(
    weak: np.ndarray,
    strong: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    run_dir: Path,
    config: Mapping[str, Any],
    seed: int,
) -> DLDPrecorrectionArtifact:
    path = run_dir / "dld_pre_correction.npz"
    if path.exists():
        return DLDPrecorrectionArtifact.load(path)
    settings = dict(config.get("dld", {}) or {})
    artifact = precorrect_two_views(
        weak,
        strong,
        labels,
        indices,
        k=int(settings.get("k", 50)),
        use_cosine=bool(settings.get("use_cosine", True)),
        seed=seed,
    )
    artifact.save(path)
    return artifact


def run_dld_experiment(
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *,
    context: RunContext | None = None,
) -> Path:
    """Run one DLD configuration with deterministic artifact/resume checks."""

    config = deepcopy(dict(config))
    seed = int(config.get("seed", 123))
    seed_everything(seed)
    trainer = dict(config.get("trainer", {}) or {})
    device = resolve_device(str(trainer.get("device", "auto")))
    if resume is not None:
        resume_path = Path(resume)
        if resume_path.is_dir():
            resume_path = resume_path / "last.pt"
        run_dir = resume_path.resolve().parent
    else:
        root = Path(output_dir or config.get("output_root", "artifacts/runs"))
        run_dir = root.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_noisy_classification(config, run_dir, seed)
    _align_strong_view_normalization(data, config)
    settings = dict(config.get("dld", {}) or {})
    noise_path = settings.get("noise_path")
    noise_metadata = None
    if noise_path:
        source = Path(str(noise_path))
        if not source.is_file():
            raise FileNotFoundError(f"DLD official noise file not found: {source}")
        noise_metadata = _load_official_noise(data, source, run_dir)

    feature_model_config = dict(config.get("feature_model", config.get("model", {})) or {})
    feature_model = build_reproduction_model(feature_model_config, config["data"], data.num_classes).to(device)
    feature_model.eval()
    weak_features, strong_features, feature_hash = _load_or_collect_features(
        feature_model,
        data.train_loader,
        np.asarray(data.train_indices, dtype=np.int64),
        run_dir,
        device,
    )
    configured_feature_dim = settings.get("feature_dim")
    if configured_feature_dim is not None and int(configured_feature_dim) != weak_features.shape[1]:
        raise ValueError(
            f"DLD feature_dim={configured_feature_dim} does not match extracted dimension {weak_features.shape[1]}"
        )
    artifact = _load_artifact_or_build(
        weak_features,
        strong_features,
        np.asarray(data.noisy_targets, dtype=np.int64),
        np.asarray(data.train_indices, dtype=np.int64),
        run_dir,
        config,
        seed,
    )
    if not np.array_equal(artifact.global_indices, np.asarray(data.train_indices, dtype=np.int64)):
        raise ValueError("DLD artifact indices do not match the current training split")
    feature_average = (weak_features + strong_features) * 0.5
    index_to_row = {int(index): row for row, index in enumerate(data.train_indices)}

    model = DirectionalDiffusion(
        data.num_classes,
        int(weak_features.shape[1]),
        num_timesteps=int(settings.get("num_timesteps", 1000)),
        hidden_width=int(settings.get("hidden_width", 512)),
        time_dim=int(settings.get("time_dim", 64)),
        beta_start=float(settings.get("beta_start", 1e-3)),
        beta_end=float(settings.get("beta_end", 2e-2)),
        schedule=str(
            settings.get(
                "schedule",
                "cosine" if bool(settings.get("use_cosine", True)) else "average",
            )
        ),
        image_base_width=int(settings.get("image_base_width", 16)),
        image_initialization=str(settings.get("image_initialization", "torch_default")),
    ).to(device)
    optimizer_config = dict(settings.get("optimizer", config.get("optimizer", {})) or {})
    optimizer_config.setdefault("name", "adam")
    official_learning_rate = float(settings.get("lr_input", 1e-3))
    optimizer_config.setdefault("lr", official_learning_rate)
    optimizer_config.setdefault("weight_decay", 0.0)
    optimizer_config["lr"] = official_learning_rate
    optimizer = build_optimizer(
        _branch_module(model.residual_model, model.residual_image_encoder), optimizer_config
    )
    noise_optimizer = build_optimizer(
        _branch_module(model.noise_model, model.noise_image_encoder), optimizer_config
    )
    residual_ema = ModelEMA(model.residual_model, float(settings.get("ema", 0.999)))
    noise_ema = ModelEMA(model.noise_model, float(settings.get("ema", 0.999)))
    # The shared CLI writes an explicit override to trainer.epochs.  It must
    # take precedence over a paper default kept in the DLD algorithm block;
    # otherwise ``--epochs 5`` would silently continue toward 200 epochs.
    epochs = int(trainer["epochs"] if "epochs" in trainer else settings.get("epochs", 200))
    warmup_epochs = int(settings.get("warmup_epochs", 5))
    sampling_timesteps = int(settings.get("sampling_timesteps", 10))
    start_epoch = 0
    rows: list[dict[str, Any]] = []
    if resume is not None:
        payload = read_checkpoint(resume_path, device)
        if payload.get("method") != "dld" or payload.get("config") != config:
            raise ValueError("DLD resume identity mismatch: resolved configuration changed")
        if payload.get("artifact_hash") != artifact.artifact_hash or payload.get("feature_hash") != feature_hash:
            raise ValueError("DLD resume identity mismatch: feature or artifact hash changed")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        noise_optimizer.load_state_dict(payload["noise_optimizer"])
        residual_ema.load_state_dict(payload["residual_ema"])
        noise_ema.load_state_dict(payload["noise_ema"])
        feature_model.load_state_dict(payload["feature_model"])
        start_epoch = int(payload["completed_epoch"]) + 1
        rows = list(payload.get("metrics", []))
        restore_rng_state(payload["rng_state"])

    session = context.session if context is not None and context.state.get("lifecycle_active") else None
    if session is not None:
        session.start_phase("diffusion_training", total_units=epochs)
    for epoch in range(start_epoch, epochs):
        model.train()
        feature_model.eval()
        total = 0
        residual_sum = 0.0
        noise_sum = 0.0
        for batch_number, batch in enumerate(data.train_loader):
            progress = epoch + batch_number / max(1, len(data.train_loader))
            _set_official_learning_rate(
                optimizer,
                progress,
                warmup_epochs=warmup_epochs,
                total_epochs=epochs,
                base_learning_rate=official_learning_rate,
            )
            _set_official_learning_rate(
                noise_optimizer,
                progress,
                warmup_epochs=warmup_epochs,
                total_epochs=epochs,
                base_learning_rate=official_learning_rate,
            )
            batch_indices = [index_to_row[int(index)] for index in batch["index"].detach().cpu().numpy().tolist()]
            positions = np.asarray(batch_indices, dtype=np.int64)
            features = torch.as_tensor(feature_average[positions], dtype=torch.float32, device=device)
            targets = torch.as_tensor(
                (artifact.weak_targets[positions] + artifact.strong_targets[positions]) * 0.5,
                dtype=torch.float32,
                device=device,
            )
            weights = torch.as_tensor(artifact.loss_weights[positions], dtype=torch.float32, device=device)
            weak_images = batch["input"].to(device)
            strong_images = batch.get("strong_input", batch["input"]).to(device)
            images = 0.5 * (weak_images + strong_images)
            y_input = torch.zeros_like(targets)
            timesteps = _symmetric_timesteps(targets.shape[0], model.num_timesteps, device)
            predicted_residual, predicted_noise, residual, sampled_noise, _ = model.forward_t(
                y_input, targets, features, timesteps, images=images
            )
            residual_loss = weighted_mse(predicted_residual, residual, weights).mean()
            noise_loss = weighted_mse(predicted_noise, sampled_noise, weights).mean()
            optimizer.zero_grad(set_to_none=True)
            residual_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.residual_model.parameters(), 1.0)
            optimizer.step()
            noise_optimizer.zero_grad(set_to_none=True)
            noise_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.noise_model.parameters(), 1.0)
            noise_optimizer.step()
            residual_ema.update(model.residual_model)
            noise_ema.update(model.noise_model)
            count = int(targets.shape[0])
            total += count
            residual_sum += float(residual_loss.detach()) * count
            noise_sum += float(noise_loss.detach()) * count

        train_accuracy = _evaluate_matrix(
            model, residual_ema, noise_ema, feature_average,
            artifact.weak_targets.argmax(axis=1), device, sampling_timesteps,
        )
        if epoch >= warmup_epochs:
            evaluation = _evaluate(
                model, residual_ema, noise_ema, feature_model,
                data.validation_loader, device, sampling_timesteps,
            )
            test = _evaluate(
                model, residual_ema, noise_ema, feature_model,
                data.test_loader, device, sampling_timesteps,
            )
        else:
            evaluation = {"loss": 0.0, "accuracy": 0.0}
            test = {"loss": 0.0, "accuracy": 0.0}
        row = standardize_epoch_row({
            "method": "dld",
            "epoch": epoch + 1,
            "phase": "warmup" if epoch < warmup_epochs else "diffusion",
            "train_loss": (residual_sum + noise_sum) / max(1, total),
            "train_accuracy": train_accuracy,
            "validation_loss": evaluation["loss"],
            "validation_accuracy": evaluation["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        rows.append(row)
        if session is not None:
            session.log_epoch(
                epoch + 1,
                phase=str(row.get("phase", "diffusion")),
                **{key: value for key, value in row.items()
                   if key not in {"event", "epoch", "phase", "seq"}},
            )
        print(
            f"DLD epoch {epoch + 1}/{epochs} phase={row['phase']} "
            f"loss={row['train_loss']:.5f} val={row['validation_accuracy']:.4f} "
            f"test={row['test_accuracy']:.4f}",
            flush=True,
        )
        atomic_save({
            "method": "dld",
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "noise_optimizer": noise_optimizer.state_dict(),
            "residual_ema": residual_ema.state_dict(),
            "noise_ema": noise_ema.state_dict(),
            "feature_model": feature_model.state_dict(),
            "completed_epoch": epoch,
            "metrics": rows,
            "artifact_hash": artifact.artifact_hash,
            "feature_hash": feature_hash,
            "noise_metadata": noise_metadata,
            "rng_state": capture_rng_state(),
        }, run_dir / "last.pt")

    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if session is None:
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    else:
        session.end_phase("diffusion_training", completed_units=max(0, epochs - start_epoch))
        session.emit("final", phase="evaluation", method="dld",
                     completed_epochs=epochs,
                     test_accuracy=rows[-1].get("test_accuracy") if rows else None)
    if rows:
        write_training_curves_svg(rows, run_dir / "training_curves.svg")
    return run_dir


__all__ = ["run_dld_experiment"]
