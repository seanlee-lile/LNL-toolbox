from __future__ import annotations

"""Offline, dataset-neutral MentorNet artifact learning."""

from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np

import torch
from torch.utils.data import DataLoader

from lnl_toolbox.data.curriculum import MentorFeatureDataset
from lnl_toolbox.data import (
    NoisyTargetDataset,
    TorchCifarDataset,
    build_cifar_transform,
    load_cifar10,
)
from lnl_toolbox.models import TinyCNN
from lnl_toolbox.noise.generators import generate_symmetric
from lnl_toolbox.models.mentornet import build_mentor_model
from lnl_toolbox.training.mentor_artifacts import MentorArtifact


def prepare_trusted_mentor_features(
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    """Create isolated Student-feedback records from a trusted CIFAR-10 subset."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    data = load_cifar10(config.get("data_root"), "train")
    size = int(config.get("trusted_size", 5000))
    if not 0 < size <= len(data.labels):
        raise ValueError("trusted_size is outside CIFAR-10")
    random = np.random.default_rng(seed)
    parts = []
    per_class = size // 10
    for class_index in range(10):
        candidates = np.flatnonzero(data.labels == class_index)
        parts.append(random.choice(candidates, per_class, replace=False))
    indices = np.sort(np.concatenate(parts)).astype(np.int64)
    clean = data.labels[indices]
    noise_rate = float(config.get("noise_rate", 0.4))
    manifest = generate_symmetric(
        clean, 10, noise_rate, seed, dataset="mentornet_trusted_cifar10"
    )
    np.savez_compressed(destination / "trusted_indices.npz", indices=indices)
    manifest.save(destination / "noise_manifest.npz")

    base = TorchCifarDataset(
        data,
        indices,
        transform=build_cifar_transform(True, bool(config.get("augment", True))),
    )
    noisy_by_global = dict(zip(indices.tolist(), manifest.noisy_targets.tolist()))
    noisy = NoisyTargetDataset(
        base, indices, np.asarray([noisy_by_global[int(i)] for i in indices])
    )
    loader = DataLoader(
        noisy,
        batch_size=int(config.get("batch_size", 128)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    device = torch.device(str(config.get("device", "cpu")))
    model = TinyCNN(10, int(config.get("student_width", 32))).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config.get("student_learning_rate", 0.1)),
        momentum=0.9,
    )
    clean_by_global = dict(zip(indices.tolist(), clean.tolist()))
    losses_out: list[np.ndarray] = []
    differences_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    epochs_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    moving: float | None = None
    epochs = int(config.get("student_epochs", 20))
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = torch.as_tensor(batch["target"], device=device)
            sample_indices = torch.as_tensor(batch["index"]).cpu().numpy()
            per_sample = torch.nn.functional.cross_entropy(
                model(inputs), targets, reduction="none"
            )
            percentile = float(torch.quantile(per_sample.detach(), 0.75).item())
            moving = percentile if moving is None else 0.95 * moving + 0.05 * percentile
            optimizer.zero_grad(set_to_none=True)
            per_sample.mean().backward()
            optimizer.step()
            values = per_sample.detach().cpu().numpy().astype(np.float32)
            losses_out.append(values)
            differences_out.append((values - moving).astype(np.float32))
            labels_out.append(np.zeros(values.size, dtype=np.int64))
            epochs_out.append(
                np.full(values.size, min(99, int(100 * epoch / epochs)), dtype=np.int64)
            )
            noisy_targets = np.asarray(batch["target"], dtype=np.int64)
            targets_out.append(np.asarray([
                float(noisy_targets[position] == clean_by_global[int(index)])
                for position, index in enumerate(sample_indices)
            ], dtype=np.float32))
    feature_path = destination / "mentor_features.npz"
    np.savez_compressed(
        feature_path,
        losses=np.concatenate(losses_out),
        loss_differences=np.concatenate(differences_out),
        labels=np.concatenate(labels_out),
        epoch_percentages=np.concatenate(epochs_out),
        curriculum_targets=np.concatenate(targets_out),
    )
    metadata = {
        "seed": seed,
        "trusted_size": size,
        "noise_rate": noise_rate,
        "student_epochs": epochs,
        "feature_records": int(sum(len(value) for value in losses_out)),
        "clean_truth_scope": "offline_curriculum_target_only",
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return feature_path


def train_mentor_artifact(
    config: Mapping[str, Any],
    output_path: str | Path,
) -> MentorArtifact:
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    dataset = MentorFeatureDataset.from_npz(config["feature_data"])
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 128)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    architecture = dict(config.get("model", {}))
    device = torch.device(str(config.get("device", "cpu")))
    model = build_mentor_model(architecture).to(device)
    implementation = str(architecture.get("implementation", "legacy")).lower()
    learning_rate = float(config.get("learning_rate", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    max_steps = config.get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
    epochs = int(config.get("epochs", 20))
    if epochs <= 0:
        raise ValueError("MentorNet epochs must be positive")
    model.train()
    if implementation == "official" and max_steps is not None:
        iterator = iter(loader)
        for step in range(max_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            current_lr = learning_rate * (0.9 ** (step // 1000))
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            prediction = model(
                batch["loss"].to(device),
                batch["loss_difference"].to(device),
                batch["label"].to(device),
                batch["epoch_percentage"].to(device),
            )
            objective = torch.nn.functional.mse_loss(
                prediction, batch["curriculum_target"].to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    else:
        for _ in range(epochs):
            for batch in loader:
                prediction = model(
                    batch["loss"].to(device),
                    batch["loss_difference"].to(device),
                    batch["label"].to(device),
                    batch["epoch_percentage"].to(device),
                )
                objective = torch.nn.functional.mse_loss(
                    prediction, batch["curriculum_target"].to(device)
                )
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
    artifact = MentorArtifact.create(
        architecture=model.architecture(),
        feature_schema={
            "loss": "float",
            "loss_difference": "float",
            "label": "int",
            "epoch_percentage": "int[0,99]",
            "curriculum_target": "float[0,1]",
        },
        source={
            "feature_data": str(Path(config["feature_data"])),
            "seed": seed,
            "epochs": epochs,
            "max_steps": max_steps,
            "optimizer": "adam",
            "learning_rate_decay": "0.9 every 1000 steps" if max_steps else None,
        },
        model_state={
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
    )
    artifact.save(output_path)
    return artifact
