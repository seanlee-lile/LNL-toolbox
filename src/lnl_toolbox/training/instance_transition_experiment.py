from __future__ import annotations

"""Reusable staged CIFAR runner for instance-transition estimators."""

from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.algorithms.instance_transition import InstanceTransitionClassificationAlgorithm
from lnl_toolbox.algorithms.instance_transition import pdl_instance_corrected_losses
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.core.hyperparameters import resolve_parameter_sampling
from lnl_toolbox.data import NoisyTargetDataset
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.torch_cifar import (
    TorchCifarDataset,
    build_cifar_transform,
    stratified_split,
    train_validation_split,
)
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.models.cifar_resnet import cifar_resnet34
from lnl_toolbox.noise import (
    NoiseManifest,
    PartTransitionArtifact,
    PosteriorSnapshot,
    fit_part_representation,
    generate_pdl_idn,
)
from lnl_toolbox.noise.pdl import (
    fit_pdl_basis_matrices_pair,
    select_pdl_anchor_candidates,
)
from lnl_toolbox.plugins.builtin import (
    build_builtin_instance_transition_algorithm,
    build_builtin_instance_transition_estimator,
    build_builtin_loss,
)
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from lnl_toolbox.training.experiment import build_model, build_optimizer, build_scheduler
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.snapshots import (
    FeatureSnapshot,
    collect_feature_snapshot,
    collect_posterior_snapshot,
    pretrain_noisy_classifier,
)


def _loader(dataset, config: Mapping[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)

    def seed_worker(_worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=shuffle,
        num_workers=workers, pin_memory=bool(config.get("pin_memory", True)),
        drop_last=bool(config.get("drop_last", False)) if shuffle else False,
        persistent_workers=workers > 0, worker_init_fn=seed_worker if workers else None,
        generator=generator)


def _official_pdl_split(
    size: int, validation_size: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce ``data.py::data_split`` including global NumPy RNG state."""

    if not 0 < int(validation_size) < int(size):
        raise ValueError("PDL validation_size must lie strictly inside the dataset")
    np.random.seed(int(seed))
    all_positions = np.arange(int(size), dtype=np.int64)
    train_positions = np.random.choice(
        int(size), int(size) - int(validation_size), replace=False
    ).astype(np.int64, copy=False)
    validation_positions = np.delete(all_positions, train_positions)
    return train_positions, validation_positions


def _subset_pdl_snapshots(
    features: FeatureSnapshot,
    posteriors: PosteriorSnapshot,
    global_indices: np.ndarray,
    *,
    split: str,
) -> tuple[FeatureSnapshot, PosteriorSnapshot]:
    """Slice a shared train/validation snapshot by stable global index."""

    if not np.array_equal(features.global_indices, posteriors.global_indices):
        raise ValueError("PDL feature and posterior union snapshots are misaligned")
    requested = np.asarray(global_indices, dtype=np.int64)
    if requested.ndim != 1 or requested.size == 0 or np.unique(requested).size != requested.size:
        raise ValueError("PDL snapshot subset indices must be unique and non-empty")
    order = np.argsort(requested, kind="stable")
    sorted_requested = requested[order]
    positions = np.searchsorted(features.global_indices, sorted_requested)
    valid = positions < features.global_indices.size
    if not np.all(valid) or not np.array_equal(
        features.global_indices[positions[valid]], sorted_requested[valid]
    ):
        raise KeyError("PDL union snapshots do not cover every split index")
    selected_features = FeatureSnapshot(
        features.features[positions],
        features.noisy_targets[positions],
        sorted_requested,
        features.dataset,
        split,
    )
    selected_posteriors = PosteriorSnapshot(
        posteriors.noisy_probabilities[positions],
        posteriors.noisy_targets[positions],
        sorted_requested,
        posteriors.dataset,
        split,
    )
    return selected_features, selected_posteriors


def _run_directory(config: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    path = Path(output_dir) if output_dir is not None else Path(
        config.get("output_root", "artifacts/runs")
    ) / datetime.now().strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _transform(training: bool, config: Mapping[str, Any]):
    normalization = dict(config.get("normalization", {}) or {})
    return build_cifar_transform(
        training, bool(config.get("augment", False)) if training else False,
        preprocessing=str(config.get("preprocessing", "standard")),
        normalization_mean=normalization.get("mean"),
        normalization_std=normalization.get("std"),
    )


def _write_environment(path: Path, seed: int, device: torch.device) -> None:
    path.write_text(json.dumps({
        "python_executable": __import__("sys").executable,
        "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "device": str(device), "seed": seed,
    }, indent=2), encoding="utf-8")


def _build_instance_model(config: Mapping[str, Any], num_classes: int) -> nn.Module:
    """Build the reusable instance-transition model with optional CIFAR stem settings."""

    if str(config.get("name", "")).strip().lower() == "resnet34" and (
        "stem_padding" in config or "initialization" in config
    ):
        return cifar_resnet34(
            num_classes,
            int(config.get("base_width", 64)),
            initialization=str(config.get("initialization", "kaiming")),
            stem_padding=int(config.get("stem_padding", 1)),
        )
    return build_model(config, num_classes)


def _attach_pdl_revision_head(model: nn.Module, num_classes: int) -> None:
    """Attach the official global, bias-free ``T_revision`` parameter."""

    if not hasattr(model, "T_revision"):
        model.T_revision = nn.Linear(num_classes, num_classes, bias=False)


def _pdl_validation(
    model: nn.Module,
    loader: DataLoader,
    artifact: PartTransitionArtifact,
    device: torch.device,
    *,
    revision: bool,
) -> dict[str, float]:
    """Evaluate the same corrected probabilities used by official PDL val."""

    was_training = model.training
    model.eval()
    total = 0
    loss_sum = 0.0
    correct = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            logits = model(inputs)
            matrices = artifact.transition_for(
                inputs, batch["index"], device=device, dtype=logits.dtype
            )
            if revision:
                head = getattr(model, "T_revision")
                matrices = matrices + head.weight.to(matrices)
            # Official ``val_correction`` and ``val_revision`` normalize the
            # matrix immediately before multiplying probabilities.  The
            # revision path additionally applies abs() to matrix+T_revision.
            matrices = matrices.abs()
            matrices = matrices / matrices.sum(dim=2, keepdim=True).clamp_min(
                torch.finfo(matrices.dtype).tiny
            )
            clean = torch.softmax(logits, dim=1)
            observed = torch.bmm(clean.unsqueeze(1), matrices).squeeze(1)
            clean_y = clean.gather(1, targets[:, None]).squeeze(1)
            observed_y = observed.gather(1, targets[:, None]).squeeze(1)
            losses = (clean_y / observed_y.clamp_min(torch.finfo(logits.dtype).tiny)) * (
                -torch.log(observed_y.clamp_min(torch.finfo(logits.dtype).tiny))
            )
            count = int(targets.numel())
            total += count
            loss_sum += float(losses.sum())
            correct += int(observed.argmax(1).eq(targets).sum())
    model.train(was_training)
    return {"loss": loss_sum / total, "accuracy": correct / total}


def _run_pdl_official_phases(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    model: nn.Module,
    artifact: PartTransitionArtifact,
    validation_artifact: PartTransitionArtifact,
    revision_validation_artifact: PartTransitionArtifact,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    seed: int,
    epochs: int,
    parameter_record: Any,
    resume: str | Path | None,
) -> Path:
    """Run PDL's warm-up/correction/revision phases without a paper branch."""

    phases = dict(config.get("phases", {}))
    correction_epochs = int(phases.get("correction_epochs", 0))
    revision_epochs = int(phases.get("revision_epochs", 0))
    if correction_epochs < 0 or revision_epochs < 0:
        raise ValueError("PDL phase epochs must be non-negative")
    if correction_epochs + revision_epochs != epochs:
        raise ValueError("trainer.epochs must equal PDL correction + revision epochs")
    if not hasattr(model, "T_revision"):
        _attach_pdl_revision_head(model, artifact.num_classes)
    model.to(device)
    criterion = build_builtin_loss(config.get("loss", {"name": "ce"})).to(device)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    if parameter_record is not None:
        (run_dir / "parameter_record.json").write_text(
            json.dumps(parameter_record.to_dict(), indent=2), encoding="utf-8"
        )
    _write_environment(run_dir / "environment.json", seed, device)
    metrics_path = run_dir / "metrics.jsonl"
    rows = []
    if metrics_path.is_file():
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("event") == "epoch"
        ]
    payload = read_checkpoint(resume, device) if resume else None
    phase_start = 0
    completed_in_phase = -1
    if payload is not None:
        if payload.get("method") != "pdl" or payload.get("config") != dict(config):
            raise ValueError("PDL resume configuration mismatch")
        if payload.get("artifact_hash") != artifact.artifact_hash:
            raise ValueError("PDL transition artifact resume mismatch")
        pipeline = dict(payload.get("pipeline", {}))
        if pipeline.get("validation_artifact_hash") != validation_artifact.artifact_hash:
            raise ValueError("PDL validation artifact resume mismatch")
        if pipeline.get("revision_validation_artifact_hash") != revision_validation_artifact.artifact_hash:
            raise ValueError("PDL revision validation artifact resume mismatch")
        saved_phase = str(payload.get("pipeline", {}).get("phase", "correction"))
        phase_start = 1 if saved_phase == "revision" and payload.get("phase_complete") else 0
        if saved_phase == "revision":
            phase_start = 1
        completed_in_phase = int(payload.get("completed_epoch", -1))
        model.load_state_dict(payload["model"])

    phase_specs = [
        ("correction", correction_epochs, dict(config["optimizer"]), "pdl"),
        ("revision", revision_epochs, dict(config["revision"]["optimizer"]), "pdl_revision"),
    ]
    global_epoch = len(rows)
    best_accuracy = max((float(row.get("selection_accuracy", -1.0)) for row in rows), default=-1.0)
    (run_dir / "pipeline_state.json").write_text(json.dumps({
        "method": "pdl",
        "phase": "correction" if phase_start == 0 else "revision",
        "artifact_hash": artifact.artifact_hash,
        "validation_artifact_hash": validation_artifact.artifact_hash,
        "revision_validation_artifact_hash": revision_validation_artifact.artifact_hash,
        "train_indices": artifact.global_indices.tolist(),
        "validation_indices": validation_artifact.global_indices.tolist(),
    }, indent=2), encoding="utf-8")
    for phase_index, (phase, phase_length, optimizer_config, correction) in enumerate(phase_specs):
        if phase_index < phase_start or phase_length == 0:
            continue
        phase_best_path = run_dir / "pdl_correction_best.pt"
        phase_best_accuracy = float("-inf")
        phase_best_state = None
        if phase == "correction" and phase_best_path.is_file():
            saved_best = torch.load(phase_best_path, map_location="cpu", weights_only=False)
            phase_best_accuracy = float(saved_best["accuracy"])
            phase_best_state = saved_best["model"]
        optimizer = build_optimizer(model, optimizer_config)
        algorithm_config = dict(config["algorithm"])
        algorithm_config["correction"] = correction
        algorithm = build_builtin_instance_transition_algorithm(
            algorithm_config, model=model, optimizer=optimizer, loss=criterion,
            transition=artifact, device=device,
        )
        algorithm.setup(ExperimentContext(run_dir, dict(config), seed))
        state = RunState(phase=phase)
        local_start = 0
        if payload is not None and phase_index == phase_start:
            state, completed, loaded = load_checkpoint(resume, algorithm, device)
            local_start = int(completed) + 1
            payload = loaded
        elif phase == "revision":
            with torch.no_grad():
                model.T_revision.weight.zero_()
        algorithm.on_run_start(state)
        with metrics_path.open("a", encoding="utf-8") as handle:
            for local_epoch in range(local_start, phase_length):
                state.cycle = local_epoch
                algorithm.on_cycle_start(state)
                loss_sum = accuracy_sum = samples = 0.0
                for raw_batch in train_loader:
                    result = algorithm.step(Batch(raw_batch), state)
                    count = result.metrics["samples"]
                    samples += count
                    loss_sum += result.metrics["loss"] * count
                    accuracy_sum += result.metrics["accuracy"] * count
                algorithm.on_cycle_end(state)
                selection_artifact = (
                    validation_artifact
                    if phase == "correction"
                    else revision_validation_artifact
                )
                selection = _pdl_validation(
                    model, validation_loader, selection_artifact, device,
                    revision=phase == "revision",
                )
                test = evaluate_classification(model, test_loader, criterion, device)
                global_epoch += 1
                row = standardize_epoch_row({
                    "event": "epoch", "epoch": global_epoch, "phase": phase,
                    "phase_epoch": local_epoch + 1, "global_step": state.step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_loss": loss_sum / samples,
                    "train_accuracy": accuracy_sum / samples,
                    "selection_split": "noisy_validation",
                    "selection_loss": selection["loss"],
                    "selection_accuracy": selection["accuracy"],
                    "test_loss": test["loss"], "test_accuracy": test["accuracy"],
                    "method": "pdl",
                })
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if phase == "correction" and selection["accuracy"] > phase_best_accuracy:
                    phase_best_accuracy = selection["accuracy"]
                    phase_best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    torch.save(
                        {"accuracy": phase_best_accuracy, "model": phase_best_state},
                        phase_best_path,
                    )
                best_accuracy = max(best_accuracy, selection["accuracy"])
                save_checkpoint(
                    run_dir / "last.pt", algorithm, state, local_epoch, dict(config),
                    pipeline={
                        "phase": phase,
                        "artifact_hash": artifact.artifact_hash,
                        "validation_artifact_hash": validation_artifact.artifact_hash,
                        "revision_validation_artifact_hash": revision_validation_artifact.artifact_hash,
                    },
                    best_epoch=global_epoch, best_selection_accuracy=best_accuracy,
                    selection_split="noisy_validation",
                )
                if bool(config.get("trainer", {}).get("progress", {}).get("curves", True)):
                    write_training_curves_svg(rows, run_dir / "training_curves.svg")
        algorithm.on_run_end(state)
        if phase == "correction" and revision_epochs > 0:
            if phase_best_state is None:
                raise RuntimeError("PDL correction phase produced no best checkpoint")
            model.load_state_dict(phase_best_state)
        payload = None
    final_test = evaluate_classification(model, test_loader, criterion, device)
    final = {
        "event": "final", "method": "pdl", "completed_epochs": global_epoch,
        "test_loss": final_test["loss"], "test_accuracy": final_test["accuracy"],
        "artifact_hash": artifact.artifact_hash,
        "validation_artifact_hash": validation_artifact.artifact_hash,
        "revision_validation_artifact_hash": revision_validation_artifact.artifact_hash,
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(final) + "\n")
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return run_dir


def _prepare_pdl_manifest(
    *, config: Mapping[str, Any], data: Any, indices: np.ndarray,
    num_classes: int, run_dir: Path, resume: bool,
) -> NoiseManifest:
    path = run_dir / str(config.get("manifest_filename", "noise_manifest.npz"))
    if resume:
        manifest = NoiseManifest.load(path)
        manifest.validate_for(data.labels, data.dataset, num_classes, required_indices=indices)
        return manifest
    if str(config.get("name", "")).lower() != "pdl":
        raise ValueError("instance-transition runner currently requires noise.name=pdl")
    # The official PDL data loader passes the raw CIFAR array from data.py
    # (CHW order) flattened before Algorithm 2 consumes it.  Toolbox datasets
    # expose images in HWC order, so convert explicitly instead of letting a
    # reshape silently change the feature coordinates used by x @ W[y].
    raw_inputs = _pdl_official_raw_features(data.images[indices])
    manifest = generate_pdl_idn(
        raw_inputs,
        data.labels[indices], num_classes, float(config["rate"]),
        int(config["seed"]), data.dataset,
    )
    manifest.global_indices = indices.copy()
    manifest.split = "train"
    manifest.num_classes = num_classes
    manifest.save(path)
    return manifest


def _pdl_official_raw_features(images: np.ndarray) -> np.ndarray:
    """Return CIFAR inputs in the flattened layout used by official PDL."""

    values = np.asarray(images)
    if values.ndim == 4 and values.shape[1:] == (32, 32, 3):
        values = np.transpose(values, (0, 3, 1, 2))
    elif values.ndim != 2:
        raise ValueError(
            "PDL official raw inputs must be flattened or CIFAR HWC images"
        )
    return values.reshape(values.shape[0], -1).astype(np.float64, copy=False)


def run_instance_transition_experiment(
    raw_config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Run warm-up → snapshots → instance estimator → corrected training."""

    config, parameter_record = resolve_parameter_sampling(raw_config)
    config = dict(config)
    seed = int(config.get("seed", 1))
    epochs = int(config["trainer"]["epochs"])
    if epochs <= 0:
        raise ValueError("trainer.epochs must be positive")
    seed_everything(seed)
    device = resolve_device(str(config["trainer"].get("device", "auto")))
    run_dir = Path(resume).resolve().parent if resume else _run_directory(config, output_dir)
    checkpoint_payload = None if resume is None else read_checkpoint(resume, device)
    if checkpoint_payload is not None and dict(checkpoint_payload.get("config") or {}) != config:
        raise ValueError("Resume configuration changed")

    data_config = dict(config["data"])
    dataset_name = str(data_config["name"]).lower()
    if dataset_name == "cifar10":
        train_data, test_data, num_classes = (
            load_cifar10(data_config["root"], "train"),
            load_cifar10(data_config["root"], "test"), 10,
        )
    elif dataset_name == "cifar100":
        train_data, test_data, num_classes = (
            load_cifar100(data_config["root"], "train"),
            load_cifar100(data_config["root"], "test"), 100,
        )
    else:
        raise ValueError("instance-transition runner supports CIFAR-10/100")
    all_indices = np.arange(len(train_data), dtype=np.int64)
    maximum = data_config.get("max_train_samples")
    if maximum is not None and int(maximum) < all_indices.size:
        _, selected = stratified_split(train_data.labels, int(maximum), seed + 1)
        all_indices = selected
    manifest = _prepare_pdl_manifest(config=config["noise"], data=train_data,
        indices=all_indices, num_classes=num_classes, run_dir=run_dir, resume=resume is not None)

    noisy_validation_size = int(config["warmup"]["noisy_validation_size"])
    split_strategy = str(config["warmup"].get("split_strategy", "random"))
    split_rng = str(config["warmup"].get("split_rng", "numpy_legacy"))
    official_pdl = (
        str(config.get("algorithm", {}).get("correction", "")).lower()
        in {"pdl", "pdl_revision"}
        or "phases" in config
    )
    if official_pdl:
        train_positions, validation_positions = _official_pdl_split(
            manifest.noisy_targets.size, noisy_validation_size, seed
        )
    else:
        train_positions, validation_positions = train_validation_split(
            manifest.noisy_targets,
            noisy_validation_size,
            seed,
            strategy=split_strategy,
            rng=split_rng,
        )
    train_indices = manifest.global_indices[train_positions]
    validation_indices = manifest.global_indices[validation_positions]
    train_targets = manifest.noisy_targets[train_positions]
    validation_targets = manifest.noisy_targets[validation_positions]
    train_set = NoisyTargetDataset(TorchCifarDataset(train_data, train_indices,
        transform=_transform(True, data_config)), train_indices, train_targets)
    noisy_validation_set = NoisyTargetDataset(TorchCifarDataset(train_data, validation_indices,
        transform=_transform(False, data_config)), validation_indices, validation_targets)
    union_set = NoisyTargetDataset(TorchCifarDataset(train_data, manifest.global_indices,
        transform=_transform(False, data_config)), manifest.global_indices, manifest.noisy_targets)
    max_test = data_config.get("max_test_samples")
    test_indices = np.arange(len(test_data), dtype=np.int64)
    if max_test is not None and int(max_test) < test_indices.size:
        _, test_indices = stratified_split(test_data.labels, int(max_test), seed + 3)
    test_set = TorchCifarDataset(test_data, test_indices, transform=_transform(False, data_config))
    loader_config = dict(config["loader"])
    # The official PDL code uses a non-shuffled loader and indexes W by the
    # sequential batch position during correction.
    train_loader = _loader(train_set, loader_config, shuffle=False, seed=seed)
    noisy_validation_loader = _loader(noisy_validation_set, loader_config, shuffle=False, seed=seed)
    union_loader = _loader(union_set, loader_config, shuffle=False, seed=seed)
    test_loader = _loader(test_set, loader_config, shuffle=False, seed=seed)

    artifact_path = run_dir / "instance_transition_artifact.npz"
    validation_artifact_path = run_dir / "validation_instance_transition_artifact.npz"
    revision_validation_artifact_path = run_dir / "revision_validation_instance_transition_artifact.npz"
    if resume is not None:
        artifact = PartTransitionArtifact.load(artifact_path)
        if official_pdl:
            validation_artifact = PartTransitionArtifact.load(validation_artifact_path)
            revision_validation_artifact = PartTransitionArtifact.load(
                revision_validation_artifact_path
            )
        else:
            validation_artifact = artifact
            revision_validation_artifact = artifact
        model = _build_instance_model(config["model"], num_classes)
        if official_pdl:
            _attach_pdl_revision_head(model, num_classes)
    else:
        warmup_model = _build_instance_model(config["model"], num_classes)
        if official_pdl:
            _attach_pdl_revision_head(warmup_model, num_classes)
        warmup_optimizer = build_optimizer(warmup_model, config["warmup"]["optimizer"])
        warmup_scheduler = build_scheduler(
            warmup_optimizer, config["warmup"].get("scheduler"),
            int(config["warmup"]["epochs"]),
        )
        warmup_loss = build_builtin_loss(config["warmup"].get("loss", {"name": "ce"}))
        warmup_epochs = int(config["warmup"]["epochs"])
        best_state = None
        best_accuracy = float("-inf")
        for _ in range(warmup_epochs):
            pretrain_noisy_classifier(warmup_model, warmup_optimizer, train_loader,
                device, epochs=1, criterion=warmup_loss)
            score = evaluate_classification(warmup_model, noisy_validation_loader, warmup_loss, device)
            if score["accuracy"] > best_accuracy:
                best_accuracy = score["accuracy"]
                best_state = {key: value.detach().cpu().clone() for key, value in warmup_model.state_dict().items()}
            if warmup_scheduler is not None:
                warmup_scheduler.step()
        if best_state is not None:
            warmup_model.load_state_dict(best_state)
        torch.save({"model": warmup_model.state_dict(), "best_noisy_validation_accuracy": best_accuracy},
            run_dir / "warmup_model.pt")
        posterior = collect_posterior_snapshot(warmup_model, union_loader, device,
            dataset=dataset_name, split="train")
        features = collect_feature_snapshot(warmup_model, union_loader, device,
            dataset=dataset_name, split="train",
            feature_extractor=lambda model, inputs: forward_with_features(model, inputs).features)
        posterior.save(run_dir / "posterior_snapshot.npz")
        features.save(run_dir / "feature_snapshot.npz")
        estimator = build_builtin_instance_transition_estimator(config["instance_transition"])
        if official_pdl:
            # Official PDL runs NMF once on train+validation, but estimates
            # anchors and basis matrices independently on each split.
            # ``main.py`` concatenates the train split in random-choice order
            # followed by validation in np.delete order.  This order matters
            # because official train_m consumes the global NumPy RNG state.
            representation_indices = np.concatenate(
                [train_indices, validation_indices]
            ).astype(np.int64, copy=False)
            representation_positions = np.searchsorted(
                features.global_indices, representation_indices
            )
            if not np.array_equal(
                features.global_indices[representation_positions], representation_indices
            ):
                raise KeyError("PDL official representation order is not covered")
            representation_input = features.features[representation_positions]
            representation_parts, representation_coefficients = fit_part_representation(
                representation_input,
                estimator.num_parts,
                seed=None,
                iterations=estimator.representation_iterations,
                error_tolerance=estimator.representation_error_tolerance,
            )
            train_features, train_posteriors = _subset_pdl_snapshots(
                features,
                posterior,
                train_indices,
                split="train",
            )
            validation_features, validation_posteriors = _subset_pdl_snapshots(
                features,
                posterior,
                validation_indices,
                split="noisy_validation",
            )
            train_features.save(run_dir / "train_feature_snapshot.npz")
            validation_features.save(run_dir / "validation_feature_snapshot.npz")
            train_posteriors.save(run_dir / "train_posterior_snapshot.npz")
            validation_posteriors.save(run_dir / "validation_posterior_snapshot.npz")
            percentages = estimator.anchor_percentages_for_paper()
            train_anchor_positions = select_pdl_anchor_candidates(
                train_posteriors.noisy_probabilities, percentages
            )
            validation_anchor_positions = select_pdl_anchor_candidates(
                validation_posteriors.noisy_probabilities, percentages
            )
            train_representation_positions = np.searchsorted(
                representation_indices, train_posteriors.global_indices
            )
            validation_representation_positions = np.searchsorted(
                representation_indices, validation_posteriors.global_indices
            )
            if (
                np.any(train_representation_positions >= representation_indices.size)
                or np.any(validation_representation_positions >= representation_indices.size)
                or not np.array_equal(
                representation_indices[train_representation_positions],
                train_posteriors.global_indices,
                )
                or not np.array_equal(
                representation_indices[validation_representation_positions],
                validation_posteriors.global_indices,
                )
            ):
                raise KeyError("PDL shared representation does not cover split indices")
            train_split_coefficients = representation_coefficients[
                train_representation_positions
            ]
            validation_split_coefficients = representation_coefficients[
                validation_representation_positions
            ]
            train_basis, validation_basis = fit_pdl_basis_matrices_pair(
                train_split_coefficients[train_anchor_positions],
                train_posteriors.noisy_probabilities[train_anchor_positions],
                validation_split_coefficients[validation_anchor_positions],
                validation_posteriors.noisy_probabilities[validation_anchor_positions],
                epochs=estimator.basis_epochs,
                learning_rate=estimator.basis_learning_rate,
                loss_threshold=estimator.basis_loss_threshold,
                seed=estimator.representation_seed,
                official_raw=True,
            )
            # The reference clips only the train basis group before
            # correction; validation is normalized later by tools.norm().
            train_basis = np.where(train_basis < 1.0e-6, 0.0, train_basis)
            artifact = estimator.estimate_from_shared_representation(
                train_features,
                train_posteriors,
                representation_parts=representation_parts,
                representation_coefficients=representation_coefficients,
                representation_indices=representation_indices,
                part_matrices=train_basis,
                official_raw_basis=True,
            )
            validation_artifact = estimator.estimate_from_shared_representation(
                validation_features,
                validation_posteriors,
                representation_parts=representation_parts,
                representation_coefficients=representation_coefficients,
                representation_indices=representation_indices,
                part_matrices=validation_basis,
                official_raw_basis=True,
            )
            # This mirrors the official val_revision call: validation W is
            # paired with the train-fitted basis matrices.
            revision_validation_artifact = validation_artifact.with_part_matrices(
                artifact.part_matrices,
                role="revision_validation",
                source_artifact_hash=artifact.artifact_hash,
            )
        else:
            artifact = estimator.estimate(features, posterior)
            validation_artifact = artifact
            revision_validation_artifact = artifact
        artifact.save(artifact_path)
        if official_pdl:
            validation_artifact.save(validation_artifact_path)
            revision_validation_artifact.save(revision_validation_artifact_path)
        model = warmup_model

    if official_pdl:
        return _run_pdl_official_phases(
            config=config,
            run_dir=run_dir,
            model=model,
            artifact=artifact,
            validation_artifact=validation_artifact,
            revision_validation_artifact=revision_validation_artifact,
            train_loader=train_loader,
            validation_loader=noisy_validation_loader,
            test_loader=test_loader,
            device=device,
            seed=seed,
            epochs=epochs,
            parameter_record=parameter_record,
            resume=resume,
        )

    model = _build_instance_model(config["model"], num_classes)
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(optimizer, config.get("scheduler"), epochs)
    criterion = build_builtin_loss(config.get("loss", {"name": "ce"})).to(device)
    algorithm: InstanceTransitionClassificationAlgorithm = build_builtin_instance_transition_algorithm(
        config["algorithm"], model=model, optimizer=optimizer, loss=criterion,
        transition=artifact, device=device,
    )
    algorithm.setup(ExperimentContext(run_dir, config, seed))
    state = RunState(phase="corrected_train")
    completed_epoch, best_epoch, best_accuracy = -1, -1, float("-inf")
    if resume is not None:
        state, completed_epoch, checkpoint_payload = load_checkpoint(
            resume, algorithm, device, scheduler=scheduler
        )
        best_epoch = int(checkpoint_payload["best_epoch"])
        best_accuracy = float(checkpoint_payload["best_selection_accuracy"])

    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if parameter_record is not None:
        (run_dir / "parameter_record.json").write_text(
            json.dumps(parameter_record.to_dict(), indent=2), encoding="utf-8")
    _write_environment(run_dir / "environment.json", seed, device)
    (run_dir / "pipeline_state.json").write_text(json.dumps({
        "phase": "corrected_train", "artifact_hash": artifact.artifact_hash,
        "feature_snapshot_hash": artifact.feature_snapshot_hash,
        "posterior_snapshot_hash": artifact.posterior_snapshot_hash,
        "train_indices": train_indices.tolist(), "noisy_validation_indices": validation_indices.tolist(),
    }, indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.jsonl"
    rows = []
    if metrics_path.is_file():
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("event") == "epoch"]
    algorithm.on_run_start(state)
    with metrics_path.open("a", encoding="utf-8") as handle:
        for epoch in range(completed_epoch + 1, epochs):
            state.cycle = epoch
            algorithm.on_cycle_start(state)
            loss_sum = accuracy_sum = samples = 0.0
            for raw_batch in train_loader:
                result = algorithm.step(Batch(raw_batch), state)
                count = result.metrics["samples"]
                samples += count
                loss_sum += result.metrics["loss"] * count
                accuracy_sum += result.metrics["accuracy"] * count
            algorithm.on_cycle_end(state)
            selection = evaluate_classification(model, noisy_validation_loader, criterion, device)
            row = standardize_epoch_row({"event": "epoch", "epoch": epoch + 1,
                "global_step": state.step, "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": loss_sum / samples, "train_accuracy": accuracy_sum / samples,
                "selection_split": "noisy_validation", "selection_loss": selection["loss"],
                "selection_accuracy": selection["accuracy"]})
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if scheduler is not None:
                scheduler.step()
            if selection["accuracy"] > best_accuracy:
                best_accuracy, best_epoch = selection["accuracy"], epoch + 1
                save_checkpoint(run_dir / "best.pt", algorithm, state, epoch, config,
                    scheduler=scheduler, best_epoch=best_epoch,
                    best_selection_accuracy=best_accuracy, selection_split="noisy_validation",
                    pipeline={"phase": "corrected_train", "artifact_hash": artifact.artifact_hash})
            save_checkpoint(run_dir / "last.pt", algorithm, state, epoch, config,
                scheduler=scheduler, best_epoch=best_epoch,
                best_selection_accuracy=best_accuracy, selection_split="noisy_validation",
                pipeline={"phase": "corrected_train", "artifact_hash": artifact.artifact_hash})
            if bool(config["trainer"].get("progress", {}).get("curves", True)):
                write_training_curves_svg(rows, run_dir / "training_curves.svg")
        algorithm.on_run_end(state)
        test = evaluate_classification(model, test_loader, criterion, device)
        final = {"event": "final", "completed_epochs": epochs, "global_step": state.step,
            "best_epoch": best_epoch, "best_selection_accuracy": best_accuracy,
            "test_loss": test["loss"], "test_accuracy": test["accuracy"],
            "artifact_hash": artifact.artifact_hash}
        handle.write(json.dumps(final) + "\n")
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return run_dir


__all__ = ["run_instance_transition_experiment"]
