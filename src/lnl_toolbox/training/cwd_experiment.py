from __future__ import annotations

"""Standalone CWD lifecycle for the CIFAR airplane/automobile benchmark."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from lnl_toolbox.algorithms.cwd import CWDGlobalObjective
from lnl_toolbox.data.binary_benchmarks import (
    cifar_airplane_automobile_view,
    corrupt_binary_labels,
    stratified_binary_splits,
)
from lnl_toolbox.data.cifar import CifarData, load_cifar10
from lnl_toolbox.data.torch_cifar import (
    TorchCifarDataset,
    build_cifar_transform,
    stratified_split,
)
from lnl_toolbox.estimators.cwd import CWDEstimator
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.models.cifar_resnet import cifar_resnet34
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.models.tiny_cnn import TinyCNN
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.interfaces import RunContext
from lnl_toolbox.training.epoch_stream import (
    build_epoch_loader,
    loader_stream_metadata,
    validate_loader_stream,
)
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.snapshots import collect_feature_snapshot


def _loader(dataset, config: Mapping[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", True)),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
        drop_last=bool(config.get("drop_last", False)) if shuffle else False,
    )


def _binary_cifar_corpus(root: str | Path | None) -> tuple[CifarData, np.ndarray]:
    train = load_cifar10(root, "train")
    test = load_cifar10(root, "test")
    train_images, train_labels, train_source = cifar_airplane_automobile_view(train)
    test_images, test_labels, test_source = cifar_airplane_automobile_view(test)
    data = CifarData(
        np.concatenate((train_images, test_images)),
        np.concatenate((train_labels, test_labels)),
        ("airplane", "automobile"),
        "combined",
        "cifar10_airplane_automobile",
    )
    source_indices = np.concatenate((train_source, len(train) + test_source))
    return data, source_indices


def _limit_indices(labels: np.ndarray, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or int(maximum) >= labels.size:
        return np.arange(labels.size, dtype=np.int64)
    if int(maximum) < 4:
        raise ValueError("CWD max_samples must contain at least four samples")
    random = np.random.default_rng(seed)
    parts = []
    for class_index in (0, 1):
        candidates = np.flatnonzero(labels == class_index)
        random.shuffle(candidates)
        parts.append(candidates[: int(maximum) // 2])
    return np.sort(np.concatenate(parts))


def _build_model(config: Mapping[str, Any]):
    name = str(config.get("name", "resnet34")).strip().lower()
    if name in {"resnet34", "cifar_resnet34"}:
        outputs = int(config.get("num_outputs", 2))
        return cifar_resnet34(
            outputs,
            int(config.get("base_width", 64)),
            bias=bool(config.get("bias", True)),
        )
    if name == "tiny_cnn":
        return TinyCNN(2, int(config.get("width", 8)))
    raise ValueError(f"Unsupported CWD model: {name}")


def _classification_from_cwd_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("CWD logits must have shape [B, C]")
    if logits.shape[1] == 1:
        return (logits[:, 0] >= 0.0).long()
    return logits.argmax(1)


@torch.no_grad()
def _evaluate_cwd(model, loader, device, *, scalar_binary: bool) -> dict[str, float]:
    model.eval()
    loss_sum = correct = samples = 0.0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        output = forward_with_features(model, inputs)
        if scalar_binary:
            signed = targets.to(dtype=output.logits.dtype).mul(2.0).sub(1.0)
            margins = output.logits[:, 0]
            loss = (1.0 - signed * margins).square()
        else:
            loss = torch.nn.functional.cross_entropy(output.logits, targets, reduction="none")
        predictions = _classification_from_cwd_logits(output.logits)
        count = int(targets.numel())
        loss_sum += float(loss.sum())
        correct += float(predictions.eq(targets).sum())
        samples += count
    if samples == 0:
        raise ValueError("CWD evaluation loader must not be empty")
    return {"loss": loss_sum / samples, "accuracy": correct / samples}


def _train_epoch(model, optimizer, loader, objective, device) -> tuple[float, float]:
    model.train()
    loss_sum = correct = samples = 0.0
    base_loss = CrossEntropyLoss().to(device)
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        indices = batch["index"].to(device, non_blocking=True)
        output = forward_with_features(model, inputs)
        loss = objective.compute(
            model=model,
            logits=output.logits,
            features=output.features,
            noisy_targets=targets,
            sample_indices=indices,
            base_loss=base_loss,
            metadata={},
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count = int(targets.numel())
        loss_sum += float(loss.detach()) * count
        correct += float(_classification_from_cwd_logits(output.logits).eq(targets).sum())
        samples += count
    return loss_sum / samples, correct / samples


def _run_cwd_single_fold(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *,
    context: RunContext | None = None,
) -> Path:
    """Run one configured fold of the paper's five-fold CIFAR-binary protocol."""

    config = deepcopy(config)
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    device = resolve_device(str(config["trainer"].get("device", "auto")))
    run_dir = (
        Path(resume).resolve().parent
        if resume is not None
        else Path(
            output_dir
            or Path(config.get("output_root", "artifacts/runs"))
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        ).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None if resume is None else read_checkpoint(resume, "cpu")
    saved_config = None if checkpoint is None else checkpoint.get("config")
    if isinstance(saved_config, dict):
        saved_config = deepcopy(saved_config)
        saved_config["method"] = "cwd"
    comparable_config = deepcopy(config)
    comparable_config["method"] = "cwd"
    if checkpoint is not None and saved_config != comparable_config:
        raise ValueError("CWD resume configuration mismatch")

    data_config = config["data"]
    corpus, source_indices = _binary_cifar_corpus(data_config.get("root"))
    limited = _limit_indices(corpus.labels, data_config.get("max_samples"), seed)
    images = corpus.images[limited]
    clean_labels = corpus.labels[limited]
    source_indices = source_indices[limited]
    folds = int(data_config.get("folds", 5))
    fold_index = int(data_config.get("fold_index", 0))
    splits = stratified_binary_splits(clean_labels, folds=folds, seed=seed)
    if not 0 <= fold_index < len(splits):
        raise ValueError("CWD fold_index is outside the configured fold range")
    train_positions, test_positions = splits[fold_index]
    validation_size = int(data_config.get("validation_size", 0))
    if validation_size < 0:
        raise ValueError("CWD validation_size must be non-negative")
    if validation_size:
        retained, held_out = stratified_split(
            clean_labels[train_positions], validation_size, seed + 101
        )
        validation_positions = train_positions[held_out]
        train_positions = train_positions[retained]
    else:
        validation_positions = np.asarray([], dtype=np.int64)

    noise_config = config["noise"]
    rho_positive = float(noise_config["rho_positive"])
    rho_negative = float(noise_config["rho_negative"])
    noise_seed = int(noise_config.get("seed", seed))
    manifest = corrupt_binary_labels(
        clean_labels[train_positions], rho_positive, rho_negative, noise_seed
    )
    manifest.dataset = "cifar10_airplane_automobile"
    manifest.metadata.update(
        {
            "folds": folds,
            "fold_index": fold_index,
            "source_global_indices": source_indices[train_positions].tolist(),
        }
    )
    manifest_path = run_dir / "noise_manifest.npz"
    if checkpoint is None:
        manifest.save(manifest_path)
    else:
        expected = checkpoint.get("noise_mapping_hash")
        if expected != manifest.mapping_hash:
            raise ValueError("CWD resume noise manifest mismatch")

    train_data = CifarData(
        images[train_positions],
        manifest.noisy_targets,
        corpus.class_names,
        "train",
        corpus.dataset,
    )
    test_data = CifarData(
        images[test_positions],
        clean_labels[test_positions],
        corpus.class_names,
        "test",
        corpus.dataset,
    )
    validation_data = None
    if validation_positions.size:
        validation_data = CifarData(
            images[validation_positions],
            clean_labels[validation_positions],
            corpus.class_names,
            "validation",
            corpus.dataset,
        )
    transform = build_cifar_transform(
        True, bool(data_config.get("augment", False))
    )
    evaluation_transform = build_cifar_transform(False)
    train_set = TorchCifarDataset(train_data, transform=transform)
    snapshot_set = TorchCifarDataset(train_data, transform=evaluation_transform)
    test_set = TorchCifarDataset(test_data, transform=evaluation_transform)
    validation_set = (
        None
        if validation_data is None
        else TorchCifarDataset(validation_data, transform=evaluation_transform)
    )
    loader_config = config["loader"]
    snapshot_loader = _loader(snapshot_set, loader_config, shuffle=False, seed=seed)
    validation_loader = (
        None
        if validation_set is None
        else _loader(validation_set, loader_config, shuffle=False, seed=seed + 1)
    )
    test_loader = _loader(test_set, loader_config, shuffle=False, seed=seed)

    cwd_config = config.get("cwd", {})
    cwd_variant = str(cwd_config.get("variant", "multiclass")).strip().lower()
    model_config = dict(config["model"])
    if cwd_variant == "binary_scalar":
        model_config["num_outputs"] = 1
    model = _build_model(model_config).to(device)
    optimizer_config = config["optimizer"]
    if str(optimizer_config.get("name", "adam")).lower() != "adam":
        raise ValueError("CWD paper runner requires Adam")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    milestones = [int(value) for value in config.get("scheduler", {}).get("milestones", [])]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=float(config.get("scheduler", {}).get("gamma", 0.1)),
    )
    start_epoch = 0
    rows: list[dict[str, Any]] = []
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        exact_loader_resume = validate_loader_stream(
            checkpoint.get("loader_stream"),
            base_seed=seed,
            namespace="cwd.train",
            next_epoch=start_epoch,
        )
        if not exact_loader_resume:
            warnings.warn(
                "Legacy CWD checkpoint has no loader_stream metadata; "
                "epoch-boundary input-stream resume is best effort.",
                RuntimeWarning,
                stacklevel=2,
            )
        rows = list(checkpoint.get("metrics", []))

    transition = np.asarray(
        [
            [1.0 - rho_negative, rho_negative],
            [rho_positive, 1.0 - rho_positive],
        ],
        dtype=np.float64,
    )
    pinv_rcond = config.get("cwd", {}).get("pinv_rcond")
    if pinv_rcond is None and "ridge" in config.get("cwd", {}):
        # Backward-compatible parsing for old smoke configurations.  The
        # reproduction configuration uses the paper's unregularized pinv.
        pinv_rcond = float(config["cwd"]["ridge"])
    estimator = CWDEstimator(transition, pinv_rcond=pinv_rcond)
    criterion = CrossEntropyLoss().to(device)
    epochs = int(config["trainer"]["epochs"])
    metrics_path = run_dir / "metrics.jsonl"
    session = context.session if context is not None and context.state.get("lifecycle_active") else None
    if session is not None:
        session.start_phase("fold_training", total_units=epochs)
    for epoch in range(start_epoch, epochs):
        snapshot = collect_feature_snapshot(
            model,
            snapshot_loader,
            device,
            dataset=corpus.dataset,
            split=f"train_fold_{fold_index}",
            feature_extractor=lambda current, inputs: forward_with_features(current, inputs).features,
        )
        artifact = estimator.estimate(snapshot)
        snapshot.save(run_dir / "feature_snapshot.npz")
        artifact.save(run_dir / "statistic_artifact.npz")
        objective = CWDGlobalObjective(
            artifact,
            variant=str(config.get("cwd", {}).get("variant", "multiclass")),
            dynamic_centroid=bool(
                config.get("cwd", {}).get("dynamic_centroid", False)
            ),
        )
        train_loader = build_epoch_loader(
            train_set,
            loader_config,
            base_seed=seed,
            namespace="cwd.train",
            epoch=epoch,
            drop_last=bool(loader_config.get("drop_last", False)),
        )
        train_loss, train_accuracy = _train_epoch(
            model, optimizer, train_loader, objective, device
        )
        row = {
            "event": "epoch",
            "epoch": epoch + 1,
            "phase": "cwd",
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "statistic_hash": artifact.artifact_hash,
            "feature_snapshot_hash": snapshot.snapshot_hash,
            "fold_index": fold_index,
            "validation_metric": "unavailable",
        }
        if validation_loader is not None:
            validation = _evaluate_cwd(
                model,
                validation_loader,
                device,
                scalar_binary=cwd_variant == "binary_scalar",
            )
            row.update({
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_metric": "clean_validation_accuracy",
            })
            row = standardize_epoch_row(row)
        rows.append(row)
        if session is not None:
            session.log_epoch(
                epoch + 1,
                phase="fold_training",
                **{key: value for key, value in row.items()
                   if key not in {"event", "epoch", "phase", "seq"}},
            )
        else:
            metrics_path.write_text(
                "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
                encoding="utf-8",
            )
        scheduler.step()
        atomic_save(
            {
                "format_version": 1,
                "method": "cwd",
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "completed_epoch": epoch,
                "metrics": rows,
                "rng_state": capture_rng_state(),
                "noise_mapping_hash": manifest.mapping_hash,
                "statistic_hash": artifact.artifact_hash,
                "feature_snapshot_hash": snapshot.snapshot_hash,
                "loader_stream": loader_stream_metadata(
                    base_seed=seed,
                    namespace="cwd.train",
                    next_epoch=epoch + 1,
                ),
            },
            run_dir / "last.pt",
        )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if rows and validation_loader is not None:
        write_training_curves_svg(rows, run_dir / "training_curves.svg")
    test = _evaluate_cwd(
        model,
        test_loader,
        device,
        scalar_binary=cwd_variant == "binary_scalar",
    )
    final = {
        "event": "final",
        "method": "cwd",
        "runner": "cwd",
        "status": "completed",
        "completed_epochs": epochs,
        "fold_index": fold_index,
        "selection_protocol": (
            "independent_clean_validation"
            if validation_loader is not None
            else "fixed_budget_test_final_only"
        ),
        "validation_metric": (
            "clean_validation_accuracy" if validation_loader is not None else "unavailable"
        ),
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
    }
    if session is None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final, sort_keys=True) + "\n")
    if session is not None:
        session.end_phase("fold_training", completed_units=max(0, epochs - start_epoch))
        session.emit(
            "final",
            phase="evaluation",
            **{key: value for key, value in final.items() if key != "event"},
        )
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    return run_dir


def _config_identity(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write orchestration metadata without exposing a partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        loaded = json.loads(temporary.read_text(encoding="utf-8"))
        if loaded != payload:
            raise ValueError(f"CWD orchestration JSON verification failed: {path.name}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fold_result(run_dir: Path, fold_index: int, config: Mapping[str, Any]) -> dict[str, Any] | None:
    checkpoint_path = run_dir / "last.pt"
    final_path = run_dir / "final_metrics.json"
    if not checkpoint_path.is_file() or not final_path.is_file():
        return None
    checkpoint = read_checkpoint(checkpoint_path, "cpu")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    epochs = int(config["trainer"]["epochs"])
    if (
        checkpoint.get("method") != "cwd"
        or checkpoint.get("config") != config
        or int(checkpoint.get("completed_epoch", -1)) + 1 < epochs
        or final.get("status") != "completed"
        or int(final.get("fold_index", -1)) != fold_index
    ):
        return None
    test_accuracy = float(final["test_accuracy"])
    test_loss = float(final["test_loss"])
    if not np.isfinite(test_accuracy) or not np.isfinite(test_loss):
        raise ValueError(f"CWD fold {fold_index} contains non-finite final metrics")
    noise_seed = int(config["noise"].get("seed", config.get("seed", 1)))
    return {
        "fold_index": fold_index,
        "status": "completed",
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "run_directory": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "noise_mapping_hash": str(checkpoint["noise_mapping_hash"]),
        "split_seed": int(config.get("seed", 1)),
        "training_seed": int(config.get("seed", 1)),
        "noise_seed": noise_seed,
    }


def _root_checkpoint(
    config: Mapping[str, Any], fold_results: list[Mapping[str, Any]], *, completed: bool
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": "cwd",
        "config": deepcopy(dict(config)),
        "completed_epoch": len(fold_results) - 1,
        "protocol_state": {
            "protocol": "five_fold",
            "completed": bool(completed),
            "completed_folds": [int(item["fold_index"]) for item in fold_results],
            "fold_results": [dict(item) for item in fold_results],
            "config_identity": _config_identity(config),
        },
    }


def _run_cwd_five_fold(
    config: dict[str, Any],
    output_dir: str | Path | None,
    resume: str | Path | None,
    *,
    context: RunContext | None,
) -> Path:
    config = deepcopy(config)
    data_config = config["data"]
    folds = int(data_config.get("folds", 5))
    if folds != 5:
        raise ValueError("CWD five_fold protocol requires data.folds: 5")
    if "fold_index" in data_config:
        raise ValueError("CWD five_fold protocol manages fold_index internally")
    run_dir = (
        Path(resume).resolve().parent
        if resume is not None
        else Path(
            output_dir
            or Path(config.get("output_root", "artifacts/runs"))
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        ).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    root_last = run_dir / "last.pt"
    if resume is not None:
        checkpoint = read_checkpoint(resume, "cpu")
        state = checkpoint.get("protocol_state")
        if checkpoint.get("method") != "cwd" or not isinstance(state, Mapping):
            raise ValueError("CWD five-fold resume checkpoint is invalid")
        if checkpoint.get("config") != config:
            raise ValueError("CWD five-fold resume configuration mismatch")
        if state.get("protocol") != "five_fold":
            raise ValueError("CWD resume protocol mismatch")
        if state.get("config_identity") != _config_identity(config):
            raise ValueError("CWD five-fold resume configuration identity mismatch")

    fold_results: list[dict[str, Any]] = []
    fold_configs: list[dict[str, Any]] = []
    for fold_index in range(folds):
        fold_config = deepcopy(config)
        fold_config["cwd"]["protocol"] = "single_fold"
        fold_config["data"]["fold_index"] = fold_index
        fold_configs.append(fold_config)
    for fold_index, fold_config in enumerate(fold_configs):
        existing = _fold_result(run_dir / f"fold-{fold_index}", fold_index, fold_config)
        if existing is None:
            break
        fold_results.append(existing)

    if len(fold_results) == folds:
        state = None if resume is None else read_checkpoint(resume, "cpu").get("protocol_state")
        if isinstance(state, Mapping) and bool(state.get("completed")):
            return run_dir

    atomic_save(_root_checkpoint(config, fold_results, completed=False), root_last)
    for fold_index in range(len(fold_results), folds):
        fold_dir = run_dir / f"fold-{fold_index}"
        fold_last = fold_dir / "last.pt"
        fold_resume = fold_last if fold_last.is_file() else None
        _run_cwd_single_fold(
            fold_configs[fold_index], fold_dir, fold_resume, context=None
        )
        result = _fold_result(fold_dir, fold_index, fold_configs[fold_index])
        if result is None:
            raise RuntimeError(f"CWD fold {fold_index} did not produce a completed result")
        fold_results.append(result)
        atomic_save(_root_checkpoint(config, fold_results, completed=False), root_last)

    accuracies = np.asarray(
        [float(item["test_accuracy"]) for item in fold_results], dtype=np.float64
    )
    aggregate: dict[str, Any] = {
        "event": "final",
        "method": "cwd",
        "runner": "cwd",
        "protocol": "five_fold",
        "status": "completed",
        "completed": True,
        "fold_count": folds,
        "fold_results": fold_results,
        "test_accuracy_mean": float(accuracies.mean()),
        "test_accuracy_std": float(accuracies.std(ddof=0)),
        "split_seed": int(config.get("seed", 1)),
        "training_seed": int(config.get("seed", 1)),
        "noise_seed": int(config["noise"].get("seed", config.get("seed", 1))),
        "source_fold_directories": [item["run_directory"] for item in fold_results],
        "selection_protocol": "fixed_budget_test_final_only_per_fold",
        "reproduction_status": "protocol_ready",
    }
    _atomic_write_json(aggregate, run_dir / "aggregate_metrics.json")
    _atomic_write_json(aggregate, run_dir / "final_metrics.json")
    atomic_save(_root_checkpoint(config, fold_results, completed=True), root_last)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if context is not None and context.state.get("lifecycle_active"):
        context.session.emit(
            "final",
            phase="five_fold_evaluation",
            **{key: value for key, value in aggregate.items() if key != "event"},
        )
    return run_dir


def run_cwd_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *,
    context: RunContext | None = None,
) -> Path:
    """Run either one legacy CWD fold or the complete five-fold protocol."""

    protocol = str(config.get("cwd", {}).get("protocol", "single_fold")).strip().lower()
    if protocol == "single_fold":
        return _run_cwd_single_fold(config, output_dir, resume, context=context)
    if protocol == "five_fold":
        return _run_cwd_five_fold(config, output_dir, resume, context=context)
    raise ValueError("cwd.protocol must be one of: single_fold, five_fold")


__all__ = ["run_cwd_experiment"]
