from __future__ import annotations

"""Controlled CIFAR evidence chain for ordinary T versus Dual-T."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.dual_t import DualTAlgorithm, DualTConfig
from lnl_toolbox.algorithms.dual_t.algorithm import _train_supervised_epoch
from lnl_toolbox.algorithms.dual_t.evidence import (
    FinalArmEvidence,
    build_transition_evidence,
)
from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.algorithms.transition_risk import ForwardRiskCorrector
from lnl_toolbox.core import ExperimentContext, RunState
from lnl_toolbox.data import NoisyTargetDataset
from lnl_toolbox.data.cifar import load_cifar10
from lnl_toolbox.data.torch_cifar import (
    TorchCifarDataset,
    build_cifar_transform,
    cifar_pixel_mean,
    stratified_split,
)
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.noise.estimators import PosteriorSnapshot
from lnl_toolbox.noise.transition import TransitionArtifact
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.experiment import (
    _environment,
    _loader,
    _resolved_noise_config,
    _subset,
    build_model,
    build_optimizer,
    build_scheduler,
)
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    effective_subset_actual_rate,
    noise_mode,
    prepare_noise_manifest,
)
from lnl_toolbox.training.snapshots import collect_posterior_snapshot


def _tensor_state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError("model initial state must contain only tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class _RecordingLoader:
    """Record the actual stable-index order and transformed input tensors."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader
        self.batch_index_hashes: list[str] = []
        self.input_tensor_hashes: list[str] = []

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        index_digest = hashlib.sha256()
        input_digest = hashlib.sha256()
        for batch in self.loader:
            indices = torch.as_tensor(batch["index"]).detach().cpu().contiguous()
            inputs = torch.as_tensor(batch["input"]).detach().cpu().contiguous()
            index_digest.update(str(tuple(indices.shape)).encode("ascii"))
            index_digest.update(str(indices.dtype).encode("ascii"))
            index_digest.update(indices.numpy().tobytes(order="C"))
            input_digest.update(str(tuple(inputs.shape)).encode("ascii"))
            input_digest.update(str(inputs.dtype).encode("ascii"))
            input_digest.update(inputs.numpy().tobytes(order="C"))
            yield batch
        self.batch_index_hashes.append(index_digest.hexdigest())
        self.input_tensor_hashes.append(input_digest.hexdigest())


def _run_final_arm(
    *,
    name: str,
    initial_state: Mapping[str, Any],
    model_config: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    scheduler_config: Mapping[str, Any],
    epochs: int,
    num_classes: int,
    train_dataset: Any,
    noisy_validation_dataset: Any,
    clean_test_dataset: Any,
    loader_config: Mapping[str, Any],
    sampler_seed: int,
    rng_state: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    transition: TransitionArtifact | None,
) -> FinalArmEvidence:
    """Train one independent final arm from the shared initialization."""

    restore_rng_state(rng_state)
    model = build_model(model_config, num_classes)
    model.load_state_dict(initial_state)
    optimizer = build_optimizer(model, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config, epochs)
    criterion = build_builtin_loss({"name": "ce"}).to(device)
    train_loader = _RecordingLoader(
        _loader(
            train_dataset,
            loader_config,
            shuffle=True,
            seed=sampler_seed,
        )
    )
    noisy_validation_loader = _loader(
        noisy_validation_dataset,
        loader_config,
        shuffle=False,
        seed=sampler_seed,
    )
    clean_test_loader = _loader(
        clean_test_dataset,
        loader_config,
        shuffle=False,
        seed=sampler_seed,
    )
    risk_corrector = None if transition is None else ForwardRiskCorrector()
    algorithm = SupervisedClassificationAlgorithm(
        model,
        optimizer,
        criterion,
        device,
        risk_corrector=risk_corrector,
        transition=transition,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    state = RunState(phase=f"evidence_{name}")
    algorithm.setup(ExperimentContext(run_dir, {"arm": name}, sampler_seed))
    algorithm.on_run_start(state)
    best_accuracy = float("-inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    metric_rows: list[dict[str, Any]] = []
    try:
        for epoch in range(epochs):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_metrics = _train_supervised_epoch(
                algorithm,
                train_loader,
                state,
                epoch,
            )
            validation = evaluate_classification(
                model,
                noisy_validation_loader,
                criterion,
                device,
            )
            if validation["accuracy"] > best_accuracy:
                best_accuracy = float(validation["accuracy"])
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if scheduler is not None:
                scheduler.step()
            metric_rows.append({
                "event": "epoch",
                "arm": name,
                "epoch": epoch + 1,
                "global_step": state.step,
                "learning_rate": learning_rate,
                **train_metrics,
                "noisy_validation_loss": validation["loss"],
                "noisy_validation_accuracy": validation["accuracy"],
            })
        if best_state is None:
            raise RuntimeError(f"evidence arm {name!r} has no best checkpoint")
        final_test = evaluate_classification(
            model,
            clean_test_loader,
            criterion,
            device,
        )
        final_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(best_state)
        best_test = evaluate_classification(
            model,
            clean_test_loader,
            criterion,
            device,
        )
        model.load_state_dict(final_state)
        best_path = run_dir / "best.pt"
        atomic_save(
            {
                "arm": name,
                "completed_epoch": best_epoch,
                "model": best_state,
                "initial_state_hash": _tensor_state_hash(initial_state),
            },
            best_path,
        )
        evidence = FinalArmEvidence(
            name=name,
            initial_state_hash=_tensor_state_hash(initial_state),
            sampler_seed=sampler_seed,
            completed_epochs=epochs,
            global_step=state.step,
            best_validation_epoch=best_epoch + 1,
            best_noisy_validation_accuracy=best_accuracy,
            best_checkpoint_clean_test_loss=float(best_test["loss"]),
            best_checkpoint_clean_test_accuracy=float(
                best_test["accuracy"]
            ),
            final_epoch_clean_test_loss=float(final_test["loss"]),
            final_epoch_clean_test_accuracy=float(final_test["accuracy"]),
            batch_index_hashes=tuple(train_loader.batch_index_hashes),
            input_tensor_hashes=tuple(train_loader.input_tensor_hashes),
        )
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in metric_rows),
            encoding="utf-8",
        )
        (run_dir / "final_metrics.json").write_text(
            json.dumps(evidence.to_dict(), indent=2),
            encoding="utf-8",
        )
        algorithm.on_run_end(state)
        return evidence
    finally:
        algorithm.close()


def _require_fair_final_arms(
    arms: Mapping[str, FinalArmEvidence],
) -> None:
    values = list(arms.values())
    if len({value.initial_state_hash for value in values}) != 1:
        raise RuntimeError("evidence final arms did not share initialization")
    if len({value.sampler_seed for value in values}) != 1:
        raise RuntimeError("evidence final arms did not share sampler seed")
    if len({value.batch_index_hashes for value in values}) != 1:
        raise RuntimeError("evidence final arms saw different sample orders")
    if len({value.input_tensor_hashes for value in values}) != 1:
        raise RuntimeError(
            "evidence final arms saw different transformed input tensors"
        )


def run_dual_t_evidence_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> Path:
    """Run one shared-posterior ordinary-T versus Dual-T evidence chain."""

    config = deepcopy(config)
    method_config = DualTConfig.from_mapping(config)
    data_config = config["data"]
    dataset_name = str(data_config.get("name", "")).strip().lower()
    if dataset_name != "cifar10":
        raise ValueError("Dual-T evidence first version supports CIFAR-10 only")
    evidence_config = config.get("evidence", {}) or {}
    if not isinstance(evidence_config, Mapping):
        raise TypeError("evidence configuration must be a mapping")
    include_noisy_ce = bool(evidence_config.get("include_noisy_ce", True))
    final_seed = int(evidence_config.get("final_seed", config.get("seed", 1)))
    sampler_seed = int(
        evidence_config.get("sampler_seed", final_seed + 1000)
    )
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    trainer_config = config.get("trainer", {}) or {}
    if not isinstance(trainer_config, Mapping):
        raise TypeError("trainer configuration must be a mapping")
    device = resolve_device(trainer_config.get("device", "auto"))

    if output_dir is None:
        run_dir = Path(config.get("output_root", "artifacts/runs")) / (
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    else:
        run_dir = Path(output_dir)
    run_dir = run_dir.expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"evidence output directory already exists: {run_dir}"
        )
    run_dir.mkdir(parents=True)

    train_data = load_cifar10(data_config.get("root"), "train")
    test_data = load_cifar10(data_config.get("root"), "test")
    num_classes = 10
    full_train_indices, validation_indices = stratified_split(
        train_data.labels,
        int(data_config["validation_size"]),
        seed,
    )
    manifest_indices = np.sort(
        np.concatenate((full_train_indices, validation_indices))
    )
    manifest, manifest_path = prepare_noise_manifest(
        config,
        dataset=dataset_name,
        clean_targets=train_data.labels[manifest_indices],
        global_indices=manifest_indices,
        num_classes=num_classes,
        run_dir=run_dir,
        checkpoint_payload=None,
        dataset_targets=train_data.labels,
    )
    if (
        manifest is None
        or manifest_path is None
        or manifest.transition_matrix is None
    ):
        raise ValueError(
            "Dual-T evidence requires a synthetic manifest transition_matrix"
        )
    train_indices = _subset(
        full_train_indices,
        train_data.labels,
        data_config.get("max_train_samples"),
        seed + 1,
    )
    validation_indices = _subset(
        validation_indices,
        train_data.labels,
        data_config.get("max_validation_samples"),
        seed + 2,
    )
    test_indices = _subset(
        np.arange(len(test_data)),
        test_data.labels,
        data_config.get("max_test_samples"),
        seed + 3,
    )
    preprocessing = str(
        data_config.get("preprocessing", "standard")
    ).lower()
    pixel_mean = (
        cifar_pixel_mean(train_data.images)
        if preprocessing == "gce2018"
        else None
    )
    transform_options = {
        "preprocessing": preprocessing,
        "pixel_mean": pixel_mean,
    }
    clean_train_set = TorchCifarDataset(
        train_data,
        train_indices,
        transform=build_cifar_transform(
            True,
            bool(data_config.get("augment", True)),
            **transform_options,
        ),
    )
    noisy_train_set = NoisyTargetDataset(
        clean_train_set,
        manifest.global_indices,
        manifest.noisy_targets,
    )
    clean_validation_set = TorchCifarDataset(
        train_data,
        validation_indices,
        transform=build_cifar_transform(False, **transform_options),
    )
    noisy_validation_set = NoisyTargetDataset(
        clean_validation_set,
        manifest.global_indices,
        manifest.noisy_targets,
    )
    clean_test_set = TorchCifarDataset(
        test_data,
        test_indices,
        transform=build_cifar_transform(False, **transform_options),
    )
    loader_config = config["loader"]
    posterior_train_loader = _loader(
        noisy_train_set,
        loader_config,
        shuffle=True,
        seed=seed,
    )
    posterior_validation_loader = _loader(
        noisy_validation_set,
        loader_config,
        shuffle=False,
        seed=seed,
    )
    posterior_test_loader = _loader(
        clean_test_set,
        loader_config,
        shuffle=False,
        seed=seed,
    )
    noise_metadata = checkpoint_noise_metadata(
        manifest,
        manifest_path,
        run_dir,
        effective_subset_actual_rate(manifest, train_indices),
        mode=noise_mode(config),
        validation_targets="noisy",
        effective_validation_rate=effective_subset_actual_rate(
            manifest,
            validation_indices,
        ),
    )
    config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(seed, device), indent=2),
        encoding="utf-8",
    )
    (run_dir / "noise_summary.json").write_text(
        json.dumps(noise_metadata, indent=2),
        encoding="utf-8",
    )

    posterior_model = build_model(
        method_config.posterior_stage.model,
        num_classes,
    )
    posterior_optimizer = build_optimizer(
        posterior_model,
        method_config.posterior_stage.optimizer,
    )
    posterior_scheduler = build_scheduler(
        posterior_optimizer,
        method_config.posterior_stage.scheduler,
        method_config.posterior_stage.epochs,
    )
    unused_final_model = build_model(
        method_config.final_stage.model,
        num_classes,
    )
    unused_final_optimizer = build_optimizer(
        unused_final_model,
        method_config.final_stage.optimizer,
    )
    unused_final_scheduler = build_scheduler(
        unused_final_optimizer,
        method_config.final_stage.scheduler,
        method_config.final_stage.epochs,
    )
    posterior_loss = build_builtin_loss(
        dict(config["posterior_stage"]).get("loss", {"name": "ce"})
    ).to(device)
    posterior_owner = DualTAlgorithm(
        posterior_model=posterior_model,
        posterior_optimizer=posterior_optimizer,
        posterior_scheduler=posterior_scheduler,
        final_model=unused_final_model,
        final_optimizer=unused_final_optimizer,
        final_scheduler=unused_final_scheduler,
        posterior_loss=posterior_loss,
        final_loss=build_builtin_loss({"name": "ce"}).to(device),
        train_loader=posterior_train_loader,
        noisy_validation_loader=posterior_validation_loader,
        clean_test_loader=posterior_test_loader,
        device=device,
        run_dir=run_dir,
        config=config,
        dataset=dataset_name,
        noise_metadata=noise_metadata,
    )
    try:
        posterior_owner.train_posterior()
        best_payload = read_checkpoint(
            posterior_owner.posterior_best_path,
            "cpu",
        )
        if best_payload.get("checkpoint_role") != "posterior_best":
            raise ValueError("posterior best checkpoint identity mismatch")
        posterior_owner.posterior_algorithm.load_state_dict(
            best_payload["posterior_algorithm"]
        )
        snapshot = collect_posterior_snapshot(
            posterior_owner.posterior_algorithm.model,
            posterior_train_loader,
            device,
            dataset=dataset_name,
            split="train",
        )
        snapshot_path = run_dir / "posterior_snapshot.npz"
        snapshot.save(snapshot_path)
        persisted_snapshot = PosteriorSnapshot.load(snapshot_path)
        if persisted_snapshot.snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("persisted evidence snapshot hash mismatch")
        transition_evidence = build_transition_evidence(
            snapshot=persisted_snapshot,
            manifest=manifest,
            sample_indices=train_indices,
            metadata={
                "posterior_best_checkpoint_sha256": (
                    posterior_owner.state.best_posterior_checkpoint_sha256
                ),
                "noise_manifest_sha256": noise_metadata["manifest_sha256"],
                "noise_mapping_hash": noise_metadata["mapping_hash"],
            },
        )
        anchor_path = run_dir / "transition_anchor.npz"
        dual_t_path = run_dir / "transition_dual_t.npz"
        transition_evidence.anchor_artifact.save(anchor_path)
        transition_evidence.dual_t_artifact.save(dual_t_path)
        anchor = TransitionArtifact.load(anchor_path)
        dual_t = TransitionArtifact.load(dual_t_path)
        if {
            anchor.source_snapshot_hash,
            dual_t.source_snapshot_hash,
        } != {persisted_snapshot.snapshot_hash}:
            raise ValueError("evidence artifacts do not share one snapshot")
    finally:
        posterior_owner.close()

    seed_everything(final_seed)
    reference_model = build_model(
        method_config.final_stage.model,
        num_classes,
    )
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in reference_model.state_dict().items()
    }
    initial_state_hash = _tensor_state_hash(initial_state)
    arm_rng_state = capture_rng_state()
    arm_specs: list[tuple[str, TransitionArtifact | None]] = [
        ("anchor_forward", anchor),
        ("dual_t_forward", dual_t),
    ]
    if include_noisy_ce:
        arm_specs.insert(0, ("noisy_ce", None))
    arms: dict[str, FinalArmEvidence] = {}
    for name, transition in arm_specs:
        evidence = _run_final_arm(
            name=name,
            initial_state=initial_state,
            model_config=method_config.final_stage.model,
            optimizer_config=method_config.final_stage.optimizer,
            scheduler_config=method_config.final_stage.scheduler,
            epochs=method_config.final_stage.epochs,
            num_classes=num_classes,
            train_dataset=noisy_train_set,
            noisy_validation_dataset=noisy_validation_set,
            clean_test_dataset=clean_test_set,
            loader_config=loader_config,
            sampler_seed=sampler_seed,
            rng_state=arm_rng_state,
            device=device,
            run_dir=run_dir / "arms" / name,
            transition=transition,
        )
        if evidence.initial_state_hash != initial_state_hash:
            raise RuntimeError(f"arm {name!r} changed initial-state identity")
        arms[name] = evidence
    _require_fair_final_arms(arms)

    anchor_accuracy = arms[
        "anchor_forward"
    ].best_checkpoint_clean_test_accuracy
    dual_accuracy = arms[
        "dual_t_forward"
    ].best_checkpoint_clean_test_accuracy
    summary = {
        "event": "dual_t_evidence",
        "dataset": dataset_name,
        "noise": {
            "type": manifest.noise_type,
            "target_rate": manifest.requested_rate,
            "manifest_sha256": noise_metadata["manifest_sha256"],
            "mapping_hash": noise_metadata["mapping_hash"],
        },
        "posterior": {
            "best_epoch": posterior_owner.state.best_posterior_epoch + 1,
            "best_noisy_validation_accuracy": (
                posterior_owner.state.best_posterior_validation_accuracy
            ),
            "best_checkpoint_sha256": (
                posterior_owner.state.best_posterior_checkpoint_sha256
            ),
            "snapshot_hash": persisted_snapshot.snapshot_hash,
        },
        "transition_evidence": transition_evidence.to_dict(),
        "arms": {name: value.to_dict() for name, value in arms.items()},
        "classification_comparison": {
            "anchor_forward_best_checkpoint_clean_test_accuracy": (
                anchor_accuracy
            ),
            "dual_t_forward_best_checkpoint_clean_test_accuracy": (
                dual_accuracy
            ),
            "dual_t_minus_anchor_best_checkpoint_clean_test_accuracy": (
                dual_accuracy - anchor_accuracy
            ),
        },
        "fairness": {
            "same_initial_state": True,
            "initial_state_hash": initial_state_hash,
            "same_sampler_seed": True,
            "sampler_seed": sampler_seed,
            "same_batch_index_order": True,
            "same_transformed_inputs": True,
            "augmentation": bool(data_config.get("augment", True)),
            "boundary": (
                "independent loaders use the same explicit sampler seed and "
                "each sequential arm restores the same Python, NumPy, CPU "
                "and CUDA RNG state"
            ),
        },
        "claims": {
            "tiny_run_requires_dual_t_lower_matrix_error": False,
            "tiny_run_requires_dual_t_higher_accuracy": False,
        },
    }
    (run_dir / "evidence_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return run_dir
