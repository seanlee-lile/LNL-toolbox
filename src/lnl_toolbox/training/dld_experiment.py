from __future__ import annotations

"""Unified, resumable experiment runner for paper-oriented DLD."""

from copy import deepcopy
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from lnl_toolbox.algorithms.dld import (
    DLDAlgorithm,
    DLDConfig,
    DLDLabelPredictor,
    DLDPhase,
    DLDPreCorrectionArtifact,
    DLDState,
    DirectionalDiffusionSchedule,
    PARTITION_CLEAN,
    PARTITION_HARD,
    PARTITION_NOISY,
    construct_y0,
    construct_yn,
    partition_samples,
    persist_precorrection_atomically,
    sample_labels,
    weighted_neighbor_distribution,
)
from lnl_toolbox.data import NoisyTargetDataset
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.multi_view import IndexedMultiViewCifarDataset, build_strong_cifar_transform
from lnl_toolbox.data.torch_cifar import TorchCifarDataset, build_cifar_transform, stratified_split
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state, read_checkpoint, restore_rng_state
from lnl_toolbox.training.dld_pretrained import (
    DLDUPMMainBestSource,
    load_upm_main_best_feature_source,
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
    file_sha256,
    noise_mode,
    prepare_noise_manifest,
)
from lnl_toolbox.training.snapshots import FeatureSnapshot, collect_feature_snapshot


def _model_identity(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


class _ViewDataset(Dataset[dict[str, Any]]):
    def __init__(self, source: Dataset[dict[str, Any]], field: str) -> None:
        self.source = source
        self.field = field

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.source[index]
        return {
            "input": sample[self.field],
            "target": sample["target"],
            "index": sample["index"],
        }


class _IndexDataset(Dataset[dict[str, int]]):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> dict[str, int]:
        return {"index": int(self.indices[position])}


def _feature_snapshot(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    dataset: str,
    split: str,
) -> FeatureSnapshot:
    return collect_feature_snapshot(
        model,
        loader,
        device,
        dataset=dataset,
        split=split,
        feature_extractor=lambda current, inputs: forward_with_features(current, inputs).features,
    )


def _entropy(values: np.ndarray) -> float:
    positive = np.where(values > 0, values, 1.0)
    return float(-(values * np.log(positive)).sum(axis=1).mean())


class DLDWorkflow:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        method_config: DLDConfig,
        run_dir: Path,
        device: torch.device,
        dataset: str,
        num_classes: int,
        noise_metadata: Mapping[str, Any],
        feature_model: torch.nn.Module,
        feature_identity: str,
        feature_source: DLDUPMMainBestSource | None,
        feature_source_provenance: Mapping[str, Any],
        train_indices: np.ndarray,
        dual_view_loader: Any,
        validation_snapshot: FeatureSnapshot,
        test_snapshot: FeatureSnapshot,
        loader_config: Mapping[str, Any],
    ) -> None:
        self.config = dict(config)
        self.method_config = method_config
        self.run_dir = run_dir
        self.device = device
        self.dataset = dataset
        self.num_classes = num_classes
        self.noise_metadata = dict(noise_metadata)
        self.feature_model = feature_model.to(device).eval()
        for parameter in self.feature_model.parameters():
            parameter.requires_grad_(False)
        self.feature_identity = feature_identity
        self.feature_source = feature_source
        self.feature_source_provenance = dict(feature_source_provenance)
        self.train_indices = np.sort(np.asarray(train_indices, dtype=np.int64))
        self.dual_view_loader = dual_view_loader
        self.validation_snapshot = validation_snapshot
        self.test_snapshot = test_snapshot
        self.loader_config = dict(loader_config)
        self.state = DLDState()
        self.artifact: DLDPreCorrectionArtifact | None = None
        self.algorithm: DLDAlgorithm | None = None
        self.best_algorithm_state: dict[str, Any] | None = None

    @property
    def artifact_path(self) -> Path: return self.run_dir / "dld_precorrection.npz"
    @property
    def last_path(self) -> Path: return self.run_dir / "last.pt"
    @property
    def best_path(self) -> Path: return self.run_dir / "best.pt"
    @property
    def metrics_path(self) -> Path: return self.run_dir / "metrics.jsonl"

    def _checkpoint_payload(self, role: str) -> dict[str, Any]:
        return {
            "format_version": 2,
            "method": "dld",
            "checkpoint_role": role,
            "config": self.config,
            "dld_identity_hash": self.method_config.identity_hash,
            "dld_state": self.state.state_dict(),
            "algorithm": None if self.algorithm is None else self.algorithm.state_dict(),
            "best_algorithm_state": self.best_algorithm_state,
            "noise": self.noise_metadata,
            "feature_identity": self.feature_identity,
            "feature_source": self.feature_source_provenance,
            "schedule_identity_hash": None if self.algorithm is None else self.algorithm.schedule.identity_hash,
            "rng_state": capture_rng_state(),
        }

    def _save_last(self) -> None:
        atomic_save(self._checkpoint_payload("run_state"), self.last_path)

    def _append(self, row: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")

    def build_precorrection(self) -> None:
        if self.state.phase is not DLDPhase.FEATURE_EXTRACTION:
            raise ValueError("DLD pre-correction can only run during feature extraction")
        seed_everything(int(self.config.get("seed", 1)) + 4000)
        weak_loader = self.dual_view_loader("input")
        strong_loader = self.dual_view_loader("strong_input")
        weak = _feature_snapshot(
            self.feature_model, weak_loader, self.device,
            dataset=self.dataset, split="train:weak",
        )
        strong = _feature_snapshot(
            self.feature_model, strong_loader, self.device,
            dataset=self.dataset, split="train:strong",
        )
        if not np.array_equal(weak.global_indices, strong.global_indices) or not np.array_equal(weak.noisy_targets, strong.noisy_targets):
            raise ValueError("DLD weak/strong feature snapshots are misaligned")
        feature_w = torch.as_tensor(weak.features, device=self.device, dtype=torch.float64)
        feature_s = torch.as_tensor(strong.features, device=self.device, dtype=torch.float64)
        targets = torch.as_tensor(weak.noisy_targets, device=self.device, dtype=torch.int64)
        indices = torch.as_tensor(weak.global_indices, device=self.device, dtype=torch.int64)
        pre = self.method_config.precorrection
        common = {
            "reference_targets": targets,
            "query_indices": indices,
            "reference_indices": indices,
            "num_classes": self.num_classes,
            "k": int(pre["k_neighbors"]),
            "metric": str(self.method_config.fidelity["neighbor_metric"]),
            "delta": float(pre["delta"]),
            "self_neighbor": str(self.method_config.fidelity["self_neighbor"]),
            "query_chunk_size": pre.get("query_chunk_size"),
        }
        p_w = weighted_neighbor_distribution(feature_w, feature_w, **common).probabilities
        p_s = weighted_neighbor_distribution(feature_s, feature_s, **common).probabilities
        partition = partition_samples(
            p_w, p_s, targets,
            random_state=int(pre["gmm_seed"]),
            minimum_mean_separation=float(pre.get("minimum_mean_separation", 1e-6)),
        )
        y0 = construct_y0(partition.p_ws, targets, partition.partition)
        yn = construct_yn(p_w, p_s, targets, partition.partition)
        yd = yn - y0
        metadata = {
            "dataset": self.dataset,
            "split": "train",
            "manifest_sha256": self.noise_metadata.get("manifest_sha256", ""),
            "mapping_hash": self.noise_metadata.get("mapping_hash", ""),
            "feature_extractor_identity": self.feature_identity,
            "feature_extractor": dict(self.method_config.feature_extractor),
            "feature_source": self.feature_source_provenance,
            "transform_identity": {
                "weak": "cifar_eval_standard",
                "strong": "crop_flip_randaugment_2_10_standard",
                "extraction": "seeded_eval_inference_stable_index",
            },
            "k": int(pre["k_neighbors"]),
            "metric": str(self.method_config.fidelity["neighbor_metric"]),
            "neighbor_weighting": str(
                self.method_config.fidelity["neighbor_weighting"]
            ),
            "self_neighbor": str(self.method_config.fidelity["self_neighbor"]),
            "delta": float(pre["delta"]),
            "divergence": "kl_ps_to_pw_no_softmax",
            "gmm_seed": int(pre["gmm_seed"]),
            "gmm_means": [partition.low_mean, partition.high_mean],
            "fidelity_policy": dict(self.method_config.fidelity),
            "num_classes": self.num_classes,
            "feature_dimension": int(feature_w.shape[1]),
        }
        artifact = DLDPreCorrectionArtifact(
            weak.global_indices, weak.noisy_targets,
            p_w.cpu().numpy(), p_s.cpu().numpy(), partition.p_ws.cpu().numpy(),
            partition.divergence.cpu().numpy(), partition.partition.cpu().numpy(),
            y0.cpu().numpy(), yn.cpu().numpy(), yd.cpu().numpy(),
            ((weak.features + strong.features) / 2), metadata,
        )
        persisted = persist_precorrection_atomically(artifact, self.artifact_path)
        # Only after validated atomic publication may state/checkpoint advance.
        self.artifact = persisted
        self.state.precorrection_artifact_hash = persisted.artifact_hash
        self.state.precorrection_file_sha256 = file_sha256(self.artifact_path)
        self.state.advance(DLDPhase.PRECORRECTION_READY)
        values = persisted.partition
        self._append({
            "event": "precorrection",
            "clean_partition_ratio": float(np.mean(values == PARTITION_CLEAN)),
            "noisy_partition_ratio": float(np.mean(values == PARTITION_NOISY)),
            "hard_partition_ratio": float(np.mean(values == PARTITION_HARD)),
            "divergence_mean": float(persisted.divergence.mean()),
            "divergence_min": float(persisted.divergence.min()),
            "divergence_max": float(persisted.divergence.max()),
            "gmm_low_mean": partition.low_mean,
            "gmm_high_mean": partition.high_mean,
            "y0_entropy": _entropy(persisted.y0),
            "yn_entropy": _entropy(persisted.yn),
            "artifact_hash": persisted.artifact_hash,
        })
        self._save_last()

    def _load_artifact_strict(self) -> DLDPreCorrectionArtifact:
        if not self.artifact_path.is_file():
            raise FileNotFoundError("DLD ready checkpoint is missing dld_precorrection.npz")
        if file_sha256(self.artifact_path) != self.state.precorrection_file_sha256:
            raise ValueError("DLD pre-correction artifact file hash mismatch")
        artifact = DLDPreCorrectionArtifact.load(self.artifact_path)
        if artifact.artifact_hash != self.state.precorrection_artifact_hash:
            raise ValueError("DLD pre-correction artifact content hash mismatch")
        checks = {
            "sample mapping": np.array_equal(artifact.global_indices, self.train_indices),
            "feature identity": artifact.metadata.get("feature_extractor_identity") == self.feature_identity,
            "feature source": (
                artifact.metadata.get("feature_source") == self.feature_source_provenance
                or (
                    artifact.metadata.get("feature_source") is None
                    and self.feature_source is None
                )
            ),
            "manifest": artifact.metadata.get("manifest_sha256") == self.noise_metadata.get("manifest_sha256", ""),
            "mapping": artifact.metadata.get("mapping_hash") == self.noise_metadata.get("mapping_hash", ""),
            "fidelity": artifact.metadata.get("fidelity_policy") == dict(self.method_config.fidelity),
        }
        failed = [name for name, ok in checks.items() if ok is not True]
        if failed:
            raise ValueError("DLD artifact provenance mismatch: " + ", ".join(failed))
        return artifact

    def _prepare_algorithm(self) -> None:
        if self.artifact is None:
            self.artifact = self._load_artifact_strict()
        if self.algorithm is not None:
            return
        feature_dim = int(self.artifact.condition_features.shape[1])
        model_config = self.method_config.diffusion["model"]
        direction = DLDLabelPredictor(
            self.num_classes, feature_dim,
            hidden_dim=int(model_config.get("hidden_dim", 64)),
            time_dim=int(model_config.get("time_dim", 16)),
        )
        noise = DLDLabelPredictor(
            self.num_classes, feature_dim,
            hidden_dim=int(model_config.get("hidden_dim", 64)),
            time_dim=int(model_config.get("time_dim", 16)),
        )
        optimizers = self.method_config.diffusion["optimizer"]
        direction_optimizer = build_optimizer(direction, optimizers["direction"])
        noise_optimizer = build_optimizer(noise, optimizers["noise"])
        schedulers = self.method_config.diffusion["scheduler"]
        direction_scheduler = build_scheduler(direction_optimizer, schedulers["direction"], self.method_config.epochs)
        noise_scheduler = build_scheduler(noise_optimizer, schedulers["noise"], self.method_config.epochs)
        schedule = DirectionalDiffusionSchedule.average(
            int(self.method_config.diffusion["timesteps"]), device=self.device
        )
        ema_config = self.method_config.diffusion.get("ema", {})
        ema_decay = float(ema_config.get("decay", 0.999)) if ema_config.get("enabled", False) else None
        self.algorithm = DLDAlgorithm(
            direction_model=direction, noise_model=noise,
            direction_optimizer=direction_optimizer, noise_optimizer=noise_optimizer,
            direction_scheduler=direction_scheduler, noise_scheduler=noise_scheduler,
            schedule=schedule, artifact=self.artifact, device=self.device,
            ema_decay=ema_decay,
        )
        if self.state.phase is DLDPhase.PRECORRECTION_READY:
            self.state.advance(DLDPhase.DIFFUSION_TRAINING)
            self._save_last()

    def _evaluate_metrics(self, snapshot: FeatureSnapshot) -> dict[str, float]:
        assert self.algorithm is not None
        direction, noise = self.algorithm.prediction_models()
        total = correct = 0
        generated_sum = generated_square_sum = 0.0
        generated_min = float("inf")
        generated_max = float("-inf")
        prediction_counts = np.zeros(self.num_classes, dtype=np.int64)
        batch_size = int(self.loader_config["batch_size"])
        for start in range(0, snapshot.features.shape[0], batch_size):
            stop = min(start + batch_size, snapshot.features.shape[0])
            features = torch.as_tensor(snapshot.features[start:stop], device=self.device, dtype=torch.float32)
            generated = sample_labels(
                direction, noise, features, self.algorithm.schedule,
                inference_steps=int(self.method_config.inference["steps"]),
            )
            predicted = generated.argmax(1).cpu().numpy()
            detached = generated.detach()
            generated_sum += float(detached.sum())
            generated_square_sum += float(detached.square().sum())
            generated_min = min(generated_min, float(detached.min()))
            generated_max = max(generated_max, float(detached.max()))
            prediction_counts += np.bincount(predicted, minlength=self.num_classes)
            correct += int((predicted == snapshot.noisy_targets[start:stop]).sum())
            total += stop - start
        if total <= 0:
            raise ValueError("DLD evaluation snapshot is empty")
        values = prediction_counts[prediction_counts > 0] / total
        entropy = float(-(values * np.log(values)).sum()) if values.size else 0.0
        element_count = total * self.num_classes
        mean = generated_sum / element_count
        variance = max(generated_square_sum / element_count - mean * mean, 0.0)
        result = {
            "accuracy": correct / total,
            "reverse_output_min": generated_min,
            "reverse_output_max": generated_max,
            "reverse_output_mean": mean,
            "reverse_output_std": variance ** 0.5,
            "reverse_prediction_class_count": float(np.count_nonzero(prediction_counts)),
            "reverse_prediction_entropy": entropy,
        }
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("DLD reverse inference telemetry is non-finite")
        return result

    def train(self) -> None:
        self._prepare_algorithm()
        assert self.algorithm is not None
        while self.state.completed_epochs < self.method_config.epochs:
            epoch = self.state.completed_epochs
            loader = _loader(
                _IndexDataset(self.train_indices), self.loader_config,
                shuffle=True, seed=int(self.config.get("seed", 1)) + 5000 + epoch,
            )
            totals = {
                "direction_loss": 0.0,
                "noise_loss": 0.0,
                "direction_gradient_norm": 0.0,
                "noise_gradient_norm": 0.0,
                "direction_parameter_norm": 0.0,
                "noise_parameter_norm": 0.0,
                "predicted_direction_rms": 0.0,
                "predicted_noise_rms": 0.0,
                "target_direction_rms": 0.0,
                "target_noise_rms": 0.0,
            }
            samples = 0.0
            for batch in loader:
                metrics = self.algorithm.train_step(batch["index"].to(self.device))
                count = metrics["samples"]
                for key in totals:
                    totals[key] += metrics[key] * count
                samples += count
                self.state.global_step += 1
            if samples <= 0:
                raise RuntimeError("DLD training epoch contains no samples")
            validation_metrics = self._evaluate_metrics(self.validation_snapshot)
            validation_accuracy = validation_metrics["accuracy"]
            self.state.completed_epochs = epoch + 1
            if validation_accuracy > self.state.best_validation_accuracy:
                self.state.best_epoch = epoch
                self.state.best_validation_accuracy = validation_accuracy
                self.best_algorithm_state = deepcopy(self.algorithm.state_dict())
                atomic_save(self._checkpoint_payload("best"), self.best_path)
            if self.algorithm.direction_scheduler is not None:
                self.algorithm.direction_scheduler.step()
            if self.algorithm.noise_scheduler is not None:
                self.algorithm.noise_scheduler.step()
            self._append({
                "event": "diffusion_epoch", "epoch": epoch,
                **{name: value / samples for name, value in totals.items()},
                "validation_accuracy": validation_accuracy,
                "best_validation_accuracy": self.state.best_validation_accuracy,
                "global_step": self.state.global_step,
                "direction_learning_rate": float(self.algorithm.direction_optimizer.param_groups[0]["lr"]),
                "noise_learning_rate": float(self.algorithm.noise_optimizer.param_groups[0]["lr"]),
                "artifact_hash": self.state.precorrection_artifact_hash,
                "fidelity_policy": self.method_config.fidelity["name"],
                "inference_steps": int(self.method_config.inference["steps"]),
                **{
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                    if name != "accuracy"
                },
            })
            self._save_last()

    def complete(self) -> None:
        if self.best_algorithm_state is None or not self.best_path.is_file():
            raise RuntimeError("DLD cannot complete without a paired best checkpoint")
        assert self.algorithm is not None
        current = deepcopy(self.algorithm.state_dict())
        self.algorithm.load_state_dict(self.best_algorithm_state)
        test_metrics = self._evaluate_metrics(self.test_snapshot)
        test_accuracy = test_metrics["accuracy"]
        self.algorithm.load_state_dict(current)
        self.state.advance(DLDPhase.COMPLETED)
        final = {
            "method": "dld",
            "phase": self.state.phase.value,
            "completed_epochs": self.state.completed_epochs,
            "global_step": self.state.global_step,
            "best_epoch": self.state.best_epoch,
            "best_noisy_validation_accuracy": self.state.best_validation_accuracy,
            "clean_test_accuracy": test_accuracy,
            "precorrection_artifact_hash": self.state.precorrection_artifact_hash,
            "manifest_sha256": self.noise_metadata.get("manifest_sha256", ""),
            "mapping_hash": self.noise_metadata.get("mapping_hash", ""),
            "fidelity_policy": dict(self.method_config.fidelity),
            "inference_steps": int(self.method_config.inference["steps"]),
            "test_selection_leakage": False,
            "paper_numerical_reproduction": False,
            "released_code_exact_reproduction": False,
            **{
                f"test_{name}": value
                for name, value in test_metrics.items()
                if name != "accuracy"
            },
        }
        (self.run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        self._save_last()

    def resume(self, checkpoint: str | Path) -> None:
        payload = read_checkpoint(checkpoint, "cpu")
        if payload.get("method") != "dld" or payload.get("checkpoint_role") != "run_state":
            raise ValueError("only a DLD last.pt checkpoint may be resumed")
        if payload.get("dld_identity_hash") != self.method_config.identity_hash:
            raise ValueError("DLD identity configuration changed on resume")
        if payload.get("feature_identity") != self.feature_identity:
            raise ValueError("DLD feature extractor identity changed on resume")
        saved_feature_source = payload.get("feature_source")
        if (
            saved_feature_source is None
            and self.feature_source is not None
        ) or (
            saved_feature_source is not None
            and dict(saved_feature_source) != self.feature_source_provenance
        ):
            raise ValueError("DLD feature source provenance changed on resume")
        if dict(payload.get("noise", {})) != self.noise_metadata:
            raise ValueError("DLD noise provenance changed on resume")
        self.state = DLDState.from_mapping(payload["dld_state"])
        if self.state.phase is not DLDPhase.FEATURE_EXTRACTION:
            self.artifact = self._load_artifact_strict()
            self._prepare_algorithm()
            assert self.algorithm is not None
            algorithm_state = payload.get("algorithm")
            if not isinstance(algorithm_state, Mapping):
                raise ValueError("DLD checkpoint is missing algorithm state")
            self.algorithm.load_state_dict(algorithm_state)
            _move_optimizer_state(self.algorithm.direction_optimizer, self.device)
            _move_optimizer_state(self.algorithm.noise_optimizer, self.device)
        self.best_algorithm_state = payload.get("best_algorithm_state")
        restore_rng_state(payload["rng_state"])

    def run(self) -> None:
        if self.feature_source is not None:
            self.feature_source.assert_unchanged()
        if self.state.phase is DLDPhase.COMPLETED:
            if self.state.completed_epochs >= self.method_config.epochs:
                return
            self.state.phase = DLDPhase.DIFFUSION_TRAINING
        if self.state.phase is DLDPhase.FEATURE_EXTRACTION:
            self.build_precorrection()
        if self.state.phase in {DLDPhase.PRECORRECTION_READY, DLDPhase.DIFFUSION_TRAINING}:
            self.train()
        if self.state.completed_epochs >= self.method_config.epochs and self.state.phase is DLDPhase.DIFFUSION_TRAINING:
            self.complete()
        if self.feature_source is not None:
            self.feature_source.assert_unchanged()


def run_dld_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    config = deepcopy(config)
    method = DLDConfig.from_mapping(config)
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    trainer = config.get("trainer", {}) or {}
    device = resolve_device(trainer.get("device", "auto"))
    run_dir = (
        Path(resume).resolve().parent if resume is not None
        else Path(output_dir).resolve() if output_dir is not None
        else (Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = read_checkpoint(resume, "cpu") if resume is not None else None
    if checkpoint is not None and checkpoint.get("method") != "dld":
        raise ValueError("resume checkpoint is not a DLD run")
    if checkpoint is not None:
        saved = DLDConfig.from_mapping(checkpoint["config"])
        if method.epochs < saved.epochs:
            raise ValueError("DLD diffusion epoch target cannot be reduced on resume")

    data_config = config["data"]
    dataset = str(data_config.get("name", "cifar10")).lower()
    if dataset not in {"cifar10", "cifar100"}:
        raise ValueError("DLD first version supports CIFAR-10 and CIFAR-100")
    load = load_cifar10 if dataset == "cifar10" else load_cifar100
    classes = 10 if dataset == "cifar10" else 100
    train_data = load(data_config.get("root"), "train")
    test_data = load(data_config.get("root"), "test")
    train_indices, validation_indices = stratified_split(
        train_data.labels, int(data_config["validation_size"]), seed
    )
    manifest_indices = np.sort(np.concatenate((train_indices, validation_indices)))
    manifest, manifest_path = prepare_noise_manifest(
        config, dataset=dataset,
        clean_targets=train_data.labels[manifest_indices],
        global_indices=manifest_indices, num_classes=classes, run_dir=run_dir,
        checkpoint_payload=checkpoint, dataset_targets=train_data.labels,
    )
    if manifest is None or manifest_path is None:
        raise ValueError("DLD requires noisy train and validation labels")
    train_indices = _subset(train_indices, train_data.labels, data_config.get("max_train_samples"), seed + 1)
    validation_indices = _subset(validation_indices, train_data.labels, data_config.get("max_validation_samples"), seed + 2)
    test_indices = _subset(np.arange(len(test_data)), test_data.labels, data_config.get("max_test_samples"), seed + 3)
    if int(method.precorrection["k_neighbors"]) >= len(train_indices):
        raise ValueError("DLD k_neighbors must be smaller than the effective train set")
    loader_config = config["loader"]
    noisy_map = {int(index): int(target) for index, target in zip(manifest.global_indices, manifest.noisy_targets)}
    weak_transform = build_cifar_transform(False)
    strong_transform = build_strong_cifar_transform(magnitude=10)
    multi_view = IndexedMultiViewCifarDataset(
        train_data, train_indices, weak_transform=weak_transform,
        strong_transform=strong_transform, targets_by_index=noisy_map,
    )
    def dual_view_loader(field: str):
        return _loader(
            _ViewDataset(multi_view, field), loader_config,
            shuffle=False, seed=seed + (10 if field == "input" else 11),
        )

    noisy_validation = NoisyTargetDataset(
        TorchCifarDataset(train_data, validation_indices, transform=weak_transform),
        manifest.global_indices, manifest.noisy_targets,
    )
    clean_test = TorchCifarDataset(test_data, test_indices, transform=weak_transform)
    validation_loader = _loader(noisy_validation, loader_config, shuffle=False, seed=seed + 20)
    test_loader = _loader(clean_test, loader_config, shuffle=False, seed=seed + 21)
    noise_metadata = checkpoint_noise_metadata(
        manifest, manifest_path, run_dir,
        effective_subset_actual_rate(manifest, train_indices),
        mode=noise_mode(config), validation_targets="noisy",
        effective_validation_rate=effective_subset_actual_rate(manifest, validation_indices),
    )
    config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)

    seed_everything(seed + 3000)
    feature_model = build_model(method.feature_extractor["model"], classes).to(device).eval()
    feature_source: DLDUPMMainBestSource | None = None
    source_name = str(method.feature_extractor["source"]).strip().lower()
    if source_name == "external_checkpoint":
        feature_source = load_upm_main_best_feature_source(
            method.feature_extractor["external"],
            feature_model,
            num_classes=classes,
        )
        feature_source_provenance = feature_source.provenance
    else:
        feature_source_provenance = {
            "source": "repository_frozen_model",
            "model": dict(method.feature_extractor["model"]),
            "initialization_seed": seed + 3000,
        }
    for parameter in feature_model.parameters():
        parameter.requires_grad_(False)
    feature_identity = _model_identity(feature_model)
    validation_snapshot = _feature_snapshot(feature_model, validation_loader, device, dataset=dataset, split="validation:noisy")
    test_snapshot = _feature_snapshot(feature_model, test_loader, device, dataset=dataset, split="test:clean")

    if resume is None:
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run_dir / "environment.json").write_text(json.dumps(_environment(seed, device), indent=2), encoding="utf-8")
        (run_dir / "noise_summary.json").write_text(json.dumps(noise_metadata, indent=2), encoding="utf-8")
    workflow = DLDWorkflow(
        config=config, method_config=method, run_dir=run_dir, device=device,
        dataset=dataset, num_classes=classes, noise_metadata=noise_metadata,
        feature_model=feature_model, feature_identity=feature_identity,
        feature_source=feature_source,
        feature_source_provenance=feature_source_provenance,
        train_indices=train_indices, dual_view_loader=dual_view_loader,
        validation_snapshot=validation_snapshot, test_snapshot=test_snapshot,
        loader_config=loader_config,
    )
    if resume is not None:
        workflow.resume(resume)
        if method.epochs > DLDConfig.from_mapping(checkpoint["config"]).epochs:
            temporary = (run_dir / "resolved_config.yaml").with_suffix(".yaml.tmp")
            temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            temporary.replace(run_dir / "resolved_config.yaml")
    workflow.run()
    return run_dir


__all__ = ["DLDWorkflow", "run_dld_experiment"]
