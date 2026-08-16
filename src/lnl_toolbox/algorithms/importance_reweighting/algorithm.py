from __future__ import annotations

"""Paper-specific lifecycle for binary asymmetric-RCN importance reweighting."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.noise.estimators import PosteriorSnapshot
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.treatments import (
    BinaryRCNImportanceWeightProvider,
    BinaryRCNWeightInput,
    ReductionSpec,
    SupervisedWeightInput,
    WeightResult,
)

from .artifacts import NoiseRateArtifact
from .config import ImportanceReweightingConfig
from .estimation import (
    PaperRawMinNoiseRateEstimator,
    build_binary_noisy_posterior_backend,
    posterior_backend_identity_hash,
    validate_binary_posterior_snapshot,
)
from .state import ImportanceReweightingPhase, ImportanceReweightingState


class IndexedBinaryRCNWeightProvider:
    """Look up method-owned posterior evidence before applying the exact formula."""

    def __init__(
        self,
        snapshot: PosteriorSnapshot,
        rates: NoiseRateArtifact,
    ) -> None:
        self.snapshot = validate_binary_posterior_snapshot(snapshot)
        if rates.source_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("noise-rate artifact provenance does not match snapshot")
        self.rates = rates
        self._positions = {
            int(index): position
            for position, index in enumerate(snapshot.global_indices.tolist())
        }
        self._provider = BinaryRCNImportanceWeightProvider(
            rho_positive=rates.rho_positive,
            rho_negative=rates.rho_negative,
        )

    def compute(self, weight_input: SupervisedWeightInput) -> WeightResult:
        if not isinstance(weight_input, SupervisedWeightInput):
            raise TypeError("indexed provider requires SupervisedWeightInput")
        logits = weight_input.logits
        targets = weight_input.noisy_targets
        indices = weight_input.sample_indices
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("importance reweighting model logits must have shape [B, 2]")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("importance reweighting targets must have shape [B]")
        if targets.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError("importance reweighting targets must use an integer dtype")
        if bool(((targets != 0) & (targets != 1)).any().item()):
            raise ValueError("importance reweighting targets must contain only 0 and 1")
        if indices.ndim != 1 or indices.shape != targets.shape:
            raise ValueError("importance reweighting sample indices must have shape [B]")
        if indices.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError("importance reweighting sample indices must be integers")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("importance reweighting batch sample indices must be unique")
        if indices.device != targets.device or logits.device != targets.device:
            raise ValueError("importance reweighting batch tensors must share one device")

        positions: list[int] = []
        for index in indices.detach().cpu().tolist():
            if int(index) not in self._positions:
                raise ValueError(
                    f"posterior snapshot is missing stable sample index {int(index)}"
                )
            positions.append(self._positions[int(index)])
        expected_targets = torch.as_tensor(
            self.snapshot.noisy_targets[np.asarray(positions)],
            dtype=targets.dtype,
            device=targets.device,
        )
        if not torch.equal(targets, expected_targets):
            raise ValueError(
                "batch noisy targets do not align with posterior snapshot indices"
            )
        posterior = torch.as_tensor(
            self.snapshot.noisy_probabilities[np.asarray(positions)],
            dtype=torch.float64,
            device=targets.device,
        )
        return self._provider.compute(BinaryRCNWeightInput(
            posterior_probabilities=posterior,
            observed_targets=targets,
            metadata=weight_input.metadata,
        ))


class ImportanceReweightingAlgorithm:
    """Own artifact creation, strict resume, and weighted final training."""

    METHOD = "importance_reweighting"
    FINAL_REDUCTION = ReductionSpec("batch_mean")

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        run_dir: Path,
        manifest_identity: Mapping[str, Any],
        posterior_features: np.ndarray,
        posterior_targets: np.ndarray,
        posterior_indices: np.ndarray,
        model_factory: Callable[[], nn.Module],
        optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
        scheduler_factory: Callable[[torch.optim.Optimizer], Any],
        train_loader_factory: Callable[[int], Any],
        validation_loader: Any,
        test_loader: Any,
        loss: nn.Module,
        device: torch.device,
    ) -> None:
        self.raw_config = deepcopy(dict(config))
        self.config = ImportanceReweightingConfig.from_mapping(config)
        self.run_dir = Path(run_dir)
        self.manifest_identity = dict(manifest_identity)
        if int(self.manifest_identity.get("num_classes", 0)) != 2:
            raise ValueError("importance reweighting manifest identity requires 2 classes")
        self.posterior_features = np.asarray(posterior_features)
        self.posterior_targets = np.asarray(posterior_targets)
        self.posterior_indices = np.asarray(posterior_indices)
        if self.posterior_features.ndim != 2:
            raise ValueError(
                "importance reweighting posterior features must have shape [N, D]"
            )
        self.posterior_backend = build_binary_noisy_posterior_backend(
            self.config.posterior_stage
        )
        self.posterior_backend_identity = dict(
            self.posterior_backend.identity(self.posterior_features.shape[1])
        )
        self.posterior_backend_hash = posterior_backend_identity_hash(
            self.posterior_backend_identity
        )
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.train_loader_factory = train_loader_factory
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.loss = loss
        self.device = device
        self.state = ImportanceReweightingState()
        self.snapshot: PosteriorSnapshot | None = None
        self.rates: NoiseRateArtifact | None = None
        self.final_algorithm: SupervisedClassificationAlgorithm | None = None
        self.scheduler = None
        self.run_state = RunState()

    def _method_diagnostics(self) -> dict[str, Any]:
        """Summarize estimator output without changing any estimator math."""

        if self.snapshot is None or self.rates is None:
            raise ValueError("importance diagnostics require snapshot and rates")
        probabilities = np.asarray(
            self.snapshot.noisy_probabilities, dtype=np.float64
        )
        row_errors = np.abs(probabilities.sum(axis=1) - 1.0)
        targets = torch.as_tensor(
            np.asarray(self.snapshot.noisy_targets).copy(), dtype=torch.long
        )
        result = BinaryRCNImportanceWeightProvider(
            rho_positive=self.rates.rho_positive,
            rho_negative=self.rates.rho_negative,
        ).compute(BinaryRCNWeightInput(
            posterior_probabilities=torch.as_tensor(
                probabilities.copy(), dtype=torch.float64
            ),
            observed_targets=targets,
        ))
        weights = result.sample_weights.detach().cpu().numpy().astype(np.float64)
        weight_sum = float(weights.sum())
        squared_sum = float(np.square(weights).sum())
        ess = 0.0 if squared_sum == 0.0 else weight_sum * weight_sum / squared_sum
        quantiles = np.quantile(weights, [0.5, 0.9, 0.95, 0.99])
        return {
            "posterior": {
                "backend": str(self.config.posterior_stage["name"]),
                "minimum": float(probabilities.min()),
                "mean": float(probabilities.mean()),
                "maximum": float(probabilities.max()),
                "finite_count": int(np.isfinite(probabilities).sum()),
                "value_count": int(probabilities.size),
                "row_sum_max_error": float(row_errors.max()),
                "observed_class_prior": [
                    float(np.mean(self.snapshot.noisy_targets == label))
                    for label in (0, 1)
                ],
            },
            "noise_rates": {
                "rho_positive_hat": float(self.rates.rho_positive),
                "rho_negative_hat": float(self.rates.rho_negative),
                "rho_sum": float(
                    self.rates.rho_positive + self.rates.rho_negative
                ),
                "configured_rho_positive": float(
                    self.config.noise["rho_positive"]
                ),
                "configured_rho_negative": float(
                    self.config.noise["rho_negative"]
                ),
                "realized_rho_positive": float(
                    self.manifest_identity.get("realized_rho_positive", float("nan"))
                ),
                "realized_rho_negative": float(
                    self.manifest_identity.get("realized_rho_negative", float("nan"))
                ),
                "configured_error_positive": float(
                    self.rates.rho_positive
                    - float(self.config.noise["rho_positive"])
                ),
                "configured_error_negative": float(
                    self.rates.rho_negative
                    - float(self.config.noise["rho_negative"])
                ),
            },
            "weights": {
                "minimum": float(weights.min()),
                "mean": float(weights.mean()),
                "p50": float(quantiles[0]),
                "p90": float(quantiles[1]),
                "p95": float(quantiles[2]),
                "p99": float(quantiles[3]),
                "maximum": float(weights.max()),
                "zero_ratio": float(np.mean(weights == 0.0)),
                "negative_count": int(np.sum(weights < 0.0)),
                "nonfinite_count": int(np.sum(~np.isfinite(weights))),
                "ess": float(ess),
                "ess_fraction": float(ess / max(1, weights.size)),
            },
        }

    @staticmethod
    def _parameter_vector(model: nn.Module) -> torch.Tensor:
        return torch.cat([
            parameter.detach().reshape(-1).cpu()
            for parameter in model.parameters()
        ])

    @staticmethod
    def _gradient_norm(model: nn.Module) -> float:
        squared = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().square().sum().item())
        return float(squared ** 0.5)

    @property
    def snapshot_path(self) -> Path:
        return self.run_dir / "posterior_snapshot.npz"

    @property
    def rate_path(self) -> Path:
        return self.run_dir / "noise_rate_artifact.npz"

    @property
    def last_path(self) -> Path:
        return self.run_dir / "last.pt"

    @staticmethod
    def _temporary_artifact_path(destination: Path) -> Path:
        return destination.with_name(
            f".{destination.stem}.{uuid4().hex}.pending{destination.suffix}"
        )

    def _persist_snapshot_atomically(
        self,
        snapshot: PosteriorSnapshot,
    ) -> PosteriorSnapshot:
        """Validate a temporary NPZ before atomically publishing it."""

        temporary = self._temporary_artifact_path(self.snapshot_path)
        try:
            snapshot.save(temporary)
            loaded = validate_binary_posterior_snapshot(
                PosteriorSnapshot.load(temporary)
            )
            if loaded.snapshot_hash != snapshot.snapshot_hash:
                raise ValueError(
                    "persisted posterior snapshot failed hash verification"
                )
            temporary.replace(self.snapshot_path)
            return loaded
        finally:
            if temporary.exists():
                temporary.unlink()

    def _persist_rates_atomically(
        self,
        artifact: NoiseRateArtifact,
        snapshot: PosteriorSnapshot,
    ) -> NoiseRateArtifact:
        """Validate rate identity and provenance before atomic publication."""

        temporary = self._temporary_artifact_path(self.rate_path)
        try:
            artifact.save(temporary)
            loaded = NoiseRateArtifact.load(temporary)
            if loaded.artifact_hash != artifact.artifact_hash:
                raise ValueError(
                    "persisted noise-rate artifact failed hash verification"
                )
            if loaded.source_snapshot_hash != snapshot.snapshot_hash:
                raise ValueError("persisted rate artifact provenance is invalid")
            temporary.replace(self.rate_path)
            return loaded
        finally:
            if temporary.exists():
                temporary.unlink()

    def _fit_posterior(self) -> None:
        snapshot = self.posterior_backend.fit_predict(
            self.posterior_features,
            self.posterior_targets,
            self.posterior_indices,
            dataset=str(self.config.data["name"]),
            split="train",
        )
        loaded = self._persist_snapshot_atomically(snapshot)
        self.snapshot = loaded
        self.state.posterior_backend_hash = self.posterior_backend_hash
        self.state.posterior_snapshot_hash = loaded.snapshot_hash
        self.state.advance(ImportanceReweightingPhase.POSTERIOR_READY)
        self._save_last()

    def _estimate_rates(self) -> None:
        if self.snapshot is None:
            raise ValueError("posterior snapshot is required before rate estimation")
        artifact = PaperRawMinNoiseRateEstimator().estimate(self.snapshot)
        loaded = self._persist_rates_atomically(artifact, self.snapshot)
        self.rates = loaded
        self.state.noise_rate_artifact_hash = loaded.artifact_hash
        self.state.advance(ImportanceReweightingPhase.RATE_READY)
        self._save_last()

    def _build_final(self) -> None:
        if self.snapshot is None or self.rates is None:
            raise ValueError("final training requires posterior and rate artifacts")
        model = self.model_factory()
        optimizer = self.optimizer_factory(model)
        provider = IndexedBinaryRCNWeightProvider(self.snapshot, self.rates)
        algorithm = SupervisedClassificationAlgorithm(
            model=model,
            optimizer=optimizer,
            loss=self.loss,
            device=self.device,
            weight_provider=provider,
        )
        algorithm.reduction = self.FINAL_REDUCTION
        algorithm.setup(ExperimentContext(
            work_dir=self.run_dir,
            config=self.raw_config,
            seed=self.config.seed,
        ))
        self.final_algorithm = algorithm
        self.scheduler = self.scheduler_factory(optimizer)

    def _checkpoint_payload(self, role: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": 1,
            "method": self.METHOD,
            "role": role,
            "num_classes": 2,
            "label_convention": "zero_one",
            "config": self.raw_config,
            "manifest_identity": self.manifest_identity,
            "method_state": self.state.state_dict(),
            "posterior_snapshot_hash": self.state.posterior_snapshot_hash,
            "posterior_backend_identity": self.posterior_backend_identity,
            "posterior_backend_hash": self.posterior_backend_hash,
            "noise_rate_artifact_hash": self.state.noise_rate_artifact_hash,
            "rng_state": capture_rng_state(),
        }
        if self.final_algorithm is not None:
            payload["algorithm"] = self.final_algorithm.state_dict()
            payload["scheduler"] = (
                None if self.scheduler is None else self.scheduler.state_dict()
            )
        return payload

    def _save_last(self) -> None:
        atomic_save(self._checkpoint_payload("last"), self.last_path)

    def _validate_config_for_resume(self, previous: Mapping[str, Any]) -> None:
        before = deepcopy(dict(previous))
        current = deepcopy(self.raw_config)
        before.setdefault("trainer", {})
        current.setdefault("trainer", {})
        before["trainer"] = dict(before["trainer"])
        current["trainer"] = dict(current["trainer"])
        before["trainer"].pop("epochs", None)
        current["trainer"].pop("epochs", None)
        if before != current:
            raise ValueError("importance reweighting resume configuration mismatch")

    def _restore_artifacts(self) -> None:
        if not self.snapshot_path.exists():
            raise FileNotFoundError("resume requires posterior_snapshot.npz")
        snapshot = validate_binary_posterior_snapshot(
            PosteriorSnapshot.load(self.snapshot_path)
        )
        if snapshot.snapshot_hash != self.state.posterior_snapshot_hash:
            raise ValueError("resume posterior snapshot hash mismatch")
        if not np.array_equal(snapshot.global_indices, np.sort(self.posterior_indices)):
            raise ValueError("resume posterior stable indices mismatch")
        ordered = np.argsort(self.posterior_indices, kind="stable")
        if not np.array_equal(snapshot.noisy_targets, self.posterior_targets[ordered]):
            raise ValueError("resume posterior targets are misaligned")
        self.snapshot = snapshot
        if self.state.phase in {
            ImportanceReweightingPhase.RATE_READY,
            ImportanceReweightingPhase.FINAL_TRAINING,
            ImportanceReweightingPhase.COMPLETED,
        }:
            if not self.rate_path.exists():
                raise FileNotFoundError("resume requires noise_rate_artifact.npz")
            rates = NoiseRateArtifact.load(self.rate_path)
            if rates.artifact_hash != self.state.noise_rate_artifact_hash:
                raise ValueError("resume noise-rate artifact hash mismatch")
            if rates.source_snapshot_hash != snapshot.snapshot_hash:
                raise ValueError("resume rate artifact source snapshot hash mismatch")
            self.rates = rates

    def resume(self, path: str | Path) -> None:
        payload = read_checkpoint(path, self.device)
        if payload.get("method") != self.METHOD:
            raise ValueError("resume checkpoint method identity mismatch")
        if int(payload.get("num_classes", 0)) != 2:
            raise ValueError("resume checkpoint num_classes must be 2")
        if payload.get("label_convention") != "zero_one":
            raise ValueError("resume checkpoint label convention must be zero_one")
        if dict(payload.get("manifest_identity", {})) != self.manifest_identity:
            raise ValueError("resume checkpoint manifest identity mismatch")
        if dict(payload.get("posterior_backend_identity", {})) != (
            self.posterior_backend_identity
        ):
            raise ValueError(
                "resume posterior backend identity mismatch"
            )
        if payload.get("posterior_backend_hash") != self.posterior_backend_hash:
            raise ValueError("resume posterior backend hash mismatch")
        self._validate_config_for_resume(payload.get("config", {}))
        self.state = ImportanceReweightingState.from_state_dict(
            payload.get("method_state", {})
        )
        if payload.get("posterior_snapshot_hash", "") != self.state.posterior_snapshot_hash:
            raise ValueError("checkpoint posterior hash fields disagree")
        if payload.get("noise_rate_artifact_hash", "") != self.state.noise_rate_artifact_hash:
            raise ValueError("checkpoint rate artifact hash fields disagree")
        if self.state.posterior_backend_hash != self.posterior_backend_hash:
            raise ValueError("checkpoint method-state backend hash mismatch")
        self._restore_artifacts()
        if self.state.phase in {
            ImportanceReweightingPhase.FINAL_TRAINING,
            ImportanceReweightingPhase.COMPLETED,
        }:
            self._build_final()
            if "algorithm" not in payload:
                raise ValueError("final-training checkpoint is missing algorithm state")
            self.final_algorithm.load_state_dict(dict(payload["algorithm"]))
            if self.scheduler is not None:
                if payload.get("scheduler") is None:
                    raise ValueError("resume checkpoint is missing scheduler state")
                self.scheduler.load_state_dict(payload["scheduler"])
            self.run_state.step = self.state.final_global_step
        restore_rng_state(payload["rng_state"])
        if (
            self.state.phase is ImportanceReweightingPhase.COMPLETED
            and self.config.epochs > self.state.final_completed_epochs
        ):
            self.state.reopen_final_training(self.config.epochs)

    def _train_final(self) -> None:
        if self.final_algorithm is None:
            self._build_final()
        assert self.final_algorithm is not None
        if self.state.phase is ImportanceReweightingPhase.RATE_READY:
            self.state.advance(ImportanceReweightingPhase.FINAL_TRAINING)
            self._save_last()
        metrics_path = self.run_dir / "metrics.jsonl"
        for epoch in range(self.state.final_completed_epochs, self.config.epochs):
            self.run_state.cycle = epoch
            self.final_algorithm.on_cycle_start(self.run_state)
            totals: dict[str, float] = {}
            samples = 0.0
            objective_values: list[float] = []
            gradient_norms: list[float] = []
            update_norms: list[float] = []
            relative_updates: list[float] = []
            for raw_batch in self.train_loader_factory(epoch):
                before = self._parameter_vector(self.final_algorithm.model)
                result = self.final_algorithm.step(Batch(payload=raw_batch), self.run_state)
                after = self._parameter_vector(self.final_algorithm.model)
                update_norm = float(torch.linalg.vector_norm(after - before).item())
                parameter_norm = float(torch.linalg.vector_norm(before).item())
                gradient_norm = self._gradient_norm(self.final_algorithm.model)
                if not np.isfinite(gradient_norm):
                    raise ValueError("importance reweighting gradient norm is non-finite")
                if gradient_norm > float(self.config.diagnostics["max_gradient_norm"]):
                    raise ValueError("importance reweighting gradient explosion detected")
                objective_values.append(float(result.metrics["loss"]))
                gradient_norms.append(gradient_norm)
                update_norms.append(update_norm)
                relative_updates.append(update_norm / max(parameter_norm, 1.0e-12))
                count = float(result.metrics["samples"])
                samples += count
                for name, value in result.metrics.items():
                    if name != "samples":
                        totals[name] = totals.get(name, 0.0) + float(value) * count
            if samples <= 0:
                raise ValueError("importance reweighting training loader is empty")
            train_metrics = {name: value / samples for name, value in totals.items()}
            validation = evaluate_classification(
                self.final_algorithm.model,
                self.validation_loader,
                self.loss,
                self.device,
            )
            learning_rate = float(self.final_algorithm.optimizer.param_groups[0]["lr"])
            if self.scheduler is not None:
                self.scheduler.step()
            self.state.final_completed_epochs = epoch + 1
            self.state.final_global_step = self.run_state.step
            record = {
                "phase": "final_training",
                "epoch": epoch,
                "global_step": self.run_state.step,
                "learning_rate": learning_rate,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                "validation_observed_ce_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "objective_min": float(min(objective_values)),
                "objective_max": float(max(objective_values)),
                "gradient_norm_mean": float(np.mean(gradient_norms)),
                "gradient_norm_max": float(max(gradient_norms)),
                "parameter_norm": float(torch.linalg.vector_norm(
                    self._parameter_vector(self.final_algorithm.model)
                ).item()),
                "update_norm_mean": float(np.mean(update_norms)),
                "relative_update_mean": float(np.mean(relative_updates)),
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if validation["accuracy"] > self.state.best_final_validation_accuracy:
                self.state.best_final_validation_accuracy = validation["accuracy"]
                self.state.best_final_epoch = epoch
                atomic_save(self._checkpoint_payload("best"), self.run_dir / "best.pt")
            self._save_last()

        best = read_checkpoint(self.run_dir / "best.pt", self.device)
        best_model = self.model_factory().to(self.device)
        best_model.load_state_dict(dict(best["algorithm"])["model"])
        test = evaluate_classification(
            best_model, self.test_loader, self.loss, self.device
        )
        self.state.advance(ImportanceReweightingPhase.COMPLETED)
        final = {
            "method": self.METHOD,
            "completed_epochs": self.state.final_completed_epochs,
            "global_step": self.state.final_global_step,
            "best_epoch": self.state.best_final_epoch,
            "best_validation_accuracy": self.state.best_final_validation_accuracy,
            "test_accuracy": test["accuracy"],
            "test_loss": test["loss"],
            "posterior_snapshot_hash": self.state.posterior_snapshot_hash,
            "noise_rate_artifact_hash": self.state.noise_rate_artifact_hash,
            "rho_positive_hat": self.rates.rho_positive,
            "rho_negative_hat": self.rates.rho_negative,
            "reduction": "batch_mean",
            **self._method_diagnostics(),
        }
        records = [
            json.loads(line)
            for line in (self.run_dir / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        final["optimization"] = {
            "objective_min": float(min(row["objective_min"] for row in records)),
            "objective_max": float(max(row["objective_max"] for row in records)),
            "gradient_norm_max": float(
                max(row["gradient_norm_max"] for row in records)
            ),
            "parameter_norm_final": float(records[-1]["parameter_norm"]),
            "relative_update_mean": float(np.mean([
                row["relative_update_mean"] for row in records
            ])),
            "optimizer_stall": bool(all(
                row["update_norm_mean"] == 0.0 for row in records
            )),
            "nonfinite_count": 0,
        }
        (self.run_dir / "final_metrics.json").write_text(
            json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._save_last()

    def run(self) -> Path:
        if self.state.phase is ImportanceReweightingPhase.POSTERIOR_FITTING:
            self._fit_posterior()
        if self.state.phase is ImportanceReweightingPhase.POSTERIOR_READY:
            self._estimate_rates()
        if self.state.phase in {
            ImportanceReweightingPhase.RATE_READY,
            ImportanceReweightingPhase.FINAL_TRAINING,
        }:
            self._train_final()
        return self.run_dir
