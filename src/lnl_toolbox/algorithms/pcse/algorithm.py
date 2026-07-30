from __future__ import annotations

"""First complete PCSE workflow with a replaceable transition backend.

The first phase uses the existing Dual-T estimator as an engineering backend.
It is not the minimum-volume estimator in PCSE paper Eq. (1).  The backend
boundary intentionally leaves room for a future ``paper_volmin`` producer.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from lnl_toolbox.algorithms.dual_t.algorithm import _train_supervised_epoch
from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.core import ExperimentContext, RunState
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.noise.estimators import (
    DualTransitionEstimator,
    PosteriorSnapshot,
)
from lnl_toolbox.noise.transition import (
    TransitionArtifact,
    validate_transition_matrix,
)
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.snapshots import collect_posterior_snapshot

from .artifacts import (
    PCSEEnsembleArtifact,
    PCSEGDAArtifact,
    PCSEStatisticsArtifact,
    persist_npz_atomically,
)
from .config import PCSEConfig
from .features import PCSEFeatureCollection, collect_pcse_features
from .gda import GDAEnsemble, fit_ensemble_weights, fit_gda_layers
from .state import PCSEPhase, PCSEState
from .statistics import estimate_pcse_statistics
from .volmin import (
    DiagonallyDominantTransition,
    build_volmin_optimizer,
    validate_trainable_transition,
    volmin_objective,
)


class _PCSETransitionBackend(Protocol):
    """Method-local lifecycle identity for transition production."""

    name: str
    config: Mapping[str, Any]
    requires_training: bool
    requires_snapshot: bool

    def estimate(self, snapshot: PosteriorSnapshot) -> TransitionArtifact:
        ...


class _DualTTransitionBackend:
    """First-phase engineering backend; not PCSE paper Eq. (1)."""

    name = "dual_t"
    config: Mapping[str, Any] = {}
    requires_training = False
    requires_snapshot = True

    def estimate(self, snapshot: PosteriorSnapshot) -> TransitionArtifact:
        return DualTransitionEstimator().estimate(snapshot)


class _PaperVolMinTransitionBackend:
    """Joint Eq. (1) backend with method-local resumable training."""

    name = "paper_volmin"
    requires_training = True
    requires_snapshot = False

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def estimate(self, snapshot: PosteriorSnapshot) -> TransitionArtifact:
        raise RuntimeError("paper_volmin requires its joint training lifecycle")


def _build_transition_backend(config: PCSEConfig) -> _PCSETransitionBackend:
    if config.transition_backend == "dual_t":
        return _DualTTransitionBackend()
    if config.transition_backend == "paper_volmin":
        return _PaperVolMinTransitionBackend(
            config.transition_backend_config
        )
    raise NotImplementedError(
        f"PCSE transition backend is unavailable: {config.transition_backend}"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_mapping_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _persist_snapshot(snapshot: PosteriorSnapshot, destination: Path) -> PosteriorSnapshot:
    temporary = destination.with_name(destination.name + ".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"PCSE temporary artifact already exists: {temporary}")
    try:
        snapshot.save(temporary)
        loaded = PosteriorSnapshot.load(temporary)
        if loaded.snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("PCSE posterior snapshot temporary hash mismatch")
        temporary.replace(destination)
        return loaded
    finally:
        if temporary.exists():
            temporary.unlink()


def _persist_transition(
    artifact: TransitionArtifact, destination: Path
) -> TransitionArtifact:
    temporary = destination.with_name(destination.name + ".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"PCSE temporary artifact already exists: {temporary}")
    try:
        artifact.save(temporary)
        loaded = TransitionArtifact.load(temporary)
        if loaded.artifact_hash != artifact.artifact_hash:
            raise ValueError("PCSE transition temporary hash mismatch")
        temporary.replace(destination)
        return loaded
    finally:
        if temporary.exists():
            temporary.unlink()


class PCSEAlgorithm:
    """Own noisy pretraining, statistic recovery, GDA and ensemble lifecycle."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss: nn.Module,
        train_loader: Any,
        statistics_loader: Any,
        noisy_validation_loader: Any,
        clean_test_loader: Any,
        device: torch.device,
        run_dir: str | Path,
        config: Mapping[str, Any],
        dataset: str,
        num_classes: int,
        noise_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = dict(config)
        self.method_config = PCSEConfig.from_mapping(config)
        self.transition_backend = _build_transition_backend(
            self.method_config
        )
        if num_classes < 3:
            raise ValueError("PCSE first version requires num_classes >= 3")
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = str(dataset)
        self.noise_metadata = dict(noise_metadata or {})
        self.train_loader = train_loader
        self.statistics_loader = statistics_loader
        self.noisy_validation_loader = noisy_validation_loader
        self.clean_test_loader = clean_test_loader
        self.scheduler = scheduler
        self.pretraining_algorithm = SupervisedClassificationAlgorithm(
            model, optimizer, loss, self.device
        )
        self.pretraining_algorithm.setup(
            ExperimentContext(self.run_dir, self.config)
        )
        self.pretraining_run_state = RunState(phase="pcse_pretraining")
        self.state = PCSEState()
        self.transition_model: DiagonallyDominantTransition | None = None
        self.transition_optimizer: torch.optim.Optimizer | None = None
        self.snapshot: PosteriorSnapshot | None = None
        self.transition: TransitionArtifact | None = None
        self.statistics_artifact: PCSEStatisticsArtifact | None = None
        self.gda_artifact: PCSEGDAArtifact | None = None
        self.ensemble_raw_weights: Tensor | None = None
        self.ensemble_optimizer: torch.optim.Optimizer | None = None
        self.ensemble_artifact: PCSEEnsembleArtifact | None = None

    @property
    def pretrained_best_path(self) -> Path:
        return self.run_dir / "pretrained_best.pt"

    @property
    def snapshot_path(self) -> Path:
        return self.run_dir / "posterior_snapshot.npz"

    @property
    def transition_path(self) -> Path:
        return self.run_dir / "transition_artifact.npz"

    @property
    def volmin_final_path(self) -> Path:
        return self.run_dir / "volmin_final.pt"

    @property
    def statistics_path(self) -> Path:
        return self.run_dir / "pcse_statistics.npz"

    @property
    def gda_path(self) -> Path:
        return self.run_dir / "pcse_gda.npz"

    @property
    def ensemble_path(self) -> Path:
        return self.run_dir / "pcse_ensemble.npz"

    @property
    def last_path(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    def _checkpoint_payload(self, *, role: str) -> dict[str, Any]:
        return {
            "format_version": 2,
            "method": "pcse",
            "checkpoint_role": role,
            "config": self.config,
            "pcse_state": self.state.state_dict(),
            "pretraining_algorithm": self.pretraining_algorithm.state_dict(),
            "pretraining_scheduler": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
            "pretraining_run_state": {
                "cycle": self.pretraining_run_state.cycle,
                "step": self.pretraining_run_state.step,
                "phase": self.pretraining_run_state.phase,
            },
            "transition_model": (
                None
                if self.transition_model is None
                else self.transition_model.state_dict()
            ),
            "transition_optimizer": (
                None
                if self.transition_optimizer is None
                else self.transition_optimizer.state_dict()
            ),
            "transition_backend": self.transition_backend.name,
            "transition_backend_config": dict(self.transition_backend.config),
            "ensemble_raw_weights": (
                None
                if self.ensemble_raw_weights is None
                else self.ensemble_raw_weights.detach().cpu()
            ),
            "ensemble_optimizer": (
                None
                if self.ensemble_optimizer is None
                else self.ensemble_optimizer.state_dict()
            ),
            "noise": self.noise_metadata,
            "rng_state": capture_rng_state(),
        }

    def _save(self, path: Path, *, role: str) -> None:
        atomic_save(self._checkpoint_payload(role=role), path)

    def _save_last(self) -> None:
        self._save(self.last_path, role="run_state")

    def _append_metrics(self, row: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")

    def _validate_best_checkpoint(self, *, required: bool) -> None:
        if not required:
            return
        if not self.pretrained_best_path.is_file():
            raise FileNotFoundError("PCSE pretrained best checkpoint is missing")
        if (
            _file_sha256(self.pretrained_best_path)
            != self.state.pretrained_checkpoint_sha256
        ):
            raise ValueError("PCSE pretrained best checkpoint hash mismatch")

    def train_pretraining(self, *, max_epochs: int | None = None) -> None:
        if self.state.phase is not PCSEPhase.PRETRAINING:
            raise ValueError("PCSE pretraining is not active")
        target = self.method_config.pretraining.epochs
        if max_epochs is not None:
            target = min(target, self.state.pretraining_completed_epochs + max_epochs)
        while self.state.pretraining_completed_epochs < target:
            epoch = self.state.pretraining_completed_epochs
            train = _train_supervised_epoch(
                self.pretraining_algorithm,
                self.train_loader,
                self.pretraining_run_state,
                epoch,
            )
            validation = evaluate_classification(
                self.pretraining_algorithm.model,
                self.noisy_validation_loader,
                self.pretraining_algorithm.loss,
                self.device,
            )
            learning_rate = float(
                self.pretraining_algorithm.optimizer.param_groups[0]["lr"]
            )
            if self.scheduler is not None:
                self.scheduler.step()
            self.state.pretraining_completed_epochs = epoch + 1
            self.state.pretraining_global_step = self.pretraining_run_state.step
            is_better_accuracy = (
                validation["accuracy"]
                > self.state.best_pretraining_validation_accuracy
            )
            is_better_tie = (
                validation["accuracy"]
                == self.state.best_pretraining_validation_accuracy
                and validation["loss"]
                < self.state.best_pretraining_validation_loss
            )
            if is_better_accuracy or is_better_tie:
                self.state.best_pretraining_epoch = epoch
                self.state.best_pretraining_validation_accuracy = validation[
                    "accuracy"
                ]
                self.state.best_pretraining_validation_loss = validation["loss"]
                self._save(self.pretrained_best_path, role="pretrained_best")
                self.state.pretrained_checkpoint_sha256 = _file_sha256(
                    self.pretrained_best_path
                )
            self._append_metrics({
                "event": "epoch",
                "stage": "pretraining",
                "epoch": epoch + 1,
                "global_step": self.state.pretraining_global_step,
                "learning_rate": learning_rate,
                **train,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "best_epoch": self.state.best_pretraining_epoch + 1,
                "best_validation_accuracy": (
                    self.state.best_pretraining_validation_accuracy
                ),
                "best_validation_loss": (
                    self.state.best_pretraining_validation_loss
                ),
            })
            self._save_last()
        if self.state.pretraining_completed_epochs == target == (
            self.method_config.pretraining.epochs
        ):
            self._validate_best_checkpoint(required=True)
            self.state.advance(PCSEPhase.PRETRAINED)
            self._save_last()

    def _load_best_model(self) -> None:
        self._validate_best_checkpoint(required=True)
        payload = read_checkpoint(self.pretrained_best_path, "cpu")
        if payload.get("checkpoint_role") != "pretrained_best":
            raise ValueError("PCSE pretrained checkpoint role mismatch")
        self.pretraining_algorithm.load_state_dict(
            payload["pretraining_algorithm"]
        )

    def _build_transition_runtime(self) -> None:
        if self.transition_backend.name != "paper_volmin":
            raise ValueError("transition runtime is only used by paper_volmin")
        values = self.transition_backend.config
        parameterization = values["parameterization"]
        self.transition_model = DiagonallyDominantTransition(
            self.num_classes,
            initial_flip_mass=parameterization["initial_flip_mass"],
            max_flip_mass=parameterization["max_flip_mass"],
            temperature=parameterization["temperature"],
            seed=parameterization["seed"],
        ).to(self.device)
        self.transition_optimizer = build_volmin_optimizer(
            self.pretraining_algorithm.model,
            self.transition_model,
            values["optimizer"],
        )

    def start_transition_training(self) -> None:
        if self.state.phase is not PCSEPhase.PRETRAINED:
            raise ValueError(
                "PCSE transition training requires pretrained phase"
            )
        if not self.transition_backend.requires_training:
            raise ValueError("selected PCSE transition backend is stateless")
        self._load_best_model()
        self._build_transition_runtime()
        self.state.advance(PCSEPhase.TRANSITION_TRAINING)
        self._save_last()

    def _volmin_final_payload(
        self, diagnostics: Mapping[str, float]
    ) -> dict[str, Any]:
        return {
            "format_version": 1,
            "method": "pcse",
            "artifact_role": "volmin_final_model",
            "model": self.pretraining_algorithm.model.state_dict(),
            "transition_model": self.transition_model.state_dict(),
            "source_pretrained_checkpoint_sha256": (
                self.state.pretrained_checkpoint_sha256
            ),
            "transition_backend": self.transition_backend.name,
            "transition_backend_config": dict(self.transition_backend.config),
            "transition_backend_config_hash": _stable_mapping_hash(
                self.transition_backend.config
            ),
            "completed_epochs": self.state.transition_completed_epochs,
            "global_step": self.state.transition_global_step,
            "optimizer_config": dict(
                self.transition_backend.config["optimizer"]
            ),
            "scheduler_config": dict(
                self.transition_backend.config["scheduler"]
            ),
            "diagnostics": dict(diagnostics),
        }

    def _validate_volmin_final_payload(
        self, payload: Mapping[str, Any]
    ) -> None:
        if payload.get("method") != "pcse" or payload.get(
            "artifact_role"
        ) != "volmin_final_model":
            raise ValueError("PCSE VolMin final checkpoint role mismatch")
        if payload.get("source_pretrained_checkpoint_sha256") != (
            self.state.pretrained_checkpoint_sha256
        ):
            raise ValueError("PCSE VolMin source checkpoint mismatch")
        if payload.get("transition_backend") != "paper_volmin":
            raise ValueError("PCSE VolMin backend identity mismatch")
        if payload.get("transition_backend_config") != dict(
            self.transition_backend.config
        ):
            raise ValueError("PCSE VolMin backend configuration mismatch")
        if payload.get("transition_backend_config_hash") != (
            _stable_mapping_hash(self.transition_backend.config)
        ):
            raise ValueError("PCSE VolMin backend configuration hash mismatch")
        if int(payload.get("completed_epochs", -1)) != (
            self.state.transition_completed_epochs
        ) or int(payload.get("global_step", -1)) != (
            self.state.transition_global_step
        ):
            raise ValueError("PCSE VolMin progress provenance mismatch")
        if not isinstance(payload.get("model"), Mapping) or not isinstance(
            payload.get("transition_model"), Mapping
        ):
            raise ValueError("PCSE VolMin final model state is missing")

    def _publish_volmin_transition(
        self, diagnostics: Mapping[str, float]
    ) -> TransitionArtifact:
        matrix = self.transition_model.matrix().detach().cpu().numpy()
        validate_transition_matrix(matrix, self.num_classes)
        final_payload = self._volmin_final_payload(diagnostics)
        final_temporary = self.volmin_final_path.with_name(
            self.volmin_final_path.name + ".tmp"
        )
        transition_temporary = self.transition_path.with_name(
            self.transition_path.name + ".tmp.npz"
        )
        if final_temporary.exists() or transition_temporary.exists():
            raise FileExistsError("PCSE VolMin temporary artifact already exists")
        try:
            torch.save(final_payload, final_temporary)
            loaded_final = read_checkpoint(final_temporary, "cpu")
            self._validate_volmin_final_payload(loaded_final)
            final_hash = _file_sha256(final_temporary)
            provenance = {
                "method": "pcse",
                "transition_backend": "paper_volmin",
                "transition_backend_note": (
                    "paper Eq. (1) objective with a strictly diagonally "
                    "dominant numerical-safety parameterization; this is "
                    "narrower than the paper's general nonsingular "
                    "row-stochastic assumption"
                ),
                "transition_backend_config": dict(
                    self.transition_backend.config
                ),
                "transition_backend_config_hash": _stable_mapping_hash(
                    self.transition_backend.config
                ),
                "source_pretrained_checkpoint_sha256": (
                    self.state.pretrained_checkpoint_sha256
                ),
                "feature_model_checkpoint_sha256": final_hash,
                "noise_manifest_sha256": self.noise_metadata.get(
                    "manifest_sha256", ""
                ),
                "noise_mapping_hash": self.noise_metadata.get(
                    "mapping_hash", ""
                ),
                "dataset": self.dataset,
                "lambda_volume": self.transition_backend.config[
                    "lambda_volume"
                ],
                "epochs": self.state.transition_completed_epochs,
                "global_step": self.state.transition_global_step,
                "optimizer": dict(
                    self.transition_backend.config["optimizer"]
                ),
                "parameterization": dict(
                    self.transition_backend.config["parameterization"]
                ),
                **dict(diagnostics),
            }
            artifact = TransitionArtifact(
                matrix=matrix,
                estimator="paper_volmin",
                source_snapshot_hash="",
                metadata=provenance,
            )
            artifact.save(transition_temporary)
            loaded_transition = TransitionArtifact.load(
                transition_temporary
            )
            if loaded_transition.artifact_hash != artifact.artifact_hash:
                raise ValueError("PCSE VolMin transition hash mismatch")
            if loaded_transition.metadata.get(
                "feature_model_checkpoint_sha256"
            ) != final_hash:
                raise ValueError(
                    "PCSE VolMin transition model provenance mismatch"
                )
            final_temporary.replace(self.volmin_final_path)
            transition_temporary.replace(self.transition_path)
            self.state.volmin_final_checkpoint_sha256 = final_hash
            self.state.feature_model_checkpoint_sha256 = final_hash
            self.state.transition_artifact_hash = (
                loaded_transition.artifact_hash
            )
            self.transition = loaded_transition
            self.state.advance(PCSEPhase.TRANSITION_READY)
            self._append_metrics({
                "event": "transition",
                "stage": "transition",
                "backend": "paper_volmin",
                "feature_model_checkpoint_sha256": final_hash,
                "transition_artifact_hash": loaded_transition.artifact_hash,
                **dict(diagnostics),
            })
            self._save_last()
            return loaded_transition
        finally:
            if final_temporary.exists():
                final_temporary.unlink()
            if transition_temporary.exists():
                transition_temporary.unlink()

    def train_transition(
        self, *, max_epochs: int | None = None
    ) -> TransitionArtifact | None:
        if self.state.phase is not PCSEPhase.TRANSITION_TRAINING:
            raise ValueError("PCSE transition training is not active")
        if self.transition_model is None or self.transition_optimizer is None:
            raise ValueError("PCSE transition-training state is incomplete")
        values = self.transition_backend.config
        target = int(values["epochs"])
        if max_epochs is not None:
            target = min(
                target,
                self.state.transition_completed_epochs + max_epochs,
            )
        latest_metrics: Mapping[str, float] | None = None
        model = self.pretraining_algorithm.model
        while self.state.transition_completed_epochs < target:
            model.train()
            samples = 0
            totals: dict[str, float] = {}
            for batch in self.train_loader:
                if not isinstance(batch, Mapping):
                    raise TypeError(
                        "PCSE transition loader must yield mapping batches"
                    )
                inputs = batch["input"].to(self.device, non_blocking=True)
                noisy_targets = batch["target"].to(
                    self.device, non_blocking=True
                )
                logits = model(inputs).to(dtype=torch.float64)
                transition = self.transition_model.matrix()
                objective, batch_metrics = volmin_objective(
                    logits,
                    noisy_targets,
                    transition,
                    lambda_volume=values["lambda_volume"],
                    determinant_tolerance=values[
                        "determinant_tolerance"
                    ],
                    condition_limit=values["condition_limit"],
                )
                self.transition_optimizer.zero_grad(set_to_none=True)
                objective.backward()
                self.transition_optimizer.step()
                count = int(noisy_targets.numel())
                samples += count
                for name, metric in batch_metrics.items():
                    totals[name] = totals.get(name, 0.0) + metric * count
                self.state.transition_global_step += 1
            if samples == 0:
                raise ValueError("PCSE transition loader must not be empty")
            latest_metrics = {
                name: value / samples for name, value in totals.items()
            }
            self.state.transition_completed_epochs += 1
            self._append_metrics({
                "event": "epoch",
                "stage": "transition",
                "epoch": self.state.transition_completed_epochs,
                "global_step": self.state.transition_global_step,
                **latest_metrics,
            })
            self._save_last()
        if self.state.transition_completed_epochs == int(values["epochs"]):
            transition = self.transition_model.matrix()
            _, diagnostics = validate_trainable_transition(
                transition,
                determinant_tolerance=values["determinant_tolerance"],
                condition_limit=values["condition_limit"],
            )
            return self._publish_volmin_transition({
                "determinant": diagnostics.determinant,
                "log_determinant": diagnostics.log_determinant,
                "minimum_singular_value": (
                    diagnostics.minimum_singular_value
                ),
                "condition_number": diagnostics.condition_number,
            })
        return None

    def estimate_transition(self) -> TransitionArtifact:
        if self.state.phase is not PCSEPhase.PRETRAINED:
            raise ValueError("PCSE transition estimation requires pretrained phase")
        if self.transition_backend.requires_training:
            self.start_transition_training()
            result = self.train_transition()
            if result is None:
                raise RuntimeError("PCSE VolMin transition did not complete")
            return result
        self._load_best_model()
        snapshot = collect_posterior_snapshot(
            self.pretraining_algorithm.model,
            self.statistics_loader,
            self.device,
            dataset=self.dataset,
            split="train",
        )
        if snapshot.num_classes != self.num_classes:
            raise ValueError("PCSE posterior class count mismatch")
        persisted_snapshot = _persist_snapshot(snapshot, self.snapshot_path)
        base = self.transition_backend.estimate(persisted_snapshot)
        provenance = dict(base.metadata)
        provenance.update({
            "method": "pcse",
            "transition_backend": self.transition_backend.name,
            "transition_backend_note": (
                "engineering backend; not PCSE paper Eq. (1) minimum-volume"
            ),
            "transition_backend_config": dict(
                self.transition_backend.config
            ),
            "pretrained_checkpoint_sha256": (
                self.state.pretrained_checkpoint_sha256
            ),
            "noise_manifest_sha256": self.noise_metadata.get(
                "manifest_sha256", ""
            ),
            "noise_mapping_hash": self.noise_metadata.get("mapping_hash", ""),
            "dataset": self.dataset,
        })
        transition = TransitionArtifact(
            matrix=base.matrix,
            estimator="dual_t",
            source_snapshot_hash=persisted_snapshot.snapshot_hash,
            metadata=provenance,
        )
        persisted_transition = _persist_transition(
            transition, self.transition_path
        )
        if (
            persisted_transition.source_snapshot_hash
            != persisted_snapshot.snapshot_hash
        ):
            raise ValueError("PCSE transition provenance mismatch")
        self.snapshot = persisted_snapshot
        self.transition = persisted_transition
        self.state.posterior_snapshot_hash = persisted_snapshot.snapshot_hash
        self.state.feature_model_checkpoint_sha256 = (
            self.state.pretrained_checkpoint_sha256
        )
        self.state.transition_artifact_hash = persisted_transition.artifact_hash
        self.state.advance(PCSEPhase.TRANSITION_READY)
        self._append_metrics({
            "event": "transition",
            "stage": "transition",
            "backend": self.transition_backend.name,
            "posterior_snapshot_hash": persisted_snapshot.snapshot_hash,
            "transition_artifact_hash": persisted_transition.artifact_hash,
        })
        self._save_last()
        return persisted_transition

    def _collect_features(self, loader: Any, *, split: str) -> PCSEFeatureCollection:
        return collect_pcse_features(
            self.pretraining_algorithm.model,
            loader,
            self.device,
            dataset=self.dataset,
            split=split,
            layers=self.method_config.feature_layers,
        )

    def _load_feature_model(self) -> None:
        if self.transition_backend.name == "dual_t":
            self._load_best_model()
            expected = self.state.pretrained_checkpoint_sha256
        else:
            if not self.volmin_final_path.is_file():
                raise FileNotFoundError(
                    "PCSE VolMin final model checkpoint is missing"
                )
            actual = _file_sha256(self.volmin_final_path)
            if actual != self.state.volmin_final_checkpoint_sha256:
                raise ValueError(
                    "PCSE VolMin final model checkpoint hash mismatch"
                )
            payload = read_checkpoint(self.volmin_final_path, "cpu")
            self._validate_volmin_final_payload(payload)
            self.pretraining_algorithm.model.load_state_dict(payload["model"])
            expected = actual
        if expected != self.state.feature_model_checkpoint_sha256:
            raise ValueError("PCSE feature-model checkpoint identity mismatch")

    def estimate_statistics(self) -> PCSEStatisticsArtifact:
        if self.state.phase is not PCSEPhase.TRANSITION_READY:
            raise ValueError("PCSE statistics require transition_ready")
        if self.transition is None:
            self._restore_transition_artifacts()
        self._load_feature_model()
        features = self._collect_features(self.statistics_loader, split="train")
        statistics = estimate_pcse_statistics(
            features.snapshots,
            features.layer_names,
            self.transition.matrix,
            condition_limit=self.method_config.condition_limit,
        )
        artifact = PCSEStatisticsArtifact(
            statistics,
            provenance={
                "method": "pcse",
                "dataset": self.dataset,
                "transition_artifact_hash": self.transition.artifact_hash,
                "posterior_snapshot_hash": (
                    self.state.posterior_snapshot_hash
                ),
                "pretrained_checkpoint_sha256": (
                    self.state.pretrained_checkpoint_sha256
                ),
                "feature_model_checkpoint_sha256": (
                    self.state.feature_model_checkpoint_sha256
                ),
                "feature_layers": [
                    {"name": item.name, "pooling": item.pooling}
                    for item in self.method_config.feature_layers
                ],
                "noise_manifest_sha256": self.noise_metadata.get(
                    "manifest_sha256", ""
                ),
            },
        )
        loaded = persist_npz_atomically(
            artifact, self.statistics_path, PCSEStatisticsArtifact.load
        )
        self.statistics_artifact = loaded
        self.state.statistics_artifact_hash = loaded.artifact_hash
        self.state.advance(PCSEPhase.STATISTICS_READY)
        self._append_metrics({
            "event": "statistics",
            "stage": "statistics",
            "statistics_artifact_hash": loaded.artifact_hash,
            "transition_condition": statistics.transition_condition,
            "coefficient_condition": statistics.coefficient_condition,
            "clean_priors": statistics.clean_priors.tolist(),
        })
        self._save_last()
        return loaded

    def build_gda(self) -> PCSEGDAArtifact:
        if self.state.phase is not PCSEPhase.STATISTICS_READY:
            raise ValueError("PCSE GDA requires statistics_ready")
        if self.statistics_artifact is None:
            self._restore_statistics_artifact()
        layers = fit_gda_layers(
            self.statistics_artifact.statistics,
            covariance_ridge=self.method_config.covariance_ridge,
        )
        artifact = PCSEGDAArtifact(
            layers,
            provenance={
                "method": "pcse",
                "statistics_artifact_hash": (
                    self.statistics_artifact.artifact_hash
                ),
                "covariance_ridge": self.method_config.covariance_ridge,
                "regularization_note": (
                    "explicit implementation choice; recovered paper covariance "
                    "is retained in the statistics artifact"
                ),
                "layer_names": [layer.name for layer in layers],
            },
        )
        loaded = persist_npz_atomically(
            artifact, self.gda_path, PCSEGDAArtifact.load
        )
        self.gda_artifact = loaded
        self.state.gda_artifact_hash = loaded.artifact_hash
        self.state.advance(PCSEPhase.GDA_READY)
        self._append_metrics({
            "event": "gda",
            "stage": "gda",
            "gda_artifact_hash": loaded.artifact_hash,
            "covariance_ridge": self.method_config.covariance_ridge,
        })
        self._save_last()
        return loaded

    def _layer_probabilities(
        self, loader: Any, *, split: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.gda_artifact is None:
            raise ValueError("PCSE GDA artifact is required")
        features = self._collect_features(loader, split=split)
        if tuple(layer.name for layer in self.gda_artifact.layers) != (
            features.layer_names
        ):
            raise ValueError("PCSE GDA layer order mismatch")
        probabilities = np.stack([
            gda.posterior(snapshot.features)
            for gda, snapshot in zip(
                self.gda_artifact.layers, features.snapshots
            )
        ])
        reference = features.snapshots[0]
        return probabilities, reference.noisy_targets, reference.global_indices

    def start_ensemble_training(self) -> None:
        if self.state.phase is not PCSEPhase.GDA_READY:
            raise ValueError("PCSE ensemble requires gda_ready")
        if self.gda_artifact is None:
            self._restore_gda_artifact()
        self.ensemble_raw_weights = torch.zeros(
            len(self.gda_artifact.layers),
            dtype=torch.float64,
            requires_grad=True,
        )
        self.ensemble_optimizer = torch.optim.Adam(
            [self.ensemble_raw_weights],
            lr=self.method_config.ensemble.learning_rate,
        )
        self.state.advance(PCSEPhase.ENSEMBLE_TRAINING)
        self._save_last()

    def train_ensemble(self, *, max_epochs: int | None = None) -> None:
        if self.state.phase is not PCSEPhase.ENSEMBLE_TRAINING:
            raise ValueError("PCSE ensemble training is not active")
        if (
            self.gda_artifact is None
            or self.ensemble_raw_weights is None
            or self.ensemble_optimizer is None
        ):
            raise ValueError("PCSE ensemble state is incomplete")
        probabilities, targets, _ = self._layer_probabilities(
            self.noisy_validation_loader, split="validation"
        )
        target = self.method_config.ensemble.epochs
        if max_epochs is not None:
            target = min(target, self.state.ensemble_completed_epochs + max_epochs)
        while self.state.ensemble_completed_epochs < target:
            raw, optimizer, losses = fit_ensemble_weights(
                probabilities,
                targets,
                epochs=1,
                learning_rate=self.method_config.ensemble.learning_rate,
                raw_weights=self.ensemble_raw_weights,
                optimizer_state=self.ensemble_optimizer.state_dict(),
            )
            self.ensemble_raw_weights = raw
            self.ensemble_optimizer = optimizer
            self.state.ensemble_completed_epochs += 1
            self.state.ensemble_global_step += 1
            weights = torch.softmax(raw.detach(), dim=0)
            self._append_metrics({
                "event": "epoch",
                "stage": "ensemble",
                "epoch": self.state.ensemble_completed_epochs,
                "global_step": self.state.ensemble_global_step,
                "validation_nll": losses[-1],
                "ensemble_weights": weights.tolist(),
            })
            self._save_last()
        if self.state.ensemble_completed_epochs == (
            self.method_config.ensemble.epochs
        ):
            self.complete()

    def complete(self) -> dict[str, Any]:
        if self.state.phase is not PCSEPhase.ENSEMBLE_TRAINING:
            raise ValueError("PCSE completion requires ensemble_training")
        weights = torch.softmax(
            self.ensemble_raw_weights.detach(), dim=0
        ).cpu().numpy()
        ensemble = PCSEEnsembleArtifact(
            layer_names=tuple(layer.name for layer in self.gda_artifact.layers),
            weights=weights,
            provenance={
                "method": "pcse",
                "gda_artifact_hash": self.gda_artifact.artifact_hash,
                "validation_split": "noisy_validation",
                "ensemble_epochs": self.method_config.ensemble.epochs,
                "ensemble_learning_rate": (
                    self.method_config.ensemble.learning_rate
                ),
            },
        )
        loaded = persist_npz_atomically(
            ensemble, self.ensemble_path, PCSEEnsembleArtifact.load
        )
        probabilities, clean_targets, _ = self._layer_probabilities(
            self.clean_test_loader, split="test"
        )
        posterior = np.einsum("l,lnc->nc", loaded.weights, probabilities)
        if not np.isfinite(posterior).all() or not np.allclose(
            posterior.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8
        ):
            raise ValueError("PCSE final posterior is invalid")
        predictions = posterior.argmax(axis=1)
        test_accuracy = float(np.mean(predictions == clean_targets))
        self.ensemble_artifact = loaded
        self.state.ensemble_artifact_hash = loaded.artifact_hash
        self.state.advance(PCSEPhase.COMPLETED)
        final = {
            "event": "final",
            "method": "pcse",
            "transition_backend": self.method_config.transition_backend,
            "completed_pretraining_epochs": (
                self.state.pretraining_completed_epochs
            ),
            "pretraining_global_step": self.state.pretraining_global_step,
            "completed_ensemble_epochs": self.state.ensemble_completed_epochs,
            "ensemble_global_step": self.state.ensemble_global_step,
            "best_pretraining_epoch": self.state.best_pretraining_epoch + 1,
            "best_pretraining_validation_accuracy": (
                self.state.best_pretraining_validation_accuracy
            ),
            "best_pretraining_validation_loss": (
                self.state.best_pretraining_validation_loss
            ),
            "posterior_snapshot_hash": self.state.posterior_snapshot_hash,
            "transition_artifact_hash": self.state.transition_artifact_hash,
            "statistics_artifact_hash": self.state.statistics_artifact_hash,
            "gda_artifact_hash": self.state.gda_artifact_hash,
            "ensemble_artifact_hash": self.state.ensemble_artifact_hash,
            "ensemble_weights": loaded.weights.tolist(),
            "test_accuracy": test_accuracy,
            "test_samples": int(clean_targets.size),
        }
        (self.run_dir / "final_metrics.json").write_text(
            json.dumps(final, indent=2), encoding="utf-8"
        )
        self._append_metrics(final)
        self._save_last()
        return final

    def _restore_transition_artifacts(self) -> None:
        if not self.transition_path.is_file():
            raise FileNotFoundError("PCSE transition-stage artifact is missing")
        transition = TransitionArtifact.load(self.transition_path)
        if transition.artifact_hash != self.state.transition_artifact_hash:
            raise ValueError("PCSE transition artifact hash mismatch")
        if transition.metadata.get("transition_backend") != (
            self.method_config.transition_backend
        ):
            raise ValueError("PCSE transition backend provenance mismatch")
        if transition.metadata.get("dataset") != self.dataset:
            raise ValueError("PCSE transition dataset provenance mismatch")
        if transition.metadata.get("transition_backend_config") != dict(
            self.method_config.transition_backend_config
        ):
            raise ValueError("PCSE transition backend-config provenance mismatch")
        if transition.metadata.get("noise_manifest_sha256") != (
            self.noise_metadata.get("manifest_sha256", "")
        ):
            raise ValueError("PCSE transition manifest provenance mismatch")
        if transition.metadata.get("noise_mapping_hash") != (
            self.noise_metadata.get("mapping_hash", "")
        ):
            raise ValueError("PCSE transition mapping provenance mismatch")
        if self.transition_backend.name == "dual_t":
            if not self.snapshot_path.is_file():
                raise FileNotFoundError(
                    "PCSE Dual-T posterior snapshot is missing"
                )
            snapshot = PosteriorSnapshot.load(self.snapshot_path)
            if snapshot.snapshot_hash != self.state.posterior_snapshot_hash:
                raise ValueError("PCSE posterior snapshot hash mismatch")
            if transition.source_snapshot_hash != snapshot.snapshot_hash:
                raise ValueError("PCSE transition source snapshot mismatch")
            if snapshot.dataset != self.dataset or snapshot.split != "train":
                raise ValueError(
                    "PCSE posterior snapshot dataset provenance mismatch"
                )
            if transition.metadata.get(
                "pretrained_checkpoint_sha256"
            ) != self.state.pretrained_checkpoint_sha256:
                raise ValueError(
                    "PCSE transition checkpoint provenance mismatch"
                )
            if self.state.feature_model_checkpoint_sha256 != (
                self.state.pretrained_checkpoint_sha256
            ):
                raise ValueError(
                    "PCSE Dual-T feature-model provenance mismatch"
                )
            self.snapshot = snapshot
        else:
            if transition.source_snapshot_hash:
                raise ValueError(
                    "PCSE VolMin transition must not claim a posterior source"
                )
            if transition.metadata.get(
                "source_pretrained_checkpoint_sha256"
            ) != self.state.pretrained_checkpoint_sha256:
                raise ValueError("PCSE VolMin source checkpoint mismatch")
            if transition.metadata.get(
                "feature_model_checkpoint_sha256"
            ) != self.state.volmin_final_checkpoint_sha256:
                raise ValueError("PCSE VolMin feature-model provenance mismatch")
            self._load_feature_model()
            self.snapshot = None
        self.transition = transition

    def _restore_statistics_artifact(self) -> None:
        if not self.statistics_path.is_file():
            raise FileNotFoundError("PCSE statistics artifact is missing")
        artifact = PCSEStatisticsArtifact.load(self.statistics_path)
        if artifact.artifact_hash != self.state.statistics_artifact_hash:
            raise ValueError("PCSE statistics artifact hash mismatch")
        if artifact.provenance.get("transition_artifact_hash") != (
            self.state.transition_artifact_hash
        ):
            raise ValueError("PCSE statistics provenance mismatch")
        expected_layers = [
            {"name": item.name, "pooling": item.pooling}
            for item in self.method_config.feature_layers
        ]
        if artifact.provenance.get("feature_layers") != expected_layers:
            raise ValueError("PCSE statistics feature provenance mismatch")
        if artifact.provenance.get("posterior_snapshot_hash") != (
            self.state.posterior_snapshot_hash
        ):
            raise ValueError("PCSE statistics snapshot provenance mismatch")
        if artifact.provenance.get("pretrained_checkpoint_sha256") != (
            self.state.pretrained_checkpoint_sha256
        ):
            raise ValueError("PCSE statistics checkpoint provenance mismatch")
        if artifact.provenance.get(
            "feature_model_checkpoint_sha256"
        ) != self.state.feature_model_checkpoint_sha256:
            raise ValueError(
                "PCSE statistics feature-model provenance mismatch"
            )
        self.statistics_artifact = artifact

    def _restore_gda_artifact(self) -> None:
        if not self.gda_path.is_file():
            raise FileNotFoundError("PCSE GDA artifact is missing")
        artifact = PCSEGDAArtifact.load(self.gda_path)
        if artifact.artifact_hash != self.state.gda_artifact_hash:
            raise ValueError("PCSE GDA artifact hash mismatch")
        if artifact.provenance.get("statistics_artifact_hash") != (
            self.state.statistics_artifact_hash
        ):
            raise ValueError("PCSE GDA provenance mismatch")
        if artifact.provenance.get("covariance_ridge") != (
            self.method_config.covariance_ridge
        ):
            raise ValueError("PCSE GDA ridge provenance mismatch")
        if artifact.provenance.get("layer_names") != [
            item.name for item in self.method_config.feature_layers
        ]:
            raise ValueError("PCSE GDA layer provenance mismatch")
        self.gda_artifact = artifact

    def resume(self, path: str | Path) -> None:
        checkpoint_path = Path(path).resolve()
        if checkpoint_path.parent != self.run_dir:
            raise ValueError("PCSE resume checkpoint must belong to run directory")
        payload = read_checkpoint(checkpoint_path, "cpu")
        if payload.get("method") != "pcse":
            raise ValueError("Checkpoint is not a PCSE checkpoint")
        if payload.get("checkpoint_role") != "run_state":
            raise ValueError("Only PCSE last.pt may be resumed")
        saved_config = payload.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("PCSE checkpoint configuration is missing")
        if PCSEConfig.from_mapping(saved_config) != self.method_config:
            raise ValueError("Resume configuration changed PCSE method settings")
        for key in ("seed", "data", "noise", "loader"):
            if saved_config.get(key) != self.config.get(key):
                raise ValueError(f"Resume configuration changed {key}")
        self.state = PCSEState.from_state_dict(payload["pcse_state"])
        self.pretraining_algorithm.load_state_dict(
            payload["pretraining_algorithm"]
        )
        scheduler_state = payload.get("pretraining_scheduler")
        if self.scheduler is None and scheduler_state is not None:
            raise ValueError("PCSE checkpoint contains unexpected scheduler state")
        if self.scheduler is not None:
            if scheduler_state is None:
                raise ValueError("PCSE checkpoint is missing scheduler state")
            self.scheduler.load_state_dict(scheduler_state)
        run_state = payload.get("pretraining_run_state", {})
        self.pretraining_run_state.cycle = int(run_state.get("cycle", 0))
        self.pretraining_run_state.step = int(run_state.get("step", 0))
        self.pretraining_run_state.phase = str(
            run_state.get("phase", "pcse_pretraining")
        )
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
        if payload.get("transition_backend") != self.transition_backend.name:
            raise ValueError("PCSE checkpoint transition backend mismatch")
        if payload.get("transition_backend_config") != dict(
            self.transition_backend.config
        ):
            raise ValueError(
                "PCSE checkpoint transition backend configuration mismatch"
            )
        self._validate_best_checkpoint(
            required=self.state.phase is not PCSEPhase.PRETRAINING
        )
        if self.state.phase is PCSEPhase.TRANSITION_TRAINING:
            if self.transition_backend.name != "paper_volmin":
                raise ValueError(
                    "Only paper_volmin may resume transition training"
                )
            transition_model_state = payload.get("transition_model")
            transition_optimizer_state = payload.get("transition_optimizer")
            if not isinstance(
                transition_model_state, Mapping
            ) or not isinstance(transition_optimizer_state, Mapping):
                raise ValueError(
                    "PCSE VolMin checkpoint transition state is missing"
                )
            self._build_transition_runtime()
            self.transition_model.load_state_dict(transition_model_state)
            self.transition_optimizer.load_state_dict(
                transition_optimizer_state
            )
        order = list(PCSEPhase)
        phase_index = order.index(self.state.phase)
        if phase_index >= order.index(PCSEPhase.TRANSITION_READY):
            self._restore_transition_artifacts()
        if phase_index >= order.index(PCSEPhase.STATISTICS_READY):
            self._restore_statistics_artifact()
        if phase_index >= order.index(PCSEPhase.GDA_READY):
            self._restore_gda_artifact()
        if phase_index >= order.index(PCSEPhase.ENSEMBLE_TRAINING):
            raw = payload.get("ensemble_raw_weights")
            optimizer_state = payload.get("ensemble_optimizer")
            if not torch.is_tensor(raw) or optimizer_state is None:
                raise ValueError("PCSE checkpoint ensemble state is missing")
            self.ensemble_raw_weights = raw.to(
                dtype=torch.float64
            ).detach().requires_grad_(True)
            self.ensemble_optimizer = torch.optim.Adam(
                [self.ensemble_raw_weights],
                lr=self.method_config.ensemble.learning_rate,
            )
            self.ensemble_optimizer.load_state_dict(optimizer_state)
        if self.state.phase is PCSEPhase.COMPLETED:
            if not self.ensemble_path.is_file():
                raise FileNotFoundError("PCSE ensemble artifact is missing")
            ensemble = PCSEEnsembleArtifact.load(self.ensemble_path)
            if ensemble.artifact_hash != self.state.ensemble_artifact_hash:
                raise ValueError("PCSE ensemble artifact hash mismatch")
            if ensemble.provenance.get("gda_artifact_hash") != (
                self.state.gda_artifact_hash
            ):
                raise ValueError("PCSE ensemble provenance mismatch")
            if (
                ensemble.provenance.get("ensemble_epochs")
                != self.method_config.ensemble.epochs
                or ensemble.provenance.get("ensemble_learning_rate")
                != self.method_config.ensemble.learning_rate
                or ensemble.provenance.get("validation_split")
                != "noisy_validation"
            ):
                raise ValueError("PCSE ensemble configuration provenance mismatch")
            self.ensemble_artifact = ensemble

    def run(self) -> dict[str, Any]:
        if self.state.phase is PCSEPhase.PRETRAINING:
            self.train_pretraining()
        if self.state.phase is PCSEPhase.PRETRAINED:
            self.estimate_transition()
        if self.state.phase is PCSEPhase.TRANSITION_TRAINING:
            self.train_transition()
        if self.state.phase is PCSEPhase.TRANSITION_READY:
            self.estimate_statistics()
        if self.state.phase is PCSEPhase.STATISTICS_READY:
            self.build_gda()
        if self.state.phase is PCSEPhase.GDA_READY:
            self.start_ensemble_training()
        if self.state.phase is PCSEPhase.ENSEMBLE_TRAINING:
            self.train_ensemble()
        final_path = self.run_dir / "final_metrics.json"
        if self.state.phase is not PCSEPhase.COMPLETED or not final_path.is_file():
            raise RuntimeError("PCSE run did not reach completion")
        return json.loads(final_path.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.pretraining_algorithm.close()
