from __future__ import annotations

"""Complete first-version Dual-T estimation followed by Forward correction."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.algorithms.transition_risk import ForwardRiskCorrector
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.noise.estimators import (
    DualTransitionEstimator,
    PosteriorSnapshot,
)
from lnl_toolbox.noise.transition import TransitionArtifact
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.snapshots import collect_posterior_snapshot

from .config import DualTConfig
from .state import DualTPhase, DualTState


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_final_risk_corrector(classifier_backend: str) -> ForwardRiskCorrector:
    """Build the current final consumer without coupling it to Dual-T estimation."""

    if classifier_backend == "forward":
        return ForwardRiskCorrector()
    raise NotImplementedError(
        "Dual-T first version only supports classifier backend 'forward'"
    )


def _train_supervised_epoch(
    algorithm: SupervisedClassificationAlgorithm,
    loader: Any,
    state: RunState,
    epoch: int,
) -> dict[str, float]:
    """Run the existing Dual-T supervised epoch over an explicit loader."""

    state.cycle = epoch
    algorithm.on_cycle_start(state)
    loss_sum = 0.0
    all_loss_sum = 0.0
    correct_sum = 0.0
    selected = 0.0
    samples = 0.0
    for raw_batch in loader:
        result = algorithm.step(Batch(raw_batch), state)
        count = result.metrics["samples"]
        selected_count = result.metrics["selected_samples"]
        loss_sum += result.metrics["loss"] * selected_count
        all_loss_sum += result.metrics["all_sample_loss"] * count
        correct_sum += result.metrics["accuracy"] * count
        selected += selected_count
        samples += count
    algorithm.on_cycle_end(state)
    if selected <= 0 or samples <= 0:
        raise RuntimeError("Dual-T training epoch selected no samples")
    return {
        "train_loss": loss_sum / selected,
        "train_all_sample_loss": all_loss_sum / samples,
        "train_accuracy": correct_sum / samples,
        "selected_samples": selected,
        "selected_ratio": selected / samples,
    }


class DualTAlgorithm:
    """Own the paper-specific two-stage Dual-T + Forward lifecycle.

    Both inner trainers use the ordinary supervised batch algorithm. This class
    owns stage ordering, best-posterior selection, artifact provenance, and
    phase-aware checkpoint restoration.
    """

    def __init__(
        self,
        *,
        posterior_model: nn.Module,
        posterior_optimizer: torch.optim.Optimizer,
        posterior_scheduler: Any,
        final_model: nn.Module,
        final_optimizer: torch.optim.Optimizer,
        final_scheduler: Any,
        posterior_loss: nn.Module,
        final_loss: nn.Module,
        train_loader: Any,
        noisy_validation_loader: Any,
        clean_test_loader: Any,
        device: torch.device,
        run_dir: str | Path,
        config: Mapping[str, Any],
        dataset: str,
        noise_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = dict(config)
        self.method_config = DualTConfig.from_mapping(config)
        if posterior_model is final_model:
            raise ValueError("posterior and final models must be distinct objects")
        if posterior_optimizer is final_optimizer:
            raise ValueError(
                "posterior and final optimizers must be distinct objects"
            )
        if (
            posterior_scheduler is not None
            and posterior_scheduler is final_scheduler
        ):
            raise ValueError(
                "posterior and final schedulers must be distinct objects"
            )

        self.device = device
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = str(dataset)
        self.noise_metadata = dict(noise_metadata or {})
        self.train_loader = train_loader
        self.noisy_validation_loader = noisy_validation_loader
        self.clean_test_loader = clean_test_loader
        self.posterior_scheduler = posterior_scheduler
        self.final_scheduler = final_scheduler

        self.posterior_algorithm = SupervisedClassificationAlgorithm(
            posterior_model,
            posterior_optimizer,
            posterior_loss,
            device,
        )
        self.final_algorithm = SupervisedClassificationAlgorithm(
            final_model,
            final_optimizer,
            final_loss,
            device,
            risk_corrector=_build_final_risk_corrector(
                self.method_config.classifier_backend
            ),
            transition=None,
        )
        context = ExperimentContext(self.run_dir, self.config)
        self.posterior_algorithm.setup(context)
        self.final_algorithm.setup(context)
        self.posterior_run_state = RunState(phase="posterior_train")
        self.final_run_state = RunState(phase="final_train")
        self.state = DualTState()
        self.snapshot: PosteriorSnapshot | None = None
        self.transition: TransitionArtifact | None = None

    @property
    def posterior_best_path(self) -> Path:
        return self.run_dir / "posterior_best.pt"

    @property
    def snapshot_path(self) -> Path:
        return self.run_dir / "posterior_snapshot.npz"

    @property
    def transition_path(self) -> Path:
        return self.run_dir / "transition_artifact.npz"

    @property
    def last_path(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "best.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    def _checkpoint_payload(self, *, role: str) -> dict[str, Any]:
        return {
            "format_version": 2,
            "method": "dual_t",
            "checkpoint_role": role,
            "config": self.config,
            "dual_t_state": self.state.state_dict(),
            "posterior_algorithm": self.posterior_algorithm.state_dict(),
            "posterior_scheduler": (
                None
                if self.posterior_scheduler is None
                else self.posterior_scheduler.state_dict()
            ),
            "posterior_run_state": {
                "cycle": self.posterior_run_state.cycle,
                "step": self.posterior_run_state.step,
                "phase": self.posterior_run_state.phase,
            },
            "final_algorithm": self.final_algorithm.state_dict(),
            "final_scheduler": (
                None
                if self.final_scheduler is None
                else self.final_scheduler.state_dict()
            ),
            "final_run_state": {
                "cycle": self.final_run_state.cycle,
                "step": self.final_run_state.step,
                "phase": self.final_run_state.phase,
            },
            "noise": self.noise_metadata,
            "rng_state": capture_rng_state(),
        }

    def _save(self, path: Path, *, role: str) -> None:
        atomic_save(self._checkpoint_payload(role=role), path)

    def _save_last(self) -> None:
        self._save(self.last_path, role="run_state")

    @staticmethod
    def _restore_run_state(
        target: RunState, value: Mapping[str, Any], *, owner: str
    ) -> None:
        if not isinstance(value, Mapping):
            raise TypeError(f"{owner} run state must be a mapping")
        target.cycle = int(value.get("cycle", 0))
        target.step = int(value.get("step", 0))
        target.phase = str(value.get("phase", owner))

    def _validate_resume_config(self, saved: Mapping[str, Any]) -> None:
        saved_method = DualTConfig.from_mapping(saved)
        if saved_method != self.method_config:
            raise ValueError("Resume configuration changed Dual-T stage settings")
        for key in ("seed", "data", "noise", "loader"):
            current_value = self.config.get(key)
            saved_value = saved.get(key)
            if current_value != saved_value:
                raise ValueError(f"Resume configuration changed {key}")

    def resume(self, path: str | Path) -> None:
        checkpoint_path = Path(path).resolve()
        if checkpoint_path.parent != self.run_dir:
            raise ValueError("Dual-T resume checkpoint must belong to the run directory")
        payload = read_checkpoint(checkpoint_path, "cpu")
        if payload.get("method") == "dual_t_forward":
            raise ValueError("method 'dual_t_forward' was renamed to 'dual_t'")
        if payload.get("method") != "dual_t":
            raise ValueError("Checkpoint is not a Dual-T checkpoint")
        if payload.get("checkpoint_role") != "run_state":
            raise ValueError(
                "Only the Dual-T run-state checkpoint may be resumed; "
                "posterior/final best checkpoints are evaluation artifacts"
            )
        saved_config = payload.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("Dual-T checkpoint is missing its configuration")
        self._validate_resume_config(saved_config)

        self.state = DualTState.from_state_dict(payload["dual_t_state"])
        self.posterior_algorithm.load_state_dict(payload["posterior_algorithm"])
        self.final_algorithm.load_state_dict(payload["final_algorithm"])
        self._load_scheduler_state(
            self.posterior_scheduler,
            payload.get("posterior_scheduler"),
            owner="posterior",
        )
        self._load_scheduler_state(
            self.final_scheduler,
            payload.get("final_scheduler"),
            owner="final",
        )
        self._restore_run_state(
            self.posterior_run_state,
            payload.get("posterior_run_state", {}),
            owner="posterior",
        )
        self._restore_run_state(
            self.final_run_state,
            payload.get("final_run_state", {}),
            owner="final",
        )
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
        self._validate_posterior_best(required=self.state.best_posterior_epoch >= 0)
        if self.state.phase in {
            DualTPhase.TRANSITION_READY,
            DualTPhase.FINAL_TRAINING,
            DualTPhase.COMPLETED,
        }:
            self._restore_transition_artifacts()

    @staticmethod
    def _load_scheduler_state(scheduler: Any, state: Any, *, owner: str) -> None:
        if scheduler is None:
            if state is not None:
                raise ValueError(
                    f"Checkpoint contains {owner} scheduler state but it is disabled"
                )
            return
        if state is None:
            raise ValueError(f"Checkpoint is missing {owner} scheduler state")
        scheduler.load_state_dict(state)

    def _validate_posterior_best(self, *, required: bool) -> None:
        if not required:
            return
        if not self.posterior_best_path.is_file():
            raise FileNotFoundError(
                "Dual-T posterior best checkpoint is missing during resume"
            )
        actual = _file_sha256(self.posterior_best_path)
        if actual != self.state.best_posterior_checkpoint_sha256:
            raise ValueError("Dual-T posterior best checkpoint hash mismatch")

    def _restore_transition_artifacts(self) -> None:
        missing = [
            path.name
            for path in (self.snapshot_path, self.transition_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Dual-T artifact missing during resume: " + ", ".join(missing)
            )
        snapshot = PosteriorSnapshot.load(self.snapshot_path)
        transition = TransitionArtifact.load(self.transition_path)
        if snapshot.snapshot_hash != self.state.posterior_snapshot_hash:
            raise ValueError("Dual-T posterior snapshot hash mismatch")
        if transition.artifact_hash != self.state.transition_artifact_hash:
            raise ValueError("Dual-T transition artifact hash mismatch")
        if transition.source_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError(
                "Dual-T transition artifact source snapshot mismatch"
            )
        expected_best = transition.metadata.get(
            "posterior_best_checkpoint_sha256"
        )
        if expected_best != self.state.best_posterior_checkpoint_sha256:
            raise ValueError(
                "Dual-T transition artifact posterior checkpoint mismatch"
            )
        self.snapshot = snapshot
        self.transition = transition
        self.final_algorithm.transition = transition

    def _append_metrics(self, row: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")

    def _train_epoch(
        self,
        algorithm: SupervisedClassificationAlgorithm,
        state: RunState,
        epoch: int,
    ) -> dict[str, float]:
        return _train_supervised_epoch(
            algorithm,
            self.train_loader,
            state,
            epoch,
        )

    def train_posterior(self, *, max_epochs: int | None = None) -> None:
        if self.state.phase is not DualTPhase.POSTERIOR_TRAINING:
            raise ValueError("posterior training is not active")
        target = self.method_config.posterior_stage.epochs
        if max_epochs is not None:
            target = min(target, self.state.posterior_completed_epochs + max_epochs)
        while self.state.posterior_completed_epochs < target:
            epoch = self.state.posterior_completed_epochs
            learning_rate = float(
                self.posterior_algorithm.optimizer.param_groups[0]["lr"]
            )
            train = self._train_epoch(
                self.posterior_algorithm, self.posterior_run_state, epoch
            )
            validation = evaluate_classification(
                self.posterior_algorithm.model,
                self.noisy_validation_loader,
                self.posterior_algorithm.loss,
                self.device,
            )
            if self.posterior_scheduler is not None:
                self.posterior_scheduler.step()
            self.state.posterior_completed_epochs = epoch + 1
            self.state.posterior_global_step = self.posterior_run_state.step
            improved = (
                validation["accuracy"]
                > self.state.best_posterior_validation_accuracy
            )
            if improved:
                self.state.best_posterior_epoch = epoch
                self.state.best_posterior_validation_accuracy = validation[
                    "accuracy"
                ]
                # This evaluation artifact cannot contain its own final file
                # SHA-256. The subsequently saved run-state checkpoint and
                # transition metadata own that external identity.
                self._save(self.posterior_best_path, role="posterior_best")
                self.state.best_posterior_checkpoint_sha256 = _file_sha256(
                    self.posterior_best_path
                )
            row = {
                "event": "epoch",
                "stage": "posterior",
                "epoch": epoch + 1,
                "global_step": self.state.posterior_global_step,
                "learning_rate": learning_rate,
                **train,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "best_epoch": self.state.best_posterior_epoch + 1,
                "best_validation_accuracy": (
                    self.state.best_posterior_validation_accuracy
                ),
            }
            self._append_metrics(row)
            self._save_last()
        if (
            self.state.posterior_completed_epochs
            == self.method_config.posterior_stage.epochs
        ):
            self._validate_posterior_best(required=True)
            self.state.advance(DualTPhase.POSTERIOR_READY)
            self._save_last()

    def estimate_transition(self) -> TransitionArtifact:
        if self.state.phase is not DualTPhase.POSTERIOR_READY:
            raise ValueError("Dual-T transition estimation requires posterior_ready")
        self._validate_posterior_best(required=True)
        payload = read_checkpoint(self.posterior_best_path, "cpu")
        if payload.get("checkpoint_role") != "posterior_best":
            raise ValueError("Dual-T posterior best checkpoint identity mismatch")
        self.posterior_algorithm.load_state_dict(payload["posterior_algorithm"])

        snapshot = collect_posterior_snapshot(
            self.posterior_algorithm.model,
            self.train_loader,
            self.device,
            dataset=self.dataset,
            split="train",
        )
        snapshot.save(self.snapshot_path)
        base_transition = DualTransitionEstimator().estimate(snapshot)
        provenance = dict(base_transition.metadata)
        provenance.update({
            "method": "dual_t",
            "classifier_backend": self.method_config.classifier_backend,
            "posterior_best_epoch": self.state.best_posterior_epoch,
            "posterior_best_validation_accuracy": (
                self.state.best_posterior_validation_accuracy
            ),
            "posterior_best_checkpoint_sha256": (
                self.state.best_posterior_checkpoint_sha256
            ),
            "noise_manifest_sha256": self.noise_metadata.get(
                "manifest_sha256", ""
            ),
            "noise_mapping_hash": self.noise_metadata.get("mapping_hash", ""),
        })
        transition = TransitionArtifact(
            matrix=base_transition.matrix,
            estimator="dual_t",
            source_snapshot_hash=snapshot.snapshot_hash,
            metadata=provenance,
        )
        transition.save(self.transition_path)

        persisted_snapshot = PosteriorSnapshot.load(self.snapshot_path)
        persisted_transition = TransitionArtifact.load(self.transition_path)
        if persisted_snapshot.snapshot_hash != snapshot.snapshot_hash:
            raise ValueError(
                "persisted Dual-T posterior snapshot hash mismatch"
            )
        if persisted_transition.artifact_hash != transition.artifact_hash:
            raise ValueError(
                "persisted Dual-T transition artifact hash mismatch"
            )
        if (
            persisted_transition.source_snapshot_hash
            != persisted_snapshot.snapshot_hash
        ):
            raise ValueError(
                "persisted Dual-T transition source snapshot mismatch"
            )

        self.snapshot = persisted_snapshot
        self.transition = persisted_transition
        self.final_algorithm.transition = persisted_transition
        self.state.posterior_snapshot_hash = persisted_snapshot.snapshot_hash
        self.state.transition_artifact_hash = persisted_transition.artifact_hash
        self.state.advance(DualTPhase.TRANSITION_READY)
        self._append_metrics({
            "event": "transition",
            "stage": "transition",
            "posterior_snapshot_hash": snapshot.snapshot_hash,
            "transition_artifact_hash": transition.artifact_hash,
            "composition": "t_club @ t_spade",
        })
        self._save_last()
        return transition

    def start_final_training(self) -> None:
        if self.state.phase is not DualTPhase.TRANSITION_READY:
            raise ValueError("final training requires transition_ready")
        if self.transition is None:
            self._restore_transition_artifacts()
        self.state.advance(DualTPhase.FINAL_TRAINING)
        self._save_last()

    def train_final(self, *, max_epochs: int | None = None) -> None:
        if self.state.phase is not DualTPhase.FINAL_TRAINING:
            raise ValueError("final training is not active")
        if self.transition is None:
            raise ValueError("final training requires a transition artifact")
        target = self.method_config.final_stage.epochs
        if max_epochs is not None:
            target = min(target, self.state.final_completed_epochs + max_epochs)
        while self.state.final_completed_epochs < target:
            epoch = self.state.final_completed_epochs
            learning_rate = float(
                self.final_algorithm.optimizer.param_groups[0]["lr"]
            )
            train = self._train_epoch(
                self.final_algorithm, self.final_run_state, epoch
            )
            validation = evaluate_classification(
                self.final_algorithm.model,
                self.noisy_validation_loader,
                self.final_algorithm.loss,
                self.device,
            )
            if self.final_scheduler is not None:
                self.final_scheduler.step()
            self.state.final_completed_epochs = epoch + 1
            self.state.final_global_step = self.final_run_state.step
            improved = (
                validation["accuracy"] > self.state.best_final_validation_accuracy
            )
            if improved:
                self.state.best_final_epoch = epoch
                self.state.best_final_validation_accuracy = validation[
                    "accuracy"
                ]
                self._save(self.best_path, role="final_best")
            row = {
                "event": "epoch",
                "stage": "final",
                "epoch": epoch + 1,
                "global_step": self.state.final_global_step,
                "learning_rate": learning_rate,
                **train,
                "validation_observed_ce_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "best_epoch": self.state.best_final_epoch + 1,
                "best_validation_accuracy": (
                    self.state.best_final_validation_accuracy
                ),
            }
            self._append_metrics(row)
            self._save_last()
        if self.state.final_completed_epochs == self.method_config.final_stage.epochs:
            self.complete()

    def complete(self) -> dict[str, Any]:
        if self.state.phase is not DualTPhase.FINAL_TRAINING:
            raise ValueError("Dual-T completion requires final_training")
        if not self.best_path.is_file():
            raise FileNotFoundError("Dual-T final best checkpoint is missing")
        payload = read_checkpoint(self.best_path, "cpu")
        if payload.get("checkpoint_role") != "final_best":
            raise ValueError("Dual-T final best checkpoint identity mismatch")
        final_epoch_state = self.final_algorithm.state_dict()
        self.final_algorithm.load_state_dict(payload["final_algorithm"])
        test = evaluate_classification(
            self.final_algorithm.model,
            self.clean_test_loader,
            self.final_algorithm.loss,
            self.device,
        )
        self.final_algorithm.load_state_dict(final_epoch_state)
        self.state.advance(DualTPhase.COMPLETED)
        final = {
            "event": "final",
            "method": "dual_t",
            "completed_posterior_epochs": (
                self.state.posterior_completed_epochs
            ),
            "posterior_global_step": self.state.posterior_global_step,
            "best_posterior_epoch": self.state.best_posterior_epoch + 1,
            "best_posterior_validation_accuracy": (
                self.state.best_posterior_validation_accuracy
            ),
            "completed_final_epochs": self.state.final_completed_epochs,
            "final_global_step": self.state.final_global_step,
            "best_final_epoch": self.state.best_final_epoch + 1,
            "best_final_validation_accuracy": (
                self.state.best_final_validation_accuracy
            ),
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "posterior_snapshot_hash": self.state.posterior_snapshot_hash,
            "transition_artifact_hash": self.state.transition_artifact_hash,
        }
        self._append_metrics(final)
        (self.run_dir / "final_metrics.json").write_text(
            json.dumps(final, indent=2), encoding="utf-8"
        )
        self._save_last()
        return final

    def run(self) -> dict[str, Any]:
        if self.state.phase is DualTPhase.POSTERIOR_TRAINING:
            self.train_posterior()
        if self.state.phase is DualTPhase.POSTERIOR_READY:
            self.estimate_transition()
        if self.state.phase is DualTPhase.TRANSITION_READY:
            self.start_final_training()
        if self.state.phase is DualTPhase.FINAL_TRAINING:
            self.train_final()
        final_path = self.run_dir / "final_metrics.json"
        if self.state.phase is not DualTPhase.COMPLETED or not final_path.is_file():
            raise RuntimeError("Dual-T run did not reach completion")
        return json.loads(final_path.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.posterior_algorithm.close()
        self.final_algorithm.close()
