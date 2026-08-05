from __future__ import annotations

"""Three-stage T-Revision Reweight-R lifecycle."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.algorithms.dual_t.algorithm import _train_supervised_epoch
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.noise.estimators import AnchorTransitionEstimator, PosteriorSnapshot
from lnl_toolbox.noise.transition import TransitionArtifact
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.snapshots import collect_posterior_snapshot

from .artifacts import RevisedTransitionArtifact
from .config import TRevisionConfig
from .objective import TRevisionObjectiveResult, t_revision_reweight_objective
from .state import TRevisionPhase, TRevisionState
from .transition import AdditiveTransitionRevision, validate_revision_optimizer


OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer]
RevisionOptimizerFactory = Callable[
    [nn.Module, AdditiveTransitionRevision], torch.optim.Optimizer
]
SchedulerFactory = Callable[[torch.optim.Optimizer], Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _atomic_npz(
    path: Path,
    value: PosteriorSnapshot | TransitionArtifact | RevisedTransitionArtifact,
    loader: Callable[[Path], Any],
    expected_hash: str,
) -> Any:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    try:
        value.save(temporary)
        persisted = loader(temporary)
        actual_hash = getattr(persisted, "snapshot_hash", None)
        if actual_hash is None:
            actual_hash = persisted.artifact_hash
        if actual_hash != expected_hash:
            raise ValueError(f"temporary {path.name} hash mismatch")
        temporary.replace(path)
        return persisted
    finally:
        if temporary.exists():
            temporary.unlink()


class TRevisionAlgorithm:
    """Own Reweight-R stage ordering, provenance, and strict resume state."""

    def __init__(
        self,
        *,
        model: nn.Module,
        stage1_optimizer: torch.optim.Optimizer,
        stage1_scheduler: Any,
        classifier_optimizer_factory: OptimizerFactory,
        classifier_scheduler_factory: SchedulerFactory,
        revision_optimizer_factory: RevisionOptimizerFactory,
        revision_scheduler_factory: SchedulerFactory,
        loss: nn.Module,
        train_loader: Any,
        posterior_loader: Any,
        noisy_validation_loader: Any,
        clean_test_loader: Any,
        device: torch.device,
        run_dir: str | Path,
        config: Mapping[str, Any],
        dataset: str,
        num_classes: int,
        noise_metadata: Mapping[str, Any],
        diagnostic_transition: np.ndarray | None = None,
    ) -> None:
        self.config = dict(config)
        self.method_config = TRevisionConfig.from_mapping(config)
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.device = device
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = str(dataset)
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("T-Revision requires at least two classes")
        self.noise_metadata = dict(noise_metadata)
        self.diagnostic_transition = (
            None
            if diagnostic_transition is None
            else np.asarray(diagnostic_transition, dtype=np.float64).copy()
        )
        if self.diagnostic_transition is not None and self.diagnostic_transition.shape != (
            self.num_classes,
            self.num_classes,
        ):
            raise ValueError("diagnostic transition has the wrong class shape")
        self.train_loader = train_loader
        self.posterior_loader = posterior_loader
        self.noisy_validation_loader = noisy_validation_loader
        self.clean_test_loader = clean_test_loader
        self.classifier_optimizer_factory = classifier_optimizer_factory
        self.classifier_scheduler_factory = classifier_scheduler_factory
        self.revision_optimizer_factory = revision_optimizer_factory
        self.revision_scheduler_factory = revision_scheduler_factory
        self.optimizer = stage1_optimizer
        self.scheduler = stage1_scheduler
        self.run_state = RunState(phase="t_revision_stage1")
        self.stage1_algorithm = SupervisedClassificationAlgorithm(
            self.model, self.optimizer, self.loss, self.device
        )
        self.stage1_algorithm.setup(ExperimentContext(self.run_dir, self.config))
        self.state = TRevisionState()
        self.snapshot: PosteriorSnapshot | None = None
        self.initial_transition: TransitionArtifact | None = None
        self.revision: AdditiveTransitionRevision | None = None
        self.revision_best_model_state: dict[str, Tensor] | None = None
        self.revision_best_delta_state: Tensor | None = None

    @property
    def stage1_best_path(self) -> Path:
        return self.run_dir / "stage1_best.pt"

    @property
    def snapshot_path(self) -> Path:
        return self.run_dir / "posterior_snapshot.npz"

    @property
    def initial_transition_path(self) -> Path:
        return self.run_dir / "transition_initial.npz"

    @property
    def stage2a_best_path(self) -> Path:
        return self.run_dir / "stage2a_best.pt"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "best.pt"

    @property
    def revised_transition_path(self) -> Path:
        return self.run_dir / "transition_revised.npz"

    @property
    def last_path(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    def _current_stage_name(self) -> str:
        if self.state.phase in {
            TRevisionPhase.STAGE1_TRAINING,
            TRevisionPhase.STAGE1_READY,
            TRevisionPhase.TRANSITION_INITIALIZED,
        }:
            return "stage1"
        if self.state.phase in {
            TRevisionPhase.CLASSIFIER_INITIALIZATION,
            TRevisionPhase.CLASSIFIER_READY,
        }:
            return "classifier_initialization"
        return "revision"

    def _checkpoint_payload(self, *, role: str) -> dict[str, Any]:
        return {
            "format_version": 2,
            "method": "t_revision",
            "objective": "reweight",
            "transition_mode": "paper_experiment_raw_additive",
            "ratio_policy": {
                "detach": False,
                "clamp": "none",
                "denominator_floor": self.method_config.denominator_floor,
            },
            "checkpoint_role": role,
            "current_stage": self._current_stage_name(),
            "config": self.config,
            "t_revision_state": self.state.state_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "run_state": {
                "cycle": self.run_state.cycle,
                "step": self.run_state.step,
                "phase": self.run_state.phase,
            },
            "initial_transition": (
                None
                if self.initial_transition is None
                else self.initial_transition.matrix.copy()
            ),
            "pseudo_anchor_indices": (
                None
                if self.initial_transition is None
                else self.initial_transition.metadata.get("anchor_global_indices")
            ),
            "delta": None if self.revision is None else self.revision.delta.detach().cpu(),
            "revision_best_model_state": self.revision_best_model_state,
            "revision_best_delta_state": self.revision_best_delta_state,
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

    def _seed_train_loader(self, stage_offset: int, epoch: int) -> None:
        generator = getattr(self.train_loader, "generator", None)
        if generator is not None:
            generator.manual_seed(
                int(self.config.get("seed", 1)) + stage_offset + int(epoch)
            )

    def _validate_config(self, saved: Mapping[str, Any]) -> int:
        saved_method = TRevisionConfig.from_mapping(saved)
        saved_revision_epochs = saved_method.revision.epochs
        current_revision_epochs = self.method_config.revision.epochs
        if current_revision_epochs < saved_revision_epochs:
            raise ValueError(
                "Resume cannot reduce t_revision.revision.epochs below the "
                "checkpoint target"
            )
        comparable_saved = replace(
            saved_method,
            revision=replace(
                saved_method.revision, epochs=current_revision_epochs
            ),
        )
        if comparable_saved != self.method_config:
            raise ValueError("Resume configuration changed T-Revision settings")
        if current_revision_epochs > saved_revision_epochs:
            scheduler = self.method_config.revision.scheduler
            if (
                str(scheduler.get("name", "none")).strip().lower() == "cosine"
                and "t_max" not in scheduler
            ):
                raise ValueError(
                    "Extending T-Revision revision epochs with cosine scheduler "
                    "requires an explicit revision.scheduler.t_max"
                )
        for name in ("seed", "data", "noise", "loader", "trainer"):
            if saved.get(name) != self.config.get(name):
                raise ValueError(f"Resume configuration changed {name}")
        return saved_revision_epochs

    def _validate_best(self, path: Path, expected: str, *, owner: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"T-Revision {owner} checkpoint is missing")
        if _file_sha256(path) != expected:
            raise ValueError(f"T-Revision {owner} checkpoint hash mismatch")

    def _restore_initial_artifacts(self) -> None:
        if not self.snapshot_path.is_file() or not self.initial_transition_path.is_file():
            raise FileNotFoundError("T-Revision initial transition artifacts are missing")
        snapshot = PosteriorSnapshot.load(self.snapshot_path)
        transition = TransitionArtifact.load(self.initial_transition_path)
        if snapshot.snapshot_hash != self.state.snapshot_hash:
            raise ValueError("T-Revision posterior snapshot hash mismatch")
        if transition.artifact_hash != self.state.initial_transition_hash:
            raise ValueError("T-Revision initial transition artifact hash mismatch")
        if transition.source_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("T-Revision initial transition provenance mismatch")
        if transition.metadata.get("stage1_best_checkpoint_sha256") != self.state.stage1_best_hash:
            raise ValueError("T-Revision initial transition checkpoint provenance mismatch")
        if transition.num_classes != self.num_classes:
            raise ValueError("T-Revision transition class count mismatch")
        self.snapshot = snapshot
        self.initial_transition = transition

    def _set_optimizer_for_phase(self, phase: TRevisionPhase) -> None:
        if phase in {
            TRevisionPhase.STAGE1_TRAINING,
            TRevisionPhase.STAGE1_READY,
            TRevisionPhase.TRANSITION_INITIALIZED,
        }:
            return
        if phase in {
            TRevisionPhase.CLASSIFIER_INITIALIZATION,
            TRevisionPhase.CLASSIFIER_READY,
        }:
            self.optimizer = self.classifier_optimizer_factory(self.model)
            self.scheduler = self.classifier_scheduler_factory(self.optimizer)
            return
        if self.initial_transition is None:
            raise ValueError("revision phase requires initial transition")
        initial = self.initial_transition.as_tensor(
            device=self.device, dtype=next(self.model.parameters()).dtype
        )
        self.revision = AdditiveTransitionRevision(initial).to(self.device)
        self.optimizer = self.revision_optimizer_factory(self.model, self.revision)
        validate_revision_optimizer(self.optimizer, self.model.parameters(), self.revision)
        self.scheduler = self.revision_scheduler_factory(self.optimizer)

    def resume(self, path: str | Path) -> None:
        checkpoint = Path(path).resolve()
        if checkpoint.parent != self.run_dir:
            raise ValueError("T-Revision resume checkpoint must belong to run directory")
        payload = read_checkpoint(checkpoint, "cpu")
        if payload.get("method") != "t_revision" or payload.get("checkpoint_role") != "run_state":
            raise ValueError("Only a T-Revision last.pt checkpoint may be resumed")
        if payload.get("objective") != "reweight":
            raise ValueError("T-Revision resume objective mismatch")
        if payload.get("transition_mode") != "paper_experiment_raw_additive":
            raise ValueError("T-Revision resume transition mode mismatch")
        saved_config = payload.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("T-Revision checkpoint is missing config")
        saved_revision_epochs = self._validate_config(saved_config)
        self.state = TRevisionState.from_state_dict(payload["t_revision_state"])
        extend_completed_revision = (
            self.state.phase is TRevisionPhase.COMPLETED
            and self.method_config.revision.epochs > saved_revision_epochs
        )
        if extend_completed_revision and (
            self.state.revision_completed_epochs != saved_revision_epochs
        ):
            raise ValueError(
                "Completed T-Revision checkpoint progress does not match its "
                "revision epoch target"
            )
        self._validate_best(
            self.stage1_best_path,
            self.state.stage1_best_hash,
            owner="stage1 best",
        ) if self.state.stage1_best_epoch >= 0 else None
        if self.state.phase not in {
            TRevisionPhase.STAGE1_TRAINING,
            TRevisionPhase.STAGE1_READY,
        }:
            self._restore_initial_artifacts()
        if self.state.phase in {
            TRevisionPhase.CLASSIFIER_READY,
            TRevisionPhase.REVISION_TRAINING,
            TRevisionPhase.COMPLETED,
        }:
            self._validate_best(
                self.stage2a_best_path,
                self.state.stage2a_best_hash,
                owner="stage2a best",
            )
        if self.state.revision_best_epoch >= 0:
            self._validate_best(
                self.best_path,
                self.state.revision_best_checkpoint_hash,
                owner="revision best",
            )
        self._set_optimizer_for_phase(self.state.phase)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        _move_optimizer_state(self.optimizer, self.device)
        saved_scheduler = payload.get("scheduler")
        if self.scheduler is None:
            if saved_scheduler is not None:
                raise ValueError("checkpoint has scheduler state but scheduler is disabled")
        else:
            if saved_scheduler is None:
                raise ValueError("checkpoint is missing scheduler state")
            self.scheduler.load_state_dict(saved_scheduler)
        if self.revision is not None:
            delta = payload.get("delta")
            if not torch.is_tensor(delta) or delta.shape != self.revision.delta.shape:
                raise ValueError("T-Revision checkpoint delta is invalid")
            self.revision.delta.data.copy_(delta.to(self.device))
        saved_best_model = payload.get("revision_best_model_state")
        saved_best_delta = payload.get("revision_best_delta_state")
        if self.state.revision_best_epoch >= 0:
            if not isinstance(saved_best_model, Mapping) or not torch.is_tensor(saved_best_delta):
                raise ValueError("T-Revision checkpoint is missing revision best state")
            self.revision_best_model_state = {
                str(name): value.detach().cpu().clone()
                for name, value in saved_best_model.items()
                if torch.is_tensor(value)
            }
            if len(self.revision_best_model_state) != len(saved_best_model):
                raise ValueError("T-Revision revision best model state is invalid")
            self.revision_best_delta_state = saved_best_delta.detach().cpu().clone()
        run_state = payload.get("run_state", {})
        self.run_state.cycle = int(run_state.get("cycle", 0))
        self.run_state.step = int(run_state.get("step", 0))
        self.run_state.phase = str(run_state.get("phase", "t_revision"))
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
        if self.state.phase is TRevisionPhase.COMPLETED:
            if not self.revised_transition_path.is_file():
                raise FileNotFoundError("T-Revision revised transition artifact is missing")
            revised = RevisedTransitionArtifact.load(self.revised_transition_path)
            if revised.artifact_hash != self.state.revised_transition_hash:
                raise ValueError("T-Revision revised transition artifact hash mismatch")
            if extend_completed_revision:
                self.state.phase = TRevisionPhase.REVISION_TRAINING
                self.state.revised_transition_hash = ""

    def _diagnostic_relative_l1(self, matrix: np.ndarray | Tensor) -> float | None:
        if self.diagnostic_transition is None:
            return None
        values = (
            matrix.detach().cpu().numpy()
            if torch.is_tensor(matrix)
            else np.asarray(matrix, dtype=np.float64)
        )
        denominator = float(np.abs(self.diagnostic_transition).sum())
        if denominator <= 0.0:
            return None
        return float(np.abs(values - self.diagnostic_transition).sum() / denominator)

    def train_stage1(self, *, max_epochs: int | None = None) -> None:
        if self.state.phase is not TRevisionPhase.STAGE1_TRAINING:
            raise ValueError("stage1 training is not active")
        target = self.method_config.stage1.epochs
        if max_epochs is not None:
            target = min(target, self.state.stage1_completed_epochs + max_epochs)
        while self.state.stage1_completed_epochs < target:
            epoch = self.state.stage1_completed_epochs
            self._seed_train_loader(10_000, epoch)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            train = _train_supervised_epoch(
                self.stage1_algorithm, self.train_loader, self.run_state, epoch
            )
            validation = evaluate_classification(
                self.model, self.noisy_validation_loader, self.loss, self.device
            )
            if self.scheduler is not None:
                self.scheduler.step()
            self.state.stage1_completed_epochs = epoch + 1
            self.state.stage1_global_step = self.run_state.step
            if validation["accuracy"] > self.state.stage1_best_metric:
                self.state.stage1_best_epoch = epoch
                self.state.stage1_best_metric = validation["accuracy"]
                self._save(self.stage1_best_path, role="stage1_best")
                self.state.stage1_best_hash = _file_sha256(self.stage1_best_path)
            self._append_metrics({
                "event": "epoch", "stage": "stage1", "epoch": epoch + 1,
                "global_step": self.state.stage1_global_step,
                "learning_rate": learning_rate, **train,
                "noisy_validation_loss": validation["loss"],
                "noisy_validation_accuracy": validation["accuracy"],
                "best_stage1_epoch": self.state.stage1_best_epoch + 1,
            })
            self._save_last()
        if self.state.stage1_completed_epochs == self.method_config.stage1.epochs:
            self._validate_best(self.stage1_best_path, self.state.stage1_best_hash, owner="stage1 best")
            self.state.advance(TRevisionPhase.STAGE1_READY)
            self._save_last()

    def initialize_transition(self) -> TransitionArtifact:
        if self.state.phase is not TRevisionPhase.STAGE1_READY:
            raise ValueError("transition initialization requires stage1_ready")
        self._validate_best(self.stage1_best_path, self.state.stage1_best_hash, owner="stage1 best")
        best = read_checkpoint(self.stage1_best_path, "cpu")
        if best.get("checkpoint_role") != "stage1_best":
            raise ValueError("stage1 best checkpoint identity mismatch")
        self.model.load_state_dict(best["model"])
        snapshot = collect_posterior_snapshot(
            self.model,
            self.posterior_loader,
            self.device,
            dataset=self.dataset,
            split="train",
        )
        if snapshot.noisy_probabilities.shape[1] != self.num_classes:
            raise ValueError(
                "T-Revision posterior snapshot class dimension does not match "
                f"the configured class count {self.num_classes}"
            )
        snapshot_classes = np.unique(snapshot.noisy_targets)
        if not np.array_equal(snapshot_classes, np.arange(self.num_classes)):
            missing = np.setdiff1d(
                np.arange(self.num_classes), snapshot_classes
            ).tolist()
            raise ValueError(
                "T-Revision posterior snapshot is missing noisy target classes: "
                f"{missing}"
            )
        base = AnchorTransitionEstimator().estimate(snapshot)
        metadata = dict(base.metadata)
        metadata.update({
            "method": "t_revision",
            "stage1_best_epoch": self.state.stage1_best_epoch,
            "stage1_best_validation_accuracy": self.state.stage1_best_metric,
            "stage1_best_checkpoint_sha256": self.state.stage1_best_hash,
            "noise_manifest_sha256": self.noise_metadata.get("manifest_sha256", ""),
            "noise_mapping_hash": self.noise_metadata.get("mapping_hash", ""),
            "matrix_convention": "T[i,j]=P(noisy=j|clean=i)",
            "condition_number": float(np.linalg.cond(base.matrix)),
            "row_sums": base.matrix.sum(axis=1).tolist(),
            "minimum": float(base.matrix.min()),
            "maximum": float(base.matrix.max()),
            "diagonal": np.diag(base.matrix).tolist(),
        })
        transition = TransitionArtifact(
            matrix=base.matrix,
            estimator="anchor",
            source_snapshot_hash=snapshot.snapshot_hash,
            metadata=metadata,
        )
        persisted_snapshot = _atomic_npz(
            self.snapshot_path,
            snapshot,
            PosteriorSnapshot.load,
            snapshot.snapshot_hash,
        )
        persisted_transition = _atomic_npz(
            self.initial_transition_path,
            transition,
            TransitionArtifact.load,
            transition.artifact_hash,
        )
        if persisted_transition.source_snapshot_hash != persisted_snapshot.snapshot_hash:
            raise ValueError("persisted initial transition provenance mismatch")
        self.snapshot = persisted_snapshot
        self.initial_transition = persisted_transition
        self.state.snapshot_hash = persisted_snapshot.snapshot_hash
        self.state.initial_transition_hash = persisted_transition.artifact_hash
        self.state.advance(TRevisionPhase.TRANSITION_INITIALIZED)
        transition_metrics = {
            "event": "transition_initialization",
            "stage": "transition_initialization",
            "posterior_snapshot_hash": self.state.snapshot_hash,
            "transition_initial_hash": self.state.initial_transition_hash,
            "pseudo_anchor_indices": persisted_transition.metadata["anchor_global_indices"],
            "pseudo_anchor_scores": persisted_transition.metadata["anchor_scores"],
            "row_sums": persisted_transition.matrix.sum(axis=1).tolist(),
            "minimum": float(persisted_transition.matrix.min()),
            "maximum": float(persisted_transition.matrix.max()),
            "diagonal": np.diag(persisted_transition.matrix).tolist(),
            "condition_number": float(np.linalg.cond(persisted_transition.matrix)),
        }
        true_error = self._diagnostic_relative_l1(persisted_transition.matrix)
        if true_error is not None:
            transition_metrics["true_T_relative_L1_error"] = true_error
        self._append_metrics(transition_metrics)
        self._save_last()
        return persisted_transition

    def start_classifier_initialization(self) -> None:
        if self.state.phase is not TRevisionPhase.TRANSITION_INITIALIZED:
            raise ValueError("classifier initialization requires transition_initialized")
        self._restore_initial_artifacts()
        best = read_checkpoint(self.stage1_best_path, "cpu")
        self.model.load_state_dict(best["model"])
        self.optimizer = self.classifier_optimizer_factory(self.model)
        self.scheduler = self.classifier_scheduler_factory(self.optimizer)
        self.run_state = RunState(phase="t_revision_classifier_initialization")
        self.state.advance(TRevisionPhase.CLASSIFIER_INITIALIZATION)
        self._save_last()

    def _transition_tensor(self) -> Tensor:
        if self.initial_transition is None:
            raise ValueError("initial transition is unavailable")
        return self.initial_transition.as_tensor(
            device=self.device, dtype=next(self.model.parameters()).dtype
        )

    def _train_reweight_epoch(self, transition_provider: Callable[[], Tensor]) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        sample_count = 0
        for raw_batch in self.train_loader:
            batch = Batch(raw_batch)
            inputs = batch.payload["input"].to(self.device, non_blocking=True)
            targets = batch.payload["target"].to(self.device, non_blocking=True)
            logits = self.model(inputs)
            result = t_revision_reweight_objective(
                logits,
                targets,
                transition_provider(),
                denominator_floor=self.method_config.denominator_floor,
            )
            self.optimizer.zero_grad(set_to_none=True)
            result.objective.backward()
            self.optimizer.step()
            count = int(targets.numel())
            sample_count += count
            for name, value in result.metrics.items():
                totals[name] = totals.get(name, 0.0) + value * count
            self.run_state.step += 1
        if sample_count == 0:
            raise RuntimeError("T-Revision training loader is empty")
        return {name: value / sample_count for name, value in totals.items()}

    def _evaluate_noisy(self, transition: Tensor) -> dict[str, float]:
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        samples = 0
        with torch.inference_mode():
            for raw_batch in self.noisy_validation_loader:
                batch = Batch(raw_batch)
                inputs = batch.payload["input"].to(self.device, non_blocking=True)
                targets = batch.payload["target"].to(self.device, non_blocking=True).long()
                clean_prob = torch.softmax(self.model(inputs), dim=1)
                noisy_prob = clean_prob @ transition
                observed = noisy_prob.gather(1, targets[:, None]).squeeze(1)
                if not bool(torch.isfinite(observed).all().item()) or bool(
                    (observed <= self.method_config.denominator_floor).any().item()
                ):
                    raise ValueError("T-Revision validation noisy probability is invalid")
                loss_sum += float((-observed.log()).sum().item())
                correct += int((noisy_prob.argmax(dim=1) == targets).sum().item())
                samples += int(targets.numel())
        if samples == 0:
            raise RuntimeError("T-Revision validation loader is empty")
        return {"loss": loss_sum / samples, "accuracy": correct / samples}

    def train_classifier_initialization(self, *, max_epochs: int | None = None) -> None:
        if self.state.phase is not TRevisionPhase.CLASSIFIER_INITIALIZATION:
            raise ValueError("classifier initialization is not active")
        target = self.method_config.classifier_initialization.epochs
        if max_epochs is not None:
            target = min(target, self.state.stage2a_completed_epochs + max_epochs)
        transition = self._transition_tensor()
        while self.state.stage2a_completed_epochs < target:
            epoch = self.state.stage2a_completed_epochs
            self._seed_train_loader(20_000, epoch)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            train = self._train_reweight_epoch(lambda: transition)
            validation = self._evaluate_noisy(transition)
            if self.scheduler is not None:
                self.scheduler.step()
            self.state.stage2a_completed_epochs = epoch + 1
            self.state.stage2a_global_step = self.run_state.step
            if validation["accuracy"] > self.state.stage2a_best_metric:
                self.state.stage2a_best_epoch = epoch
                self.state.stage2a_best_metric = validation["accuracy"]
                self._save(self.stage2a_best_path, role="stage2a_best")
                self.state.stage2a_best_hash = _file_sha256(self.stage2a_best_path)
            epoch_metrics = {
                "event": "epoch", "stage": "classifier_initialization",
                "epoch": epoch + 1, "global_step": self.state.stage2a_global_step,
                "learning_rate": learning_rate, **train,
                "revised_noisy_validation_loss": validation["loss"],
                "revised_noisy_validation_accuracy": validation["accuracy"],
                "best_epoch": self.state.stage2a_best_epoch + 1,
            }
            true_error = self._diagnostic_relative_l1(transition)
            if true_error is not None:
                epoch_metrics["true_T_relative_L1_error"] = true_error
            self._append_metrics(epoch_metrics)
            self._save_last()
        if self.state.stage2a_completed_epochs == target == self.method_config.classifier_initialization.epochs:
            self._validate_best(self.stage2a_best_path, self.state.stage2a_best_hash, owner="stage2a best")
            self.state.advance(TRevisionPhase.CLASSIFIER_READY)
            self._save_last()

    def start_revision(self) -> None:
        if self.state.phase is not TRevisionPhase.CLASSIFIER_READY:
            raise ValueError("revision requires classifier_ready")
        self._validate_best(self.stage2a_best_path, self.state.stage2a_best_hash, owner="stage2a best")
        best = read_checkpoint(self.stage2a_best_path, "cpu")
        self.model.load_state_dict(best["model"])
        self.revision = AdditiveTransitionRevision(self._transition_tensor()).to(self.device)
        self.optimizer = self.revision_optimizer_factory(self.model, self.revision)
        validate_revision_optimizer(self.optimizer, self.model.parameters(), self.revision)
        self.scheduler = self.revision_scheduler_factory(self.optimizer)
        self.run_state = RunState(phase="t_revision_revision")
        self.state.advance(TRevisionPhase.REVISION_TRAINING)
        self._save_last()

    def train_revision(self, *, max_epochs: int | None = None) -> None:
        if self.state.phase is not TRevisionPhase.REVISION_TRAINING or self.revision is None:
            raise ValueError("revision training is not active")
        target = self.method_config.revision.epochs
        if max_epochs is not None:
            target = min(target, self.state.revision_completed_epochs + max_epochs)
        while self.state.revision_completed_epochs < target:
            epoch = self.state.revision_completed_epochs
            self._seed_train_loader(30_000, epoch)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            train = self._train_reweight_epoch(self.revision)
            transition = self.revision()
            validation = self._evaluate_noisy(transition)
            if self.scheduler is not None:
                self.scheduler.step()
            self.state.revision_completed_epochs = epoch + 1
            self.state.revision_global_step = self.run_state.step
            if validation["accuracy"] > self.state.revision_best_metric:
                self.state.revision_best_epoch = epoch
                self.state.revision_best_metric = validation["accuracy"]
                self.revision_best_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                self.revision_best_delta_state = self.revision.delta.detach().cpu().clone()
                self._save(self.best_path, role="revision_best")
                self.state.revision_best_checkpoint_hash = _file_sha256(self.best_path)
            epoch_metrics = {
                "event": "epoch", "stage": "revision", "epoch": epoch + 1,
                "global_step": self.state.revision_global_step,
                "learning_rate": learning_rate, **train,
                "revised_noisy_validation_loss": validation["loss"],
                "revised_noisy_validation_accuracy": validation["accuracy"],
                **self.revision.diagnostics(),
            }
            true_error = self._diagnostic_relative_l1(transition)
            if true_error is not None:
                epoch_metrics["true_T_relative_L1_error"] = true_error
            self._append_metrics(epoch_metrics)
            self._save_last()
        if self.state.revision_completed_epochs == target == self.method_config.revision.epochs:
            self.complete()

    def complete(self) -> dict[str, Any]:
        if self.state.phase is not TRevisionPhase.REVISION_TRAINING:
            raise ValueError("T-Revision completion requires revision_training")
        if not self.best_path.is_file():
            raise FileNotFoundError("T-Revision revision best checkpoint is missing")
        self._validate_best(
            self.best_path,
            self.state.revision_best_checkpoint_hash,
            owner="revision best",
        )
        best = read_checkpoint(self.best_path, "cpu")
        if best.get("checkpoint_role") != "revision_best":
            raise ValueError("T-Revision revision best checkpoint identity mismatch")
        if self.revision is None:
            raise ValueError("T-Revision revision module is unavailable")
        current_model_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        current_delta_state = self.revision.delta.detach().cpu().clone()
        self.model.load_state_dict(best["model"])
        best_delta = best.get("delta")
        if not torch.is_tensor(best_delta):
            raise ValueError("T-Revision best checkpoint is missing delta")
        if self.revision_best_model_state is None or self.revision_best_delta_state is None:
            raise ValueError("T-Revision run state is missing revision best state")
        for name, value in best["model"].items():
            if name not in self.revision_best_model_state or not torch.equal(
                value.cpu(), self.revision_best_model_state[name]
            ):
                raise ValueError("T-Revision best model state provenance mismatch")
        if not torch.equal(best_delta.cpu(), self.revision_best_delta_state):
            raise ValueError("T-Revision best delta state provenance mismatch")
        self.revision.delta.data.copy_(best_delta.to(self.device))
        artifact = RevisedTransitionArtifact(
            initial_transition=self.initial_transition.matrix,
            delta=self.revision.delta.detach().cpu().numpy(),
            source_initial_artifact_hash=self.state.initial_transition_hash,
            stage2a_best_checkpoint_sha256=self.state.stage2a_best_hash,
            best_noisy_validation_accuracy=self.state.revision_best_metric,
            metadata={
                "method": "t_revision",
                "objective": "reweight",
                "matrix_convention": "T[i,j]=P(noisy=j|clean=i)",
                "noise_manifest_sha256": self.noise_metadata.get("manifest_sha256", ""),
                "noise_mapping_hash": self.noise_metadata.get("mapping_hash", ""),
                "best_revision_epoch": self.state.revision_best_epoch,
            },
        )
        persisted = _atomic_npz(
            self.revised_transition_path,
            artifact,
            RevisedTransitionArtifact.load,
            artifact.artifact_hash,
        )
        test = evaluate_classification(self.model, self.clean_test_loader, self.loss, self.device)
        self.state.revised_transition_hash = persisted.artifact_hash
        self.state.advance(TRevisionPhase.COMPLETED)
        final = {
            "event": "final", "method": "t_revision",
            "phase": TRevisionPhase.COMPLETED.value,
            "fidelity": "paper_experiment_raw_additive",
            "transition_mode": "paper_experiment_raw_additive",
            "revised_transition_is_probability_matrix": False,
            "raw_additive_warning": (
                "T_hat + delta is unconstrained and may contain negative values "
                "or rows that do not sum to one"
            ),
            "completed_stage1_epochs": self.state.stage1_completed_epochs,
            "stage1_global_step": self.state.stage1_global_step,
            "best_stage1_epoch": self.state.stage1_best_epoch + 1,
            "best_stage1_noisy_validation_accuracy": self.state.stage1_best_metric,
            "completed_classifier_initialization_epochs": self.state.stage2a_completed_epochs,
            "classifier_initialization_global_step": self.state.stage2a_global_step,
            "best_classifier_initialization_epoch": self.state.stage2a_best_epoch + 1,
            "best_classifier_initialization_noisy_validation_accuracy": (
                self.state.stage2a_best_metric
            ),
            "completed_revision_epochs": self.state.revision_completed_epochs,
            "revision_global_step": self.state.revision_global_step,
            "best_revision_epoch": self.state.revision_best_epoch + 1,
            "best_revision_validation_accuracy": self.state.revision_best_metric,
            "test_loss": test["loss"], "test_accuracy": test["accuracy"],
            "posterior_snapshot_hash": self.state.snapshot_hash,
            "transition_initial_hash": self.state.initial_transition_hash,
            "transition_revised_hash": self.state.revised_transition_hash,
            "artifact_paths": {
                "stage1_best": self.stage1_best_path.name,
                "posterior_snapshot": self.snapshot_path.name,
                "transition_initial": self.initial_transition_path.name,
                "stage2a_best": self.stage2a_best_path.name,
                "revision_best": self.best_path.name,
                "transition_revised": self.revised_transition_path.name,
                "resume_checkpoint": self.last_path.name,
                "metrics": self.metrics_path.name,
                "final_metrics": "final_metrics.json",
            },
            "pseudo_anchor_indices": self.initial_transition.metadata.get(
                "anchor_global_indices", []
            ),
            "initial_transition_diagnostics": {
                "row_sums": self.initial_transition.matrix.sum(axis=1).tolist(),
                "minimum": float(self.initial_transition.matrix.min()),
                "maximum": float(self.initial_transition.matrix.max()),
                "diagonal": np.diag(self.initial_transition.matrix).tolist(),
                "condition_number": float(
                    np.linalg.cond(self.initial_transition.matrix)
                ),
            },
            **persisted.diagnostics,
        }
        true_error = self._diagnostic_relative_l1(persisted.revised_transition)
        if true_error is not None:
            final["true_T_relative_L1_error"] = true_error
        self._append_metrics(final)
        final_path = self.run_dir / "final_metrics.json"
        temporary_final = final_path.with_suffix(".json.tmp")
        temporary_final.write_text(json.dumps(final, indent=2), encoding="utf-8")
        if json.loads(temporary_final.read_text(encoding="utf-8")) != final:
            raise ValueError("T-Revision final summary verification failed")
        temporary_final.replace(final_path)
        self.model.load_state_dict(current_model_state)
        self.revision.delta.data.copy_(current_delta_state.to(self.device))
        self._save_last()
        return final

    def run(self) -> dict[str, Any]:
        if self.state.phase is TRevisionPhase.STAGE1_TRAINING:
            self.train_stage1()
        if self.state.phase is TRevisionPhase.STAGE1_READY:
            self.initialize_transition()
        if self.state.phase is TRevisionPhase.TRANSITION_INITIALIZED:
            self.start_classifier_initialization()
        if self.state.phase is TRevisionPhase.CLASSIFIER_INITIALIZATION:
            self.train_classifier_initialization()
        if self.state.phase is TRevisionPhase.CLASSIFIER_READY:
            self.start_revision()
        if self.state.phase is TRevisionPhase.REVISION_TRAINING:
            self.train_revision()
        final_path = self.run_dir / "final_metrics.json"
        if self.state.phase is not TRevisionPhase.COMPLETED or not final_path.is_file():
            raise RuntimeError("T-Revision run did not reach completion")
        return json.loads(final_path.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.stage1_algorithm.close()
