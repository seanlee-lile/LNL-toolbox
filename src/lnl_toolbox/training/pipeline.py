from __future__ import annotations

"""Reusable preparation and component composition for noisy-label pipelines."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from lnl_toolbox.noise.estimators import (
    PosteriorSnapshot,
    TransitionEstimator,
)
from lnl_toolbox.noise.transition import TransitionArtifact, TransitionProvider
from lnl_toolbox.plugins.builtin.catalog import (
    build_builtin_risk_corrector,
    build_builtin_transition_estimator,
    build_builtin_weight_provider,
)
from lnl_toolbox.algorithms.transition_risk import RiskCorrector
from lnl_toolbox.treatments.weights import WeightInput, WeightProvider
from lnl_toolbox.training.snapshots import collect_posterior_snapshot, pretrain_noisy_classifier


class PipelinePhase(str, Enum):
    WARMUP = "warmup"
    SNAPSHOT = "snapshot"
    TRAIN = "train"
    EVALUATE = "evaluate"
    COMPLETE = "complete"


@dataclass
class PipelineState:
    phase: PipelinePhase = PipelinePhase.TRAIN
    cycle: int = 0
    stopped: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "cycle": self.cycle,
            "stopped": self.stopped,
            "metadata": dict(self.metadata),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.phase = PipelinePhase(str(state["phase"]))
        self.cycle = int(state["cycle"])
        self.stopped = bool(state["stopped"])
        self.metadata = dict(state.get("metadata", {}))


@dataclass(frozen=True)
class PipelineArtifacts:
    """Serializable identities produced before the main training phase."""

    snapshot: PosteriorSnapshot | None = None
    transition: TransitionProvider | None = None
    snapshot_path: str | None = None
    transition_path: str | None = None

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.snapshot is not None:
            state.update({
                "snapshot_hash": self.snapshot.snapshot_hash,
                "snapshot_path": self.snapshot_path,
            })
        if isinstance(self.transition, TransitionArtifact):
            state.update({
                "transition_artifact_hash": self.transition.artifact_hash,
                "transition_path": self.transition_path,
            })
        return state


@dataclass
class StandardNoisyERMPipeline:
    """Generic single-model pipeline with optional estimator and consumers.

    Paper-specific mathematics stays in the registered estimator, corrector, or
    weight provider. This class only owns stage ordering and artifact handoff.
    """

    risk_corrector: RiskCorrector | None = None
    transition_estimator: TransitionEstimator | None = None
    weight_provider: WeightProvider[WeightInput] | None = None
    warmup_epochs: int = 0
    artifacts: PipelineArtifacts = field(default_factory=PipelineArtifacts)
    state: PipelineState = field(default_factory=PipelineState)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "StandardNoisyERMPipeline":
        values = dict(config or {})
        risk_config = values.get("risk_corrector")
        estimator_config = values.get("transition_estimator")
        weight_config = values.get("weight_provider")
        risk = (
            None
            if risk_config in (None, False)
            else build_builtin_risk_corrector(risk_config)
        )
        estimator = (
            None
            if estimator_config in (None, False)
            else build_builtin_transition_estimator(estimator_config)
        )
        weight = (
            None
            if weight_config in (None, False)
            else build_builtin_weight_provider(weight_config)
        )
        warmup_epochs = int(values.get("warmup_epochs", 0))
        if warmup_epochs < 0:
            raise ValueError("pipeline.warmup_epochs must be non-negative")
        if risk is not None and estimator is None:
            raise ValueError(
                "pipeline.risk_corrector requires pipeline.transition_estimator"
            )
        return cls(risk, estimator, weight, warmup_epochs)

    def warmup(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader,
        device: torch.device,
    ) -> None:
        """Train a noisy-label classifier before posterior estimation."""

        if self.warmup_epochs == 0:
            return
        self.state.phase = PipelinePhase.WARMUP
        pretrain_noisy_classifier(
            model,
            optimizer,
            loader,
            device,
            epochs=self.warmup_epochs,
        )
        self.state.phase = PipelinePhase.SNAPSHOT

    def prepare_transition(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader,
        device: torch.device,
        dataset: str,
        split: str,
        run_dir: str | Path,
    ) -> PipelineArtifacts:
        """Run warm-up, collect a stable snapshot, and estimate one artifact."""

        if self.transition_estimator is None:
            return self.artifacts
        self.warmup(model, optimizer, loader, device)
        snapshot = collect_posterior_snapshot(
            model,
            loader,
            device,
            dataset=dataset,
            split=split,
        )
        destination = Path(run_dir)
        snapshot_path = destination / "posterior_snapshot.npz"
        snapshot.save(snapshot_path)
        transition = self.transition_estimator.estimate(snapshot)
        transition_path = destination / "transition_artifact.npz"
        if isinstance(transition, TransitionArtifact):
            transition.save(transition_path)
        self.artifacts = PipelineArtifacts(
            snapshot=snapshot,
            transition=transition,
            snapshot_path=str(snapshot_path),
            transition_path=str(transition_path),
        )
        self.state.phase = PipelinePhase.TRAIN
        return self.artifacts

    def load_artifacts(self, run_dir: str | Path) -> bool:
        """Restore prepared artifacts when resuming a staged run."""

        if self.transition_estimator is None:
            return False
        destination = Path(run_dir)
        snapshot_path = destination / "posterior_snapshot.npz"
        transition_path = destination / "transition_artifact.npz"
        if not snapshot_path.is_file() or not transition_path.is_file():
            return False
        snapshot = PosteriorSnapshot.load(snapshot_path)
        transition = TransitionArtifact.load(transition_path)
        if transition.source_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("transition artifact does not match posterior snapshot")
        self.artifacts = PipelineArtifacts(
            snapshot=snapshot,
            transition=transition,
            snapshot_path=str(snapshot_path),
            transition_path=str(transition_path),
        )
        self.state.phase = PipelinePhase.TRAIN
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": "standard_noisy_erm",
            "warmup_epochs": self.warmup_epochs,
            "artifacts": self.artifacts.state_dict(),
            "state": self.state.state_dict(),
        }
