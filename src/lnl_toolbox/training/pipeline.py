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
from lnl_toolbox.noise.statistics import StatisticArtifact
from lnl_toolbox.plugins.builtin.catalog import (
    build_builtin_objective_consumer,
    build_builtin_risk_corrector,
    build_builtin_regularizer,
    build_builtin_statistic_estimator,
    build_builtin_transition_estimator,
    build_builtin_weight_provider,
)
from lnl_toolbox.algorithms.transition_risk import RiskCorrector
from lnl_toolbox.treatments.weights import SupervisedWeightInput, WeightProvider
from lnl_toolbox.training.snapshots import collect_posterior_snapshot, pretrain_noisy_classifier
from lnl_toolbox.training.snapshots import FeatureSnapshot, collect_feature_snapshot
from lnl_toolbox.models.feature_output import forward_with_features


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
    feature_snapshot: FeatureSnapshot | None = None
    statistic: StatisticArtifact | None = None
    feature_snapshot_path: str | None = None
    statistic_path: str | None = None

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.snapshot is not None:
            state.update({
                "snapshot_hash": self.snapshot.snapshot_hash,
                "snapshot_path": self.snapshot_path,
                "snapshot_dataset": self.snapshot.dataset,
                "snapshot_split": self.snapshot.split,
            })
        if isinstance(self.transition, TransitionArtifact):
            state.update({
                "transition_artifact_hash": self.transition.artifact_hash,
                "transition_path": self.transition_path,
                "transition_estimator": self.transition.estimator,
                "transition_source_snapshot_hash": (
                    self.transition.source_snapshot_hash
                ),
            })
        if self.feature_snapshot is not None:
            state.update({
                "feature_snapshot_hash": self.feature_snapshot.snapshot_hash,
                "feature_snapshot_path": self.feature_snapshot_path,
                "feature_snapshot_dataset": self.feature_snapshot.dataset,
                "feature_snapshot_split": self.feature_snapshot.split,
            })
        if self.statistic is not None:
            state.update({
                "statistic_artifact_hash": self.statistic.artifact_hash,
                "statistic_path": self.statistic_path,
                "statistic_estimator": self.statistic.estimator,
                "statistic_source_snapshot_hash": self.statistic.metadata.get(
                    "source_snapshot_hash", ""
                ),
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
    statistic_estimator: Any | None = None
    objective_consumer: Any | None = None
    weight_provider: WeightProvider[SupervisedWeightInput] | None = None
    regularizer: Any | None = None
    warmup_epochs: int = 0
    artifacts: PipelineArtifacts = field(default_factory=PipelineArtifacts)
    state: PipelineState = field(default_factory=PipelineState)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "StandardNoisyERMPipeline":
        values = dict(config or {})
        risk_config = values.get("risk_corrector")
        estimator_config = values.get("transition_estimator")
        statistic_config = values.get("statistic_estimator")
        objective_config = values.get("objective_consumer")
        weight_config = values.get("weight_provider")
        regularizer_config = values.get("regularizer")
        if objective_config not in (None, False):
            configured_conflicts = [
                name
                for name, value in (
                    ("risk_corrector", risk_config),
                    ("transition_estimator", estimator_config),
                    ("WeightProvider", weight_config),
                )
                if value not in (None, False)
            ]
            if configured_conflicts:
                raise ValueError(
                    "pipeline.objective_consumer cannot be combined with "
                    + ", ".join(configured_conflicts)
                )
        if isinstance(weight_config, Mapping):
            weight_name = str(weight_config.get("name", "")).strip().lower()
            if weight_name == "binary_rcn_importance":
                raise ValueError(
                    "binary_rcn_importance requires an explicit noisy-label "
                    "posterior producer and cannot be connected to the ordinary "
                    "supervised pipeline"
                )
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
        statistic_estimator = (
            None
            if statistic_config in (None, False)
            else build_builtin_statistic_estimator(statistic_config)
        )
        objective_consumer = (
            None
            if objective_config in (None, False)
            else build_builtin_objective_consumer(objective_config)
        )
        regularizer = (
            None
            if regularizer_config in (None, False)
            else build_builtin_regularizer(regularizer_config)
        )
        warmup_epochs = int(values.get("warmup_epochs", 0))
        if warmup_epochs < 0:
            raise ValueError("pipeline.warmup_epochs must be non-negative")
        if risk is not None and estimator is None:
            raise ValueError(
                "pipeline.risk_corrector requires pipeline.transition_estimator"
            )
        if objective_consumer is not None:
            incompatible = []
            if risk is not None or estimator is not None:
                incompatible.append("risk/transition correction")
            if weight is not None:
                incompatible.append("WeightProvider")
            unknown_composition = sorted(
                set(values)
                - {
                    "risk_corrector",
                    "transition_estimator",
                    "weight_provider",
                    "statistic_estimator",
                    "objective_consumer",
                    "regularizer",
                    "warmup_epochs",
                }
            )
            if unknown_composition:
                incompatible.extend(unknown_composition)
            if incompatible:
                raise ValueError(
                    "pipeline.objective_consumer cannot be combined with "
                    + ", ".join(incompatible)
                )
        if statistic_estimator is not None and objective_consumer is None:
            raise ValueError(
                "pipeline.statistic_estimator requires pipeline.objective_consumer"
            )
        return cls(
            risk_corrector=risk,
            transition_estimator=estimator,
            statistic_estimator=statistic_estimator,
            weight_provider=weight,
            objective_consumer=objective_consumer,
            regularizer=regularizer,
            warmup_epochs=warmup_epochs,
        )

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
        model.to(device)
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

    def prepare_statistics(
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
        """Run warm-up, feature snapshot, and a generic statistic estimator."""

        if self.statistic_estimator is None:
            return self.artifacts
        model.to(device)
        self.warmup(model, optimizer, loader, device)
        snapshot = collect_feature_snapshot(
            model,
            loader,
            device,
            dataset=dataset,
            split=split,
            feature_extractor=lambda current_model, inputs: forward_with_features(
                current_model, inputs
            ).features,
        )
        destination = Path(run_dir)
        feature_snapshot_path = destination / "feature_snapshot.npz"
        snapshot.save(feature_snapshot_path)
        statistic = self.statistic_estimator.estimate(snapshot)
        if not isinstance(statistic, StatisticArtifact):
            raise TypeError("statistic estimator must return a StatisticArtifact")
        statistic_path = destination / "statistic_artifact.npz"
        statistic.save(statistic_path)
        binder = getattr(self.objective_consumer, "bind_statistic", None)
        if callable(binder):
            binder(statistic)
        elif self.objective_consumer is not None and hasattr(self.objective_consumer, "statistic"):
            self.objective_consumer.statistic = statistic
        self.artifacts = PipelineArtifacts(
            feature_snapshot=snapshot,
            statistic=statistic,
            feature_snapshot_path=str(feature_snapshot_path),
            statistic_path=str(statistic_path),
        )
        self.state.phase = PipelinePhase.TRAIN
        return self.artifacts

    def prepare(self, **kwargs: Any) -> PipelineArtifacts:
        """Prepare whichever artifact lifecycle is selected by configuration."""

        if self.statistic_estimator is not None:
            return self.prepare_statistics(**kwargs)
        return self.prepare_transition(**kwargs)

    def load_artifacts(self, run_dir: str | Path) -> bool:
        """Restore prepared artifacts when resuming a staged run."""

        if self.statistic_estimator is not None:
            destination = Path(run_dir)
            feature_snapshot_path = destination / "feature_snapshot.npz"
            statistic_path = destination / "statistic_artifact.npz"
            if not feature_snapshot_path.is_file() or not statistic_path.is_file():
                return False
            snapshot = FeatureSnapshot.load(feature_snapshot_path)
            statistic = StatisticArtifact.load(statistic_path)
            if statistic.metadata.get("source_snapshot_hash") != snapshot.snapshot_hash:
                raise ValueError("statistic artifact does not match feature snapshot")
            binder = getattr(self.objective_consumer, "bind_statistic", None)
            if callable(binder):
                binder(statistic)
            self.artifacts = PipelineArtifacts(
                feature_snapshot=snapshot,
                statistic=statistic,
                feature_snapshot_path=str(feature_snapshot_path),
                statistic_path=str(statistic_path),
            )
            self.state.phase = PipelinePhase.TRAIN
            return True
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

    def _owned_components(self) -> dict[str, Any]:
        """Return stateful preparation components owned by this pipeline."""

        components: dict[str, Any] = {}
        if self.transition_estimator is not None:
            components["transition_estimator"] = self.transition_estimator
        if self.objective_consumer is not None:
            components["objective_consumer"] = self.objective_consumer
        return components

    def component_state_dict(self) -> dict[str, Any]:
        """Serialize only pipeline-owned components that expose state."""

        states: dict[str, Any] = {}
        for name, component in self._owned_components().items():
            state_dict = getattr(component, "state_dict", None)
            if callable(state_dict):
                state = state_dict()
                if not isinstance(state, Mapping):
                    raise TypeError(
                        f"pipeline component {name!r} state must be a mapping"
                    )
                states[name] = dict(state)
        return states

    def load_component_states(
        self,
        states: Mapping[str, Any] | None,
        *,
        legacy: bool = False,
    ) -> None:
        """Restore registered pipeline-owned components in stable name order."""

        if states is None:
            if legacy:
                return
            states = {}
        if not isinstance(states, Mapping):
            raise TypeError("checkpoint component_states must be a mapping")
        components = self._owned_components()
        unknown = sorted(set(states) - set(components))
        if unknown:
            raise ValueError(
                f"checkpoint contains unknown pipeline component states: {unknown}"
            )
        for name in sorted(components):
            component = components[name]
            loader = getattr(component, "load_state_dict", None)
            if not callable(loader):
                if name in states:
                    raise ValueError(
                        f"pipeline component {name!r} is not stateful"
                    )
                continue
            if name not in states:
                if legacy:
                    continue
                raise ValueError(
                    f"checkpoint is missing pipeline component state {name!r}"
                )
            component_state = states[name]
            if not isinstance(component_state, Mapping):
                raise TypeError(
                    f"pipeline component {name!r} state must be a mapping"
                )
            loader(dict(component_state))

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore lifecycle state while preserving configured components."""

        if not isinstance(state, Mapping):
            raise TypeError("checkpoint pipeline state must be a mapping")
        if state.get("name") != "standard_noisy_erm":
            raise ValueError("checkpoint pipeline identity mismatch")
        if int(state.get("warmup_epochs", -1)) != self.warmup_epochs:
            raise ValueError("checkpoint pipeline warmup configuration mismatch")
        lifecycle = state.get("state")
        if not isinstance(lifecycle, Mapping):
            raise ValueError("checkpoint pipeline state is missing lifecycle state")
        self.state.load_state_dict(lifecycle)

    @staticmethod
    def _load_resume_artifacts(
        run_dir: str | Path,
        *,
        dataset: str,
        split: str,
    ) -> PipelineArtifacts:
        destination = Path(run_dir)
        snapshot_path = destination / "posterior_snapshot.npz"
        transition_path = destination / "transition_artifact.npz"
        missing = [
            str(path.name)
            for path in (snapshot_path, transition_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"artifact missing during resume: {', '.join(missing)}"
            )
        try:
            snapshot = PosteriorSnapshot.load(snapshot_path)
            transition = TransitionArtifact.load(transition_path)
        except ValueError as exc:
            if "hash" in str(exc).lower():
                raise ValueError(f"artifact hash mismatch: {exc}") from exc
            raise
        if snapshot.dataset != dataset or snapshot.split != split:
            raise ValueError(
                "artifact provenance mismatch: "
                f"expected dataset={dataset!r}, split={split!r}; "
                f"found dataset={snapshot.dataset!r}, split={snapshot.split!r}"
            )
        if transition.source_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError(
                "artifact provenance mismatch: transition source snapshot "
                "does not match posterior snapshot"
            )
        return PipelineArtifacts(
            snapshot=snapshot,
            transition=transition,
            snapshot_path=str(snapshot_path),
            transition_path=str(transition_path),
        )

    @staticmethod
    def _load_resume_statistics(
        run_dir: str | Path,
        *,
        dataset: str,
        split: str,
    ) -> PipelineArtifacts:
        destination = Path(run_dir)
        snapshot_path = destination / "feature_snapshot.npz"
        statistic_path = destination / "statistic_artifact.npz"
        missing = [
            str(path.name)
            for path in (snapshot_path, statistic_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"statistic artifact missing during resume: {', '.join(missing)}"
            )
        snapshot = FeatureSnapshot.load(snapshot_path)
        statistic = StatisticArtifact.load(statistic_path)
        if snapshot.dataset != dataset or snapshot.split != split:
            raise ValueError(
                "statistic artifact provenance mismatch: "
                f"expected dataset={dataset!r}, split={split!r}; "
                f"found dataset={snapshot.dataset!r}, split={snapshot.split!r}"
            )
        if statistic.metadata.get("source_snapshot_hash") != snapshot.snapshot_hash:
            raise ValueError("statistic artifact source snapshot hash mismatch")
        return PipelineArtifacts(
            feature_snapshot=snapshot,
            statistic=statistic,
            feature_snapshot_path=str(snapshot_path),
            statistic_path=str(statistic_path),
        )

    def restore_for_resume(
        self,
        run_dir: str | Path,
        *,
        checkpoint_state: Mapping[str, Any] | None,
        component_states: Mapping[str, Any] | None,
        dataset: str,
        split: str,
    ) -> list[str]:
        """Strictly restore pipeline state and verify persisted artifacts."""

        legacy = checkpoint_state is None
        warnings: list[str] = []
        if legacy:
            warnings.append(
                "Legacy checkpoint has no pipeline state; validated artifacts "
                "were restored without checkpoint identity comparison."
            )
        else:
            self.load_state_dict(checkpoint_state)

        if self.statistic_estimator is not None:
            artifacts = self._load_resume_statistics(
                run_dir,
                dataset=dataset,
                split=split,
            )
            if not legacy:
                expected = checkpoint_state.get("artifacts")
                if not isinstance(expected, Mapping):
                    raise ValueError("checkpoint identity mismatch: statistic artifact identity is missing")
                actual = artifacts.state_dict()
                for name in ("feature_snapshot_hash", "statistic_artifact_hash"):
                    if expected.get(name) != actual.get(name):
                        raise ValueError(f"checkpoint identity mismatch for {name}")
            self.artifacts = artifacts
            binder = getattr(self.objective_consumer, "bind_statistic", None)
            if callable(binder):
                binder(artifacts.statistic)
        elif self.transition_estimator is not None:
            artifacts = self._load_resume_artifacts(
                run_dir,
                dataset=dataset,
                split=split,
            )
            if not legacy:
                expected = checkpoint_state.get("artifacts")
                if not isinstance(expected, Mapping):
                    raise ValueError(
                        "checkpoint identity mismatch: pipeline artifact "
                        "identity is missing"
                    )
                actual = artifacts.state_dict()
                identity_fields = (
                    "snapshot_hash",
                    "snapshot_dataset",
                    "snapshot_split",
                    "transition_artifact_hash",
                    "transition_estimator",
                    "transition_source_snapshot_hash",
                )
                required_identity = (
                    "snapshot_hash",
                    "transition_artifact_hash",
                )
                missing_required = [
                    name for name in required_identity if name not in expected
                ]
                if missing_required:
                    raise ValueError(
                        "checkpoint identity mismatch: required pipeline "
                        f"artifact identity is missing: {missing_required}"
                    )
                mismatched = [
                    name
                    for name in identity_fields
                    if name in expected and expected.get(name) != actual.get(name)
                ]
                if mismatched:
                    raise ValueError(
                        "checkpoint identity mismatch for pipeline artifacts: "
                        f"{mismatched}"
                    )
                legacy_identity = [
                    name for name in identity_fields if name not in expected
                ]
                if legacy_identity:
                    warnings.append(
                        "Legacy checkpoint pipeline artifact identity lacks "
                        f"{legacy_identity}; available hashes were verified."
                    )
            self.artifacts = artifacts
        elif not legacy and checkpoint_state.get("artifacts"):
            raise ValueError(
                "checkpoint identity mismatch: current pipeline does not use "
                "transition artifacts"
            )

        self.load_component_states(component_states, legacy=legacy)
        return warnings

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": "standard_noisy_erm",
            "warmup_epochs": self.warmup_epochs,
            "artifacts": self.artifacts.state_dict(),
            "state": self.state.state_dict(),
        }
