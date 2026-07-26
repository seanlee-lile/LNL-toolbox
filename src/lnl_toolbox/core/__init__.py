"""Framework contracts that do not depend on a task, model library, or LNL method."""

from .algorithm import Algorithm
from .batch import Batch
from .component import Component, Stateful
from .context import ExperimentContext
from .evaluator import Evaluator
from .result import (
    Artifact,
    CandidateLabelResult,
    PseudoLabelResult,
    SoftTargetResult,
    StepResult,
)
from .state import RunState
from .storage import ArtifactRef, ArtifactSink, Checkpoint, CheckpointStore
from .targets import (
    CandidateSetProvider,
    ComplementaryLabelResult,
    LabelProvider,
    PseudoLabelProvider,
    SoftTargetProvider,
    TargetInput,
)

__all__ = [
    "Algorithm",
    "Artifact",
    "ArtifactRef",
    "ArtifactSink",
    "CandidateLabelResult",
    "CandidateSetProvider",
    "Batch",
    "Component",
    "Checkpoint",
    "CheckpointStore",
    "Evaluator",
    "ExperimentContext",
    "ComplementaryLabelResult",
    "LabelProvider",
    "PseudoLabelProvider",
    "PseudoLabelResult",
    "RunState",
    "Stateful",
    "SoftTargetProvider",
    "SoftTargetResult",
    "StepResult",
    "TargetInput",
]
