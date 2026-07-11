from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .result import Artifact
from .state import RunState


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    uri: str
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Checkpoint:
    run_state: RunState
    component_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactSink(Protocol):
    """Storage adapter for files, object stores, databases, or experiment trackers."""

    def write(self, artifact: Artifact, state: RunState) -> ArtifactRef: ...


class CheckpointStore(Protocol):
    """Persistence boundary; serialization format is selected by the adapter."""

    def save(self, name: str, checkpoint: Checkpoint) -> str: ...
    def load(self, name: str) -> Checkpoint: ...

