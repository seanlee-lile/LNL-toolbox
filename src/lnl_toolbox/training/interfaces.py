from __future__ import annotations

"""Public contracts shared by every paper runner."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .artifacts import ArtifactRef
from .reporting import RunSession, load_metric_events


@dataclass
class RunContext:
    """Resolved execution context passed through the common runner lifecycle."""

    config: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    run_dir: Path
    session: RunSession
    device: Any
    seed: int
    phase: str = "default"
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def events(self) -> list[dict[str, Any]]:
        return load_metric_events(self.run_dir / "metrics.jsonl")


@dataclass(frozen=True)
class EvaluationResult:
    """Common evaluation result independent of the paper method."""

    metrics: Mapping[str, Any]
    split: str = "test"


@dataclass(frozen=True)
class RunResult:
    """Stable return value for every public runner."""

    run_dir: Path
    status: str
    final_metrics: Mapping[str, Any]
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    artifacts: Sequence[ArtifactRef | Mapping[str, Any]] = ()

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        *,
        status: str = "completed",
        resolve: bool = True,
    ) -> "RunResult":
        root = Path(run_dir).expanduser()
        if resolve:
            root = root.resolve()
        final: Mapping[str, Any] = {}
        final_path = root / "final_metrics.json"
        if final_path.is_file():
            try:
                loaded = json.loads(final_path.read_text(encoding="utf-8"))
                if isinstance(loaded, Mapping):
                    final = dict(loaded)
            except (OSError, json.JSONDecodeError):
                final = {}
        artifacts: Sequence[ArtifactRef | Mapping[str, Any]] = ()
        artifact_path = root / "artifacts.json"
        if artifact_path.is_file():
            try:
                loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
                if isinstance(loaded, Mapping) and isinstance(loaded.get("artifacts"), list):
                    artifacts = tuple(item for item in loaded["artifacts"] if isinstance(item, Mapping))
            except (OSError, json.JSONDecodeError):
                artifacts = ()
        return cls(
            run_dir=root,
            status=status,
            final_metrics=final,
            best_checkpoint=(root / "best.pt") if (root / "best.pt").is_file() else None,
            last_checkpoint=(root / "last.pt") if (root / "last.pt").is_file() else None,
            artifacts=artifacts,
        )


class ExperimentRunner(Protocol):
    """Public protocol implemented by every registered paper runner."""

    spec: Any

    def prepare(
        self,
        context: RunContext | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> RunContext: ...

    def fit(
        self,
        context: RunContext | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> RunResult: ...

    def evaluate(self, result: RunResult) -> EvaluationResult: ...

    def save_checkpoint(self, result: RunResult, boundary: str = "last") -> Path: ...

    def load_checkpoint(self, context: RunContext, path: str | Path) -> None: ...


__all__ = ["EvaluationResult", "ExperimentRunner", "RunContext", "RunResult"]
