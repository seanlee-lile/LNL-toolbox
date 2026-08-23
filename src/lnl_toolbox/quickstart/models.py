from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class QuickStartDatasetSummary:
    alias: str
    adapter: str
    path: str | None
    display_name: str
    num_classes: int | None
    train_size: int | None
    validation_size: int | None
    test_size: int | None
    noise_status: str
    noise_origin: str
    clean_train_labels: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QuickStartNoiseSelection:
    kind: str
    key: str
    rate: float | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QuickStartMethodOption:
    paper_id: str
    acronym: str
    title: str
    summary: str
    venue: str
    year: int
    status: str
    reasons: tuple[str, ...] = ()
    required_user_inputs: tuple[str, ...] = ()
    recipe_id: str | None = None
    config_kind: str | None = None
    candidate_config: Mapping[str, Any] | None = None
    required_input_paths: tuple[
        tuple[str, tuple[tuple[str, ...], ...]], ...
    ] = ()

    def __post_init__(self) -> None:
        if self.status not in {"ready", "needs_input", "unsupported", "metadata_error"}:
            raise ValueError(f"invalid Quick Start method status: {self.status}")

    def to_dict(self, *, include_config: bool = False) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        value["required_user_inputs"] = list(self.required_user_inputs)
        value["required_input_paths"] = [
            [name, [list(path) for path in paths]]
            for name, paths in self.required_input_paths
        ]
        if not include_config:
            value.pop("candidate_config", None)
        return value


@dataclass(frozen=True, slots=True)
class QuickStartPlan:
    plan_id: str
    dataset_alias: str
    paper_id: str
    method: str
    noise: QuickStartNoiseSelection
    config_kind: str
    recipe_id: str | None
    generated_config_path: str | None
    status: str
    required_user_inputs: tuple[str, ...] = ()
    command: str | None = None
    dry_run_command: str | None = None
    summary: str = ""
    provenance_path: str | None = None
    details: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["noise"] = self.noise.to_dict()
        value["required_user_inputs"] = list(self.required_user_inputs)
        value["details"] = list(self.details)
        return value


__all__ = [
    "QuickStartDatasetSummary",
    "QuickStartMethodOption",
    "QuickStartNoiseSelection",
    "QuickStartPlan",
]
