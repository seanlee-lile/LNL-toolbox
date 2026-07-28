from __future__ import annotations

"""Deterministic, auditable paper-parameter sampling contracts."""

from dataclasses import dataclass
import json
import random
from typing import Any, Mapping, Sequence


def _json_safe(value: Any) -> Any:
    """Return a detached JSON-compatible value for run provenance."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise TypeError("paper parameters must be JSON-compatible") from exc


@dataclass(frozen=True)
class ParameterRecord:
    """The one sampled parameter set used by a paper reproduction."""

    paper: str
    sampling_seed: int
    parameters: Mapping[str, Any]
    candidates: Mapping[str, tuple[Any, ...]]
    sources: Mapping[str, str]

    def __post_init__(self) -> None:
        paper = str(self.paper).strip()
        if not paper:
            raise ValueError("paper must not be empty")
        if int(self.sampling_seed) < 0:
            raise ValueError("sampling_seed must be non-negative")
        parameters = dict(self.parameters)
        candidates = {str(name): tuple(values) for name, values in dict(self.candidates).items()}
        _json_safe(parameters)
        _json_safe({name: list(values) for name, values in candidates.items()})
        sources = {str(name): str(value) for name, value in dict(self.sources).items()}
        unknown = set(parameters) - set(candidates)
        if unknown:
            raise ValueError(f"sampled parameters lack candidate sets: {sorted(unknown)}")
        for name, value in parameters.items():
            if value not in candidates[name]:
                raise ValueError(f"parameter {name!r} is outside its candidate set")
        object.__setattr__(self, "paper", paper)
        object.__setattr__(self, "sampling_seed", int(self.sampling_seed))
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "sources", sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper": self.paper,
            "sampling_seed": self.sampling_seed,
            "parameters": _json_safe(dict(self.parameters)),
            "candidates": _json_safe({key: list(value) for key, value in self.candidates.items()}),
            "sources": dict(self.sources),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterRecord":
        if not isinstance(value, Mapping):
            raise TypeError("parameter record must be a mapping")
        return cls(
            paper=str(value["paper"]),
            sampling_seed=int(value["sampling_seed"]),
            parameters=dict(value["parameters"]),
            candidates={key: tuple(values) for key, values in dict(value["candidates"]).items()},
            sources=dict(value.get("sources", {})),
        )


def sample_parameters(
    paper: str,
    seed: int,
    candidates: Mapping[str, Sequence[Any]],
    *,
    sources: Mapping[str, str] | None = None,
) -> ParameterRecord:
    """Sample exactly one value per parameter with a reproducible local RNG."""

    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    normalized = {str(name): tuple(values) for name, values in candidates.items()}
    if not normalized or any(not values for values in normalized.values()):
        raise ValueError("every paper parameter must have a non-empty candidate set")
    rng = random.Random(int(seed))
    parameters = {name: rng.choice(values) for name, values in sorted(normalized.items())}
    return ParameterRecord(
        paper=paper,
        sampling_seed=int(seed),
        parameters=parameters,
        candidates=normalized,
        sources=sources or {},
    )


def resolve_parameter_sampling(config: Mapping[str, Any]) -> tuple[dict[str, Any], ParameterRecord | None]:
    """Resolve an optional ``parameter_sampling`` config without changing old configs."""

    resolved = dict(config)
    sampling = resolved.get("parameter_sampling")
    if sampling in (None, False):
        record = resolved.get("parameter_record")
        return resolved, None if record is None else ParameterRecord.from_dict(record)
    if not isinstance(sampling, Mapping):
        raise TypeError("parameter_sampling must be a mapping or false")
    candidates = sampling.get("candidates")
    if not isinstance(candidates, Mapping):
        raise TypeError("parameter_sampling.candidates must be a mapping")
    record = sample_parameters(
        str(sampling.get("paper", "unspecified")),
        int(sampling.get("seed", resolved.get("seed", 1))),
        candidates,
        sources=sampling.get("sources"),
    )
    resolved["resolved_parameters"] = dict(record.parameters)
    resolved["parameter_record"] = record.to_dict()
    return resolved, record


PAPER_PARAMETER_CANDIDATES: dict[str, dict[str, tuple[Any, ...]]] = {
    "binary_risk": {
        "noise_setting": ((0.2, 0.2), (0.3, 0.1), (0.4, 0.4)),
        "split_count": (3,),
        "corruption_repeats": (3,),
        "kernel_width": (1.0,),
    },
    "loss_correction": {
        "resnet_depth": (14, 32),
        "learning_rate": (0.01,),
        "milestones": ((40, 80),),
        "epochs": (120,),
        "weight_decay": (1e-4,),
    },
    "cwd": {
        "noise_setting": ((0.0, 0.0), (0.2, 0.2), (0.4, 0.4), (0.3, 0.1)),
        "optimizer": ("adam",),
        "learning_rate": (0.05,),
        "milestones": ((40, 120),),
        "epochs": (200,),
        "weight_decay": (1e-4,),
    },
    "fine": {
        "beta": (0.001,),
        "gamma": (0.1,),
    },
}


__all__ = [
    "PAPER_PARAMETER_CANDIDATES",
    "ParameterRecord",
    "resolve_parameter_sampling",
    "sample_parameters",
]
