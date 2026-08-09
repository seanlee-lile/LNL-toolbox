from __future__ import annotations

"""Unified Python API for all catalogued LNL experiments."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Type

from .interfaces import EvaluationResult, ExperimentRunner, RunContext, RunResult


_METHOD_TO_RUNNER = {
    "apl": "supervised",
    "gce": "supervised",
    "dss": "supervised",
    "cdr": "supervised",
    "loss_correction": "supervised",
    "loss-correction": "supervised",
    "jocor": "multi_model",
    "coteaching": "coteaching",
    "binary_risk": "binary",
    "binary-risk": "binary",
    "natarajan": "binary",
    "pdl": "instance_transition",
    "cwd": "cwd",
    "fine": "fine",
    "mentornet": "supervised",
    "dividemix": "dividemix",
    "volmin": "volmin",
    "volminnet": "volminnet",
    "dld": "dld",
    "upm": "upm",
    "lend": "lend",
}


class Toolbox:
    """Registry and execution facade used by both Python and CLI callers."""

    def __init__(self) -> None:
        self._custom: dict[str, tuple[Type[Any], dict[str, Any]]] = {}

    def register(
        self,
        name: str,
        runner: Type[Any],
        *,
        lifecycle: str = "single_stage",
        checkpoint_unit: str = "epoch",
    ) -> None:
        key = self._normalize(name)
        self._custom[key] = (
            runner,
            {"lifecycle": lifecycle, "checkpoint_unit": checkpoint_unit},
        )

    def get(self, name: str, *, config: Mapping[str, Any] | None = None) -> ExperimentRunner:
        key = self._normalize(name)
        if key in self._custom:
            runner_class, metadata = self._custom[key]
            return runner_class(name=key, metadata=metadata)
        from .runners import create_runner_registry
        registry = create_runner_registry()
        runner_name = _METHOD_TO_RUNNER.get(key, key)
        if config is not None:
            from .runners import resolve_runner

            runner_name = resolve_runner(config).name
        spec = registry.get(runner_name)
        return spec.build(method=key if key != runner_name else None)

    def run(
        self,
        method: str | None = None,
        config: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> RunResult:
        if config is None:
            raise TypeError("config is required")
        selected = method or self._config_method(config)
        runner = self.get(selected, config=config)
        return runner.fit(config=config, output_dir=output_dir, resume=resume)

    @staticmethod
    def evaluate(runner: ExperimentRunner, result: RunResult) -> EvaluationResult:
        return runner.evaluate(result)

    @staticmethod
    def _normalize(value: object) -> str:
        return str(value).strip().lower().replace("-", "_")

    @staticmethod
    def _config_method(config: Mapping[str, Any]) -> str:
        value = config.get("method", "")
        if isinstance(value, Mapping):
            value = value.get("name", "")
        return str(value).strip() or "supervised"


toolbox = Toolbox()


__all__ = ["Toolbox", "toolbox"]
