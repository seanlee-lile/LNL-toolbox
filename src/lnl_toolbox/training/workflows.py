from __future__ import annotations

"""Lazy registry for experiments with dedicated lifecycle runners."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from difflib import get_close_matches
from pathlib import Path
from typing import Any


WorkflowRunner = Callable[
    [dict[str, Any], str | Path | None, str | Path | None],
    Path,
]


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    module: str
    runner: str

    def load(self) -> WorkflowRunner:
        candidate = getattr(import_module(self.module), self.runner)
        if not callable(candidate):
            raise TypeError(f"workflow runner {self.module}.{self.runner} is not callable")
        return candidate


class WorkflowRegistry:
    """Map method names to lazily imported workflow runners."""

    def __init__(self) -> None:
        self._specs: dict[str, WorkflowSpec] = {}
        self._renamed: dict[str, str] = {}

    def add(self, name: str, module: str, runner: str) -> None:
        key = _normalize_name(name)
        if key in self._specs or key in self._renamed:
            raise KeyError(f"workflow {key!r} is already registered")
        self._specs[key] = WorkflowSpec(key, str(module), str(runner))

    def add_renamed(self, old_name: str, new_name: str) -> None:
        old_key = _normalize_name(old_name)
        new_key = _normalize_name(new_name)
        if old_key in self._specs or old_key in self._renamed:
            raise KeyError(f"workflow {old_key!r} is already registered")
        self._renamed[old_key] = new_key

    def resolve(self, name: str) -> WorkflowRunner | None:
        key = _normalize_name(name, allow_empty=True)
        if not key:
            return None
        if key in self._renamed:
            replacement = self._renamed[key]
            raise ValueError(f"method {key!r} was renamed to {replacement!r}")
        spec = self._specs.get(key)
        if spec is None:
            suggestion = get_close_matches(key, self.names(), n=1)
            hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
            raise ValueError(
                f"unknown method {key!r}{hint}; valid methods: " + ", ".join(self.names())
            )
        return spec.load()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


def _normalize_name(name: object, *, allow_empty: bool = False) -> str:
    value = str(name).strip().lower()
    if not value and not allow_empty:
        raise ValueError("workflow name must not be empty")
    return value


def method_name(config: Mapping[str, Any]) -> str:
    value = config.get("method")
    if isinstance(value, Mapping):
        value = value.get("name", "")
    return _normalize_name(value or "", allow_empty=True)


def create_workflow_registry() -> WorkflowRegistry:
    registry = WorkflowRegistry()
    registry.add(
        "coteaching",
        "lnl_toolbox.training.coteaching_experiment",
        "run_coteaching_experiment",
    )
    registry.add(
        "dual_t",
        "lnl_toolbox.training.dual_t_experiment",
        "run_dual_t_experiment",
    )
    registry.add(
        "importance_reweighting",
        "lnl_toolbox.training.importance_reweighting_experiment",
        "run_importance_reweighting_experiment",
    )
    registry.add(
        "pcse",
        "lnl_toolbox.training.pcse_experiment",
        "run_pcse_experiment",
    )
    registry.add_renamed("dual_t_forward", "dual_t")
    return registry


_BUILTIN_WORKFLOWS = create_workflow_registry()


def resolve_workflow(config: Mapping[str, Any]) -> WorkflowRunner | None:
    """Resolve a workflow without importing its runner until selected."""

    if not isinstance(config, Mapping):
        raise TypeError("experiment configuration must be a mapping")
    return _BUILTIN_WORKFLOWS.resolve(method_name(config))


__all__ = [
    "WorkflowRegistry",
    "WorkflowRunner",
    "WorkflowSpec",
    "create_workflow_registry",
    "method_name",
    "resolve_workflow",
]
