from __future__ import annotations

"""Compatibility facade over the central :mod:`training.runners` registry."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
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
    """Legacy workflow API; built-ins are resolved by ``RunnerRegistry``."""

    def __init__(self, *, include_builtins: bool = False) -> None:
        self._specs: dict[str, WorkflowSpec] = {}
        self._renamed: dict[str, str] = {}
        self._include_builtins = include_builtins

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
        if spec is not None:
            return spec.load()
        if self._include_builtins:
            from lnl_toolbox.training.runners import resolve_runner

            return resolve_runner({"method": key}).load()
        raise ValueError(
            f"unknown method {key!r}; valid methods: " + ", ".join(self.names())
        )

    def names(self) -> tuple[str, ...]:
        names = set(self._specs)
        if self._include_builtins:
            from lnl_toolbox.training.runners import method_names

            names.update(method_names())
        return tuple(sorted(names))


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
    registry = WorkflowRegistry(include_builtins=True)
    registry.add_renamed("dual_t_forward", "dual_t")
    return registry


_BUILTIN_WORKFLOWS = create_workflow_registry()


def resolve_workflow(config: Mapping[str, Any]) -> WorkflowRunner | None:
    """Resolve a workflow without importing its runner until selected."""

    if not isinstance(config, Mapping):
        raise TypeError("experiment configuration must be a mapping")
    name = method_name(config)
    return _BUILTIN_WORKFLOWS.resolve(name) if name else None


__all__ = [
    "WorkflowRegistry",
    "WorkflowRunner",
    "WorkflowSpec",
    "create_workflow_registry",
    "method_name",
    "resolve_workflow",
]
