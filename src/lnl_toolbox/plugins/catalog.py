from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True, slots=True)
class PluginSpec:
    kind: str
    name: str
    factory: Callable[..., Any]
    capabilities: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PluginCatalog:
    """Registry keyed by component kind and name, with discoverable capabilities."""

    def __init__(self) -> None:
        self._plugins: dict[tuple[str, str], PluginSpec] = {}

    def add(
        self,
        kind: str,
        name: str,
        factory: Callable[..., Any],
        *,
        capabilities: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> PluginSpec:
        key = (kind.strip().lower(), name.strip().lower())
        if key in self._plugins:
            raise KeyError(f"Plugin {key[0]}/{key[1]} is already registered")
        spec = PluginSpec(key[0], key[1], factory, frozenset(capabilities), metadata or {})
        self._plugins[key] = spec
        return spec

    def get(self, kind: str, name: str) -> PluginSpec:
        key = (kind.strip().lower(), name.strip().lower())
        try:
            return self._plugins[key]
        except KeyError as exc:
            raise KeyError(f"Unknown plugin {key[0]}/{key[1]}") from exc

    def build(self, kind: str, name: str, /, **kwargs: Any) -> Any:
        return self.get(kind, name).factory(**kwargs)

    def find(self, *, kind: str | None = None, capability: str | None = None) -> tuple[PluginSpec, ...]:
        values = self._plugins.values()
        if kind is not None:
            values = (item for item in values if item.kind == kind.lower())
        if capability is not None:
            values = (item for item in values if capability in item.capabilities)
        return tuple(sorted(values, key=lambda item: (item.kind, item.name)))

