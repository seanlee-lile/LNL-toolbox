from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    """Small explicit registry used by datasets, losses, and algorithms."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        key = name.strip().lower()

        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            if key in self._items:
                raise KeyError(f"{key!r} is already registered in {self.name}")
            self._items[key] = factory
            return factory

        return decorator

    def build(self, name: str, /, **kwargs: Any) -> Any:
        key = name.strip().lower()
        try:
            factory = self._items[key]
        except KeyError as exc:
            choices = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.name} {name!r}; choices: {choices}") from exc
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

