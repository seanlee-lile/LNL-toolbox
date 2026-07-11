from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Component(Protocol):
    """Minimal lifecycle shared by replaceable framework components."""

    def setup(self, context: Any) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class Stateful(Protocol):
    """Optional checkpoint contract, separate from the basic component contract."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

