from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class ExperimentContext:
    """Runtime dependencies supplied to components without global imports."""

    work_dir: Path
    config: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    services: dict[str, Any] = field(default_factory=dict)

    def service(self, name: str) -> Any:
        try:
            return self.services[name]
        except KeyError as exc:
            raise KeyError(f"Experiment service {name!r} is not configured") from exc

