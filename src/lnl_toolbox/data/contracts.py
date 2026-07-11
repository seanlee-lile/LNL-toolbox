from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Sample:
    """Stable dataset protocol; clean fields are evaluator-only."""

    image: Any
    target: int
    index: int
    clean_target: int | None = None
    is_clean: bool | None = None

    def training_view(self) -> dict[str, Any]:
        return {"image": self.image, "target": self.target, "index": self.index}

