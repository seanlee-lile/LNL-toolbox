from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Batch:
    """Opaque algorithm input plus optional framework-readable metadata.

    The payload may be tensors, arrays, graphs, text, or a task-defined object. The
    core deliberately does not require labels, sample indices, or a batch dimension.
    """

    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

