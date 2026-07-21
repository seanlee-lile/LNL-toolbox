from __future__ import annotations

"""Validated class-conditional transition providers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

    from .manifest import NoiseManifest


def validate_transition_matrix(
    matrix: np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """Return a validated copy of ``T[i, j] = P(noisy=j | clean=i)``."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("transition_matrix must have square shape [C, C]")
    if num_classes is not None and values.shape != (num_classes, num_classes):
        raise ValueError(
            f"transition_matrix must have shape [{num_classes}, {num_classes}], "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("transition_matrix must contain only finite values")
    if (values < 0.0).any():
        raise ValueError("transition_matrix must be non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("every transition_matrix row must sum to one")
    return values.copy()


@runtime_checkable
class TransitionProvider(Protocol):
    """Provide a validated class-conditional transition matrix."""

    @property
    def num_classes(self) -> int:
        ...

    def as_tensor(self, *, device: Any = None, dtype: Any = None) -> "Tensor":
        ...


@dataclass(frozen=True, slots=True)
class KnownTransition:
    """Known/oracle transition matrix for controlled experiments."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", validate_transition_matrix(self.matrix))

    @property
    def num_classes(self) -> int:
        return int(self.matrix.shape[0])

    def as_tensor(self, *, device: Any = None, dtype: Any = None) -> "Tensor":
        import torch

        return torch.as_tensor(
            self.matrix,
            device=device,
            dtype=torch.float32 if dtype is None else dtype,
        )

    @classmethod
    def from_manifest(cls, manifest: "NoiseManifest") -> "KnownTransition":
        if manifest.transition_matrix is None:
            raise ValueError("Noise manifest does not contain a class transition matrix")
        return cls(manifest.transition_matrix)
