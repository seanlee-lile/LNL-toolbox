"""Optional reference plugins; none are required by the framework core."""

from .catalog import (
    build_builtin_loss,
    build_builtin_selector,
    build_builtin_transition_estimator,
    create_builtin_catalog,
)

__all__ = [
    "build_builtin_loss",
    "build_builtin_selector",
    "build_builtin_transition_estimator",
    "create_builtin_catalog",
]

