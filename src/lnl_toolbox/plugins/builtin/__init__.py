"""Optional reference plugins; none are required by the framework core."""

from .catalog import build_builtin_loss, create_builtin_catalog

__all__ = ["build_builtin_loss", "create_builtin_catalog"]

