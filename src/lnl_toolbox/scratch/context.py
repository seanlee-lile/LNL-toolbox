"""Shared slot table used by all Scratch blocks."""

from __future__ import annotations

from typing import Any


class ScratchContext(dict[str, Any]):
    """A deliberately small shared workbench for sequential blocks."""

