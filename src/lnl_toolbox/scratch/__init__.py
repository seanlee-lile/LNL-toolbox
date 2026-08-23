"""Independent, recipe-driven LNL scratch builder."""

from .context import ScratchContext
from .executor import ScratchExecutionError, execute_recipe, execute_steps
from .recipe import load_recipe, resolve_recipe, save_recipe
from .registry import BLOCKS, describe_block, get_block, list_blocks
from .validation import ScratchValidationError, validate_recipe

__all__ = [
    "BLOCKS",
    "ScratchContext",
    "ScratchExecutionError",
    "ScratchValidationError",
    "describe_block",
    "execute_recipe",
    "execute_steps",
    "get_block",
    "list_blocks",
    "load_recipe",
    "save_recipe",
    "validate_recipe",
]
