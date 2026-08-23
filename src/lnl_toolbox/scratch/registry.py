"""Single registry for Scratch blocks and their user-facing metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


BlockCallable = Callable[..., Any]
BLOCKS: dict[str, "BlockDefinition"] = {}
_KINDS = {"action", "loop", "condition"}


@dataclass(frozen=True, slots=True)
class BlockDefinition:
    id: str
    name: str
    category: str
    description: str
    kind: str
    params: Mapping[str, Mapping[str, Any]]
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    execute: BlockCallable

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "kind": self.kind,
            "params": {key: dict(value) for key, value in self.params.items()},
            "requires": list(self.requires),
            "provides": list(self.provides),
        }


def register_block(definition: BlockDefinition) -> BlockDefinition:
    if not definition.id or not definition.id.replace("_", "").isalnum():
        raise ValueError("block id must be a non-empty identifier")
    if definition.kind not in _KINDS:
        raise ValueError("block kind must be action, loop, or condition")
    if not definition.name or not definition.category:
        raise ValueError("block name and category must not be empty")
    if not callable(definition.execute):
        raise TypeError("block execute must be callable")
    if definition.id in BLOCKS:
        raise ValueError(f"duplicate Scratch block id: {definition.id}")
    BLOCKS[definition.id] = definition
    return definition


def block(
    *,
    id: str,
    name: str,
    category: str,
    description: str = "",
    kind: str = "action",
    params: Mapping[str, Mapping[str, Any]] | None = None,
    requires: Iterable[str] = (),
    provides: Iterable[str] = (),
) -> Callable[[BlockCallable], BlockCallable]:
    def decorate(function: BlockCallable) -> BlockCallable:
        register_block(BlockDefinition(
            id=id,
            name=name,
            category=category,
            description=description,
            kind=kind,
            params=dict(params or {}),
            requires=tuple(requires),
            provides=tuple(provides),
            execute=function,
        ))
        return function

    return decorate


def get_block(block_id: str) -> BlockDefinition:
    try:
        return BLOCKS[block_id]
    except KeyError as exc:
        raise KeyError(f"unknown Scratch block: {block_id}") from exc


def list_blocks(category: str | None = None) -> tuple[BlockDefinition, ...]:
    values = BLOCKS.values()
    if category is not None:
        values = (value for value in values if value.category == category)
    return tuple(sorted(values, key=lambda value: (value.category, value.name, value.id)))


def describe_block(block_id: str) -> dict[str, Any]:
    return get_block(block_id).describe()

