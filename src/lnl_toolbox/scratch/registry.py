"""Single registry for Scratch blocks and their user-facing metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping


BlockCallable = Callable[..., Any]
BLOCKS: dict[str, "BlockDefinition"] = {}
_KINDS = {"action", "loop", "condition"}
_PLACEMENTS = {"top", "epoch", "batch", "any"}
_STAGES = {"data", "setup", "train", "evaluate"}


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
    placement: tuple[str, ...] = ("any",)
    stage: str = "train"
    ui_group: str = ""
    formula: str | None = None
    formula_ref: str | None = None
    paper: str | None = None
    beginner_visible: bool = True
    formula_safe: bool = False
    formula_group: str | None = None

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
            "placement": list(self.placement),
            "stage": self.stage,
            "ui_group": self.ui_group,
            "formula": self.formula,
            "formula_ref": self.formula_ref,
            "paper": self.paper,
            "beginner_visible": self.beginner_visible,
            "formula_safe": self.formula_safe,
            "formula_group": self.formula_group,
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
    if not definition.placement or not set(definition.placement).issubset(_PLACEMENTS):
        raise ValueError(f"invalid Scratch block placement: {definition.id}")
    if definition.stage not in _STAGES:
        raise ValueError(f"invalid Scratch block stage: {definition.id}")
    # Legacy/paper blocks that have not opted into the V2 palette metadata must
    # stay out of the beginner palette until they receive an explicit UI group.
    if definition.beginner_visible and not definition.ui_group:
        definition = replace(definition, beginner_visible=False)
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
    placement: Iterable[str] = ("any",),
    stage: str = "train",
    ui_group: str = "",
    formula: str | None = None,
    formula_ref: str | None = None,
    paper: str | None = None,
    beginner_visible: bool = True,
    formula_safe: bool | None = None,
    formula_group: str | None = None,
) -> Callable[[BlockCallable], BlockCallable]:
    def decorate(function: BlockCallable) -> BlockCallable:
        # Existing canonical tensor/loss/selection operations predate the
        # explicit flag.  Infer a conservative default for those operations,
        # while allowing stateful/model/data blocks to remain opt-in only.
        safe_default = (
            kind == "action"
            and paper is None
            and category in {"Forward", "Tensor Operation", "Loss", "Correction", "Transition", "Sample Selection", "Weighting"}
            and any(place in {"batch", "any"} for place in placement)
            and id not in {
                "forward", "module_forward", "forward_feature", "compose_revision_transition",
                "classwise_percentile_anchor_candidates",
            }
        )
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
            placement=tuple(placement),
            stage=stage,
            ui_group=ui_group,
            formula=formula,
            formula_ref=formula_ref,
            paper=paper,
            beginner_visible=beginner_visible,
            formula_safe=safe_default if formula_safe is None else bool(formula_safe),
            formula_group=formula_group or (category.lower().replace(" ", "_") if safe_default else None),
        ))
        return function

    return decorate


def get_block(block_id: str) -> BlockDefinition:
    # Formula YAML uses the readable ``formula/user/name`` identifier while
    # the single Block Registry stores a filesystem-safe internal id.
    if block_id.startswith("formula/"):
        internal_id = "formula__" + block_id.removeprefix("formula/").replace("/", "__")
        if internal_id in BLOCKS:
            return BLOCKS[internal_id]
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
