"""Load and save the intentionally small Scratch recipe format."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def resolve_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with block defaults expanded for human-readable artifacts."""
    from .registry import get_block

    def visit(steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for step in steps:
            definition = get_block(str(step["block"]))
            params = dict(step.get("params", {}))
            for name, schema in definition.params.items():
                if name not in params and "default" in schema:
                    params[name] = schema["default"]
            item = {"block": step["block"], "params": params}
            if "steps" in step:
                item["steps"] = visit(step["steps"])
            result.append(item)
        return result

    output = deepcopy(dict(recipe))
    output["steps"] = visit(list(recipe.get("steps", [])))
    return output


def load_recipe(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Scratch recipe must contain a YAML mapping")
    return deepcopy(dict(payload))


def save_recipe(recipe: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(dict(recipe), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination
