from __future__ import annotations

"""Task-neutral dotted-path configuration overrides."""

from copy import deepcopy
from difflib import get_close_matches
import json
import re
from typing import Any, Mapping, Sequence


_INTEGER = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(
    r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$"
)


def parse_override_value(text: str) -> Any:
    """Parse a CLI scalar/list without requiring a YAML dependency."""

    value = text.strip()
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if _INTEGER.fullmatch(value):
        return int(value)
    if _FLOAT.fullmatch(value):
        return float(value)
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid list override {text!r}; use JSON list syntax"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError(f"override value is not a list: {text!r}")
        return parsed
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _unknown_path(path: str, component: str, choices: Sequence[str]) -> ValueError:
    suggestion = get_close_matches(component, choices, n=1)
    hint = f" Did you mean {suggestion[0]}?" if suggestion else ""
    return ValueError(f"Unknown config path: {path}.{hint}")


def apply_override(config: Mapping[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Return a copy with one existing dotted mapping path replaced."""

    keys = [part.strip() for part in path.split(".")]
    if not keys or any(not key for key in keys):
        raise ValueError(f"invalid config path: {path!r}")
    result = deepcopy(dict(config))
    current: dict[str, Any] = result
    for key in keys[:-1]:
        if key not in current:
            raise _unknown_path(path, key, tuple(current))
        child = current[key]
        if not isinstance(child, dict):
            raise ValueError(
                f"Unknown config path: {path}; {key!r} is not a mapping"
            )
        current = child
    leaf = keys[-1]
    if leaf not in current:
        raise _unknown_path(path, leaf, tuple(current))
    current[leaf] = value
    return result


def apply_override_assignments(
    config: Mapping[str, Any], assignments: Sequence[str]
) -> dict[str, Any]:
    """Apply repeated ``path=value`` assignments in order."""

    result = deepcopy(dict(config))
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(
                f"invalid override {assignment!r}; expected path=value"
            )
        path, raw = assignment.split("=", 1)
        result = apply_override(result, path.strip(), parse_override_value(raw))
    return result


__all__ = ["apply_override", "apply_override_assignments", "parse_override_value"]
