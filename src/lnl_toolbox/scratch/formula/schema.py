"""Serializable schema for Scratch formulas.

The schema intentionally models a sequential list of Registry operations,
not an expression language or a separate graph/IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


@dataclass
class FormulaInputSpec:
    name: str
    description: str = ""
    type: str = "tensor"

    @classmethod
    def from_value(cls, name: str, value: Any) -> "FormulaInputSpec":
        if isinstance(value, Mapping):
            return cls(name=name, description=str(value.get("description", "")), type=str(value.get("type", "tensor")))
        return cls(name=name, description=str(value or ""))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"description": self.description}
        if self.type != "tensor":
            result["type"] = self.type
        return result


@dataclass
class FormulaParameterSpec:
    name: str
    type: str = "float"
    default: Any = None
    description: str = ""
    minimum: float | int | None = None
    maximum: float | int | None = None
    options: list[Any] | None = None

    @classmethod
    def from_value(cls, name: str, value: Any) -> "FormulaParameterSpec":
        if not isinstance(value, Mapping):
            return cls(name=name, default=value)
        options = value.get("options")
        return cls(
            name=name,
            type=str(value.get("type", "float")),
            default=value.get("default"),
            description=str(value.get("description", "")),
            minimum=value.get("minimum", value.get("min")),
            maximum=value.get("maximum", value.get("max")),
            options=list(options) if isinstance(options, (list, tuple)) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type, "default": self.default}
        if self.description:
            result["description"] = self.description
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.options is not None:
            result["options"] = list(self.options)
        return result


@dataclass
class FormulaStepSpec:
    id: str
    block: str
    bindings: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "FormulaStepSpec":
        if not isinstance(value, Mapping):
            raise TypeError("formula step must be a mapping")
        return cls(
            id=str(value.get("id", "")),
            block=str(value.get("block", "")),
            bindings=_mapping(value.get("bindings", {}), "step.bindings"),
            parameters=_mapping(value.get("parameters", {}), "step.parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.id, "block": self.block}
        if self.bindings:
            result["bindings"] = dict(self.bindings)
        if self.parameters:
            result["parameters"] = dict(self.parameters)
        return result


@dataclass
class FormulaOutputSpec:
    name: str
    source: str
    description: str = ""

    @classmethod
    def from_value(cls, name: str, value: Any) -> "FormulaOutputSpec":
        if isinstance(value, Mapping):
            return cls(name=name, source=str(value.get("source", "")), description=str(value.get("description", "")))
        return cls(name=name, source=str(value))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"source": self.source}
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class FormulaSpec:
    id: str
    name: str
    description: str = ""
    inputs: dict[str, FormulaInputSpec] = field(default_factory=dict)
    parameters: dict[str, FormulaParameterSpec] = field(default_factory=dict)
    steps: list[FormulaStepSpec] = field(default_factory=list)
    outputs: dict[str, FormulaOutputSpec] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormulaSpec":
        if not isinstance(value, Mapping):
            raise TypeError("FormulaSpec must be a mapping")
        raw_inputs = _mapping(value.get("inputs", {}), "inputs")
        raw_parameters = _mapping(value.get("parameters", {}), "parameters")
        raw_outputs = _mapping(value.get("outputs", {}), "outputs")
        raw_steps = value.get("steps", [])
        if not isinstance(raw_steps, list):
            raise TypeError("steps must be a list")
        return cls(
            id=str(value.get("id", "")),
            name=str(value.get("name", "")),
            description=str(value.get("description", "")),
            inputs={name: FormulaInputSpec.from_value(name, item) for name, item in raw_inputs.items()},
            parameters={name: FormulaParameterSpec.from_value(name, item) for name, item in raw_parameters.items()},
            steps=[FormulaStepSpec.from_value(item) for item in raw_steps],
            outputs={name: FormulaOutputSpec.from_value(name, item) for name, item in raw_outputs.items()},
            metadata=_mapping(value.get("metadata", {}), "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "inputs": {name: item.to_dict() for name, item in self.inputs.items()},
            "parameters": {name: item.to_dict() for name, item in self.parameters.items()},
            "steps": [step.to_dict() for step in self.steps],
            "outputs": {name: item.to_dict() for name, item in self.outputs.items()},
            "metadata": dict(self.metadata),
        }

