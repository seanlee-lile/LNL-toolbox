from __future__ import annotations

"""Shared metadata and readiness states for external prerequisites.

This module deliberately does not know how PCSE, DLD, or CAL consume an
artifact.  Method-specific modules register validators that return the common
readiness record defined here.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class ReadinessLevel(IntEnum):
    CONFIGURED = 1
    PATH_EXISTS = 2
    SCHEMA_VALID = 3
    IDENTITY_COMPATIBLE = 4
    CONSUMER_VALIDATED = 5


class ReadinessStatus(str):
    NEEDS_INPUT = "needs_input"
    INVALID = "invalid"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    key: str
    kind: str
    name: str
    requirement: str
    obtain: str
    provide: str
    validator: str
    supported_sources: tuple[str, ...] = ()
    config_paths: tuple[tuple[str, ...], ...] = ()
    environment_variable: str | None = None
    clean_data_usage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "name": self.name,
            "requirement": self.requirement,
            "obtain": self.obtain,
            "provide": self.provide,
            "supported_sources": list(self.supported_sources),
            "config_paths": [list(path) for path in self.config_paths],
            "environment_variable": self.environment_variable,
            "clean_data_usage": self.clean_data_usage,
        }


@dataclass(frozen=True, slots=True)
class ValidationMetadata:
    descriptor: SourceDescriptor
    level: ReadinessLevel | None
    status: str
    message: str
    configured: bool = False
    path_exists: bool = False
    schema_valid: bool = False
    identity_compatible: bool = False
    consumer_validated: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def needs_input(
        cls, descriptor: SourceDescriptor, message: str
    ) -> "ValidationMetadata":
        return cls(descriptor, None, ReadinessStatus.NEEDS_INPUT, message)

    @classmethod
    def invalid(
        cls,
        descriptor: SourceDescriptor,
        level: ReadinessLevel,
        message: str,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> "ValidationMetadata":
        return cls(
            descriptor,
            level,
            ReadinessStatus.INVALID,
            message,
            configured=level >= ReadinessLevel.CONFIGURED,
            path_exists=level >= ReadinessLevel.PATH_EXISTS,
            schema_valid=level >= ReadinessLevel.SCHEMA_VALID,
            identity_compatible=level >= ReadinessLevel.IDENTITY_COMPATIBLE,
            consumer_validated=False,
            provenance=dict(provenance or {}),
        )

    @classmethod
    def ready(
        cls,
        descriptor: SourceDescriptor,
        message: str,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> "ValidationMetadata":
        return cls(
            descriptor,
            ReadinessLevel.CONSUMER_VALIDATED,
            ReadinessStatus.READY,
            message,
            configured=True,
            path_exists=True,
            schema_valid=True,
            identity_compatible=True,
            consumer_validated=True,
            provenance=dict(provenance or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.descriptor.to_dict(),
            "status": self.status,
            "level": None if self.level is None else self.level.name,
            "message": self.message,
            "configured": self.configured,
            "path_exists": self.path_exists,
            "schema_valid": self.schema_valid,
            "identity_compatible": self.identity_compatible,
            "consumer_validated": self.consumer_validated,
            "provenance": dict(self.provenance),
        }


PrerequisiteValidator = Callable[
    [SourceDescriptor, Mapping[str, Any]], ValidationMetadata
]


class PrerequisiteRegistry:
    def __init__(self) -> None:
        self._validators: dict[str, PrerequisiteValidator] = {}

    def register(self, name: str, validator: PrerequisiteValidator) -> None:
        key = str(name).strip()
        if not key or key in self._validators:
            raise ValueError(f"duplicate or empty prerequisite validator: {key!r}")
        self._validators[key] = validator

    def evaluate(
        self, descriptor: SourceDescriptor, config: Mapping[str, Any]
    ) -> ValidationMetadata:
        try:
            validator = self._validators[descriptor.validator]
        except KeyError as error:
            raise KeyError(
                f"prerequisite validator is not registered: {descriptor.validator}"
            ) from error
        result = validator(descriptor, config)
        if result.descriptor != descriptor:
            raise ValueError("prerequisite validator returned the wrong descriptor")
        return result


__all__ = [
    "PrerequisiteRegistry",
    "PrerequisiteValidator",
    "ReadinessLevel",
    "ReadinessStatus",
    "SourceDescriptor",
    "ValidationMetadata",
]
