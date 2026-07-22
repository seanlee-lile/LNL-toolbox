"""Stateless keep-rate schedules for generic batch selectors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Protocol, runtime_checkable


def _validate_rate(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 < rate <= 1.0:
        raise ValueError(f"{name} must be finite and in the interval (0, 1]")
    return rate


def _validate_epoch(epoch: Any) -> int:
    if isinstance(epoch, bool) or not isinstance(epoch, Integral):
        raise TypeError("epoch must be a zero-based integer")
    epoch = int(epoch)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return epoch


@runtime_checkable
class KeepRateSchedule(Protocol):
    """Return a keep rate for a zero-based epoch without storing runtime state."""

    def rate_at(self, epoch: int) -> float:
        """Return the keep rate for ``epoch``."""


@dataclass(frozen=True)
class ConstantKeepRateSchedule:
    """Use one keep rate for every epoch."""

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_rate(self.value, "value"))

    def rate_at(self, epoch: int) -> float:
        _validate_epoch(epoch)
        return self.value


@dataclass(frozen=True)
class LinearKeepRateSchedule:
    """Interpolate from ``start`` to ``end`` over zero-based warmup epochs."""

    start: float
    end: float
    warmup_epochs: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _validate_rate(self.start, "start"))
        object.__setattr__(self, "end", _validate_rate(self.end, "end"))
        if isinstance(self.warmup_epochs, bool) or not isinstance(
            self.warmup_epochs, Integral
        ):
            raise TypeError("warmup_epochs must be a positive integer")
        if int(self.warmup_epochs) <= 0:
            raise ValueError("warmup_epochs must be a positive integer")
        object.__setattr__(self, "warmup_epochs", int(self.warmup_epochs))

    def rate_at(self, epoch: int) -> float:
        epoch = _validate_epoch(epoch)
        progress = min(epoch / self.warmup_epochs, 1.0)
        return self.start + progress * (self.end - self.start)


def build_keep_rate_schedule(config: Any) -> KeepRateSchedule:
    """Normalize a fixed number or YAML-compatible mapping into a schedule."""

    if isinstance(config, Real) and not isinstance(config, bool):
        return ConstantKeepRateSchedule(config)
    if not isinstance(config, Mapping):
        raise TypeError("keep_rate must be a real number or a schedule mapping")

    values = dict(config)
    name = str(values.pop("name", "")).strip().lower()
    if name == "constant":
        if set(values) != {"value"}:
            raise ValueError("constant keep-rate schedule requires only value")
        return ConstantKeepRateSchedule(values["value"])
    if name == "linear":
        if set(values) != {"start", "end", "warmup_epochs"}:
            raise ValueError(
                "linear keep-rate schedule requires only start, end, and warmup_epochs"
            )
        return LinearKeepRateSchedule(
            start=values["start"],
            end=values["end"],
            warmup_epochs=values["warmup_epochs"],
        )
    raise ValueError("keep-rate schedule name must be constant or linear")
