from __future__ import annotations

"""Stateful, metric-agnostic early stopping for reproducible pipelines."""

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass
class EarlyStopping:
    """Track one validation metric and persist all stopping decisions."""

    monitor: str = "selection_accuracy"
    mode: str = "max"
    patience: int = 0
    min_delta: float = 0.0
    best: float | None = None
    bad_epochs: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        if self.mode not in {"min", "max"}:
            raise ValueError("early stopping mode must be 'min' or 'max'")
        if self.patience < 0:
            raise ValueError("early stopping patience must be non-negative")
        if not math.isfinite(float(self.min_delta)) or self.min_delta < 0:
            raise ValueError("early stopping min_delta must be finite and non-negative")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "EarlyStopping | None":
        if config in (None, False):
            return None
        if config is True:
            config = {}
        if not isinstance(config, Mapping):
            raise TypeError("early_stopping configuration must be a mapping or false")
        return cls(
            monitor=str(config.get("monitor", "selection_accuracy")),
            mode=str(config.get("mode", "max")),
            patience=int(config.get("patience", 10)),
            min_delta=float(config.get("min_delta", 0.0)),
        )

    def update(self, metrics: Mapping[str, Any]) -> bool:
        if self.monitor not in metrics:
            raise KeyError(f"early stopping metric {self.monitor!r} is missing")
        value = float(metrics[self.monitor])
        if not math.isfinite(value):
            raise ValueError("early stopping metric must be finite")
        improved = self.best is None or (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs > self.patience
        return improved

    def state_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "bad_epochs": self.bad_epochs,
            "stopped": self.stopped,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {"monitor", "mode", "patience", "min_delta", "best", "bad_epochs", "stopped"}
        if not isinstance(state, Mapping) or not expected.issubset(state):
            raise ValueError("early stopping state is incomplete")
        if str(state["monitor"]) != self.monitor or str(state["mode"]).lower() != self.mode:
            raise ValueError("early stopping configuration does not match checkpoint")
        if int(state["patience"]) != self.patience or float(state["min_delta"]) != self.min_delta:
            raise ValueError("early stopping configuration does not match checkpoint")
        self.best = None if state["best"] is None else float(state["best"])
        self.bad_epochs = int(state["bad_epochs"])
        self.stopped = bool(state["stopped"])
