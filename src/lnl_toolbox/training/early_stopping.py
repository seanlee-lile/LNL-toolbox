from __future__ import annotations

"""Stateful, metric-agnostic early stopping for reproducible pipelines."""

from dataclasses import dataclass
import math
from typing import Any, Mapping
import json
from pathlib import Path


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


class StageEarlyStopping:
    """One stage, one boolean, exactly 15 consecutive validation failures.

    The caller owns phase transitions and the stage's best-model restoration.
    This object never stops a process or substitutes a training/test metric.
    """

    PATIENCE = 15

    def __init__(self, config: Mapping[str, Any], path: str, monitor: str,
                 *, mode: str = "max", run_dir: str | Path | None = None) -> None:
        value: Any = config
        for component in path.split("."):
            value = value.get(component, False) if isinstance(value, Mapping) else False
        if not isinstance(value, bool):
            raise TypeError(path + " must be a boolean")
        self.enabled = value
        self.path = path
        self.counter = EarlyStopping(monitor=monitor, mode=mode, patience=self.PATIENCE)
        self.history: list[dict[str, Any]] = []
        self.run_dir = None if run_dir is None else Path(run_dir)

    @property
    def stopped(self) -> bool:
        return self.enabled and self.counter.stopped

    def update(self, row: Mapping[str, Any], *, epoch: int | None = None,
               persist: bool = True) -> bool:
        if not self.enabled:
            return False
        if row.get("selection_split") == "test":
            raise ValueError(self.path + " cannot monitor test-set model selection")
        number = row.get("epoch") if epoch is None else epoch
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError(self.path + " requires a completed epoch number")
        if self.history and number <= self.history[-1]["epoch"]:
            return self.stopped
        self.counter.update(row)
        self.counter.stopped = self.counter.bad_epochs >= self.PATIENCE
        self.history.append({"epoch": number, "value": float(row[self.counter.monitor]),
                             "best": self.counter.best, "bad_epochs": self.counter.bad_epochs})
        if persist and self.run_dir is not None:
            destination = self.run_dir / "stage-early-stopping.json"
            document = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else {}
            document[self.path] = {"enabled": True, **self.counter.state_dict(), "history": self.history}
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(destination)
            if self.stopped:
                print(json.dumps({"event": "stage_early_stop", "stage": self.path.rsplit(".", 1)[0],
                                  "epoch": number, "monitor": self.counter.monitor,
                                  "best": self.counter.best, "bad_epochs": self.counter.bad_epochs}), flush=True)
        return self.stopped

    def replay(self, rows, *, event: str | None = None, phase: str | None = None,
               completed: int | None = None, zero_based: bool = False) -> None:
        if not self.enabled:
            return
        for row in rows:
            if event is not None and row.get("event", "epoch") != event:
                continue
            if phase is not None and row.get("stage", row.get("phase")) != phase:
                continue
            epoch = row.get("epoch")
            if not isinstance(epoch, int) or self.counter.monitor not in row:
                continue
            if completed is not None and epoch >= completed + (0 if zero_based else 1):
                continue
            self.update(row, persist=False)

    def replay_file(self, path: str | Path, **filters) -> None:
        source = Path(path)
        if not self.enabled or not source.is_file():
            return
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.replay(rows, **filters)

    def observe(self, row: Mapping[str, Any], components: Mapping[str, Any]) -> bool:
        """Checkpoint a paired stage optimum and restore it before advancing."""
        if not self.enabled:
            return False
        if self.run_dir is None:
            raise ValueError("stage early stopping requires a run directory")
        previous = self.counter.best
        stopped = self.update(row)
        if previous != self.counter.best:
            import torch
            destination = self.run_dir / (self.path.replace(".", "_") + "_best.pt")
            temporary = destination.with_suffix(".pt.tmp")
            torch.save({name: (component.detach().cpu().clone() if isinstance(component, torch.Tensor)
                               else component.state_dict()) for name, component in components.items()
                        if component is not None}, temporary)
            temporary.replace(destination)
        if stopped:
            self.restore_best(components)
        return stopped

    def restore_best(self, components: Mapping[str, Any]) -> None:
        if not self.enabled or self.run_dir is None or self.counter.best is None:
            return
        import torch
        source = self.run_dir / (self.path.replace(".", "_") + "_best.pt")
        if not source.is_file():
            raise RuntimeError("missing best stage checkpoint: " + str(source))
        payload = torch.load(source, map_location="cpu", weights_only=False)
        for name, component in components.items():
            if component is not None:
                if isinstance(component, torch.Tensor):
                    with torch.no_grad():
                        component.copy_(payload[name].to(component))
                else:
                    component.load_state_dict(payload[name])


def stage_epochs(start: int, total: int, warmup: int,
                 warmup_stopping: StageEarlyStopping, main_stopping: StageEarlyStopping):
    """Skip only the stopped phase; retain the subsequent phase's budget."""
    for epoch in range(start, min(warmup, total)):
        if warmup_stopping.stopped:
            break
        yield epoch
    for epoch in range(max(start, warmup), total):
        if main_stopping.stopped:
            break
        yield epoch


def forward_validation(model, transition, loader, device) -> dict[str, float]:
    """Validation on observed noisy labels for a jointly fitted transition."""
    import torch
    if loader is None:
        raise ValueError("stage early stopping requires a validation split")
    was_training = model.training
    model.eval()
    loss_sum, correct, samples = 0.0, 0, 0
    try:
        with torch.inference_mode():
            for batch in loader:
                targets = batch["target"].to(device)
                logits = model(batch["input"].to(device)).to(torch.float64)
                probabilities = logits.softmax(-1) @ transition.matrix().to(logits)
                probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
                loss_sum += float(torch.nn.functional.nll_loss(probabilities.log(), targets, reduction="sum"))
                correct += int(probabilities.argmax(-1).eq(targets).sum())
                samples += targets.numel()
    finally:
        model.train(was_training)
    if not samples:
        raise ValueError("stage early stopping requires nonempty validation data")
    return {"validation_loss": loss_sum / samples, "validation_accuracy": correct / samples}
