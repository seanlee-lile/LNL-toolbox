from __future__ import annotations

"""Central lazy registry for user-facing experiment execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from importlib import import_module
from pathlib import Path
from typing import Any, Callable


Runner = Callable[[dict[str, Any], str | Path | None, str | Path | None], Path]


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    name: str
    module: str
    function: str
    supports_resume: bool = True

    def load(self) -> Callable[..., Path]:
        candidate = getattr(import_module(self.module), self.function)
        if not callable(candidate):
            raise TypeError(f"runner {self.module}.{self.function} is not callable")
        return candidate

    def invoke(
        self,
        config: dict[str, Any],
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> Path:
        if resume is not None and not self.supports_resume:
            raise ValueError(f"runner {self.name!r} does not support resume")
        runner = self.load()
        if self.supports_resume:
            return runner(config, output_dir, resume)
        return runner(config, output_dir)


class RunnerRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, RunnerSpec] = {}

    def add(
        self,
        name: str,
        module: str,
        function: str,
        *,
        supports_resume: bool = True,
    ) -> None:
        key = _normalize(name)
        if key in self._specs:
            raise KeyError(f"runner {key!r} is already registered")
        self._specs[key] = RunnerSpec(key, module, function, supports_resume)

    def get(self, name: str) -> RunnerSpec:
        key = _normalize(name)
        try:
            return self._specs[key]
        except KeyError as exc:
            suggestion = get_close_matches(key, self.names(), n=1)
            hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
            raise ValueError(
                f"unknown execution.runner {key!r}{hint}; valid runners: "
                + ", ".join(self.names())
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


def _normalize(value: object) -> str:
    name = str(value).strip().lower().replace("-", "_")
    if not name:
        raise ValueError("runner name must not be empty")
    return name


def create_runner_registry() -> RunnerRegistry:
    registry = RunnerRegistry()
    registry.add("supervised", "lnl_toolbox.training.experiment", "run_supervised_experiment")
    registry.add("clean", "lnl_toolbox.training.clean_baseline", "run_clean_experiment")
    registry.add("multi_model", "lnl_toolbox.training.multi_model_experiment", "run_multi_model_experiment")
    registry.add("cwd", "lnl_toolbox.training.cwd_experiment", "run_cwd_experiment")
    registry.add("fine", "lnl_toolbox.training.fine_experiment", "run_fine_experiment")
    registry.add("binary", "lnl_toolbox.training.binary_experiment", "run_binary_experiment", supports_resume=False)
    registry.add(
        "instance_transition",
        "lnl_toolbox.training.instance_transition_experiment",
        "run_instance_transition_experiment",
    )
    registry.add("coteaching", "lnl_toolbox.training.coteaching_experiment", "run_coteaching_experiment")
    registry.add("dual_t", "lnl_toolbox.training.dual_t_experiment", "run_dual_t_experiment")
    registry.add(
        "importance_reweighting",
        "lnl_toolbox.training.importance_reweighting_experiment",
        "run_importance_reweighting_experiment",
    )
    registry.add("pcse", "lnl_toolbox.training.pcse_experiment", "run_pcse_experiment")
    registry.add(
        "volminnet",
        "lnl_toolbox.training.volminnet_experiment",
        "run_volminnet_experiment",
    )
    registry.add("upm", "lnl_toolbox.training.upm_experiment", "run_upm_experiment")
    registry.add("dld", "lnl_toolbox.training.dld_experiment", "run_dld_experiment")
    registry.add("dividemix", "lnl_toolbox.training.dividemix_experiment", "run_dividemix_experiment")
    registry.add("cnlcu", "lnl_toolbox.training.cnlcu_experiment", "run_cnlcu_experiment")
    registry.add(
        "t_revision",
        "lnl_toolbox.training.t_revision_experiment",
        "run_t_revision_experiment",
    )
    return registry


_RUNNERS = create_runner_registry()
_METHOD_RUNNERS = frozenset(
    {
        "cnlcu",
        "coteaching",
        "dual_t",
        "dld",
        "dividemix",
        "importance_reweighting",
        "pcse",
        "t_revision",
        "upm",
        "volminnet",
    }
)
_RENAMED_METHODS = {"dual_t_forward": "dual_t"}
_DEDICATED_SECTIONS = {
    "cwd": "cwd",
    "fine": "fine",
    "instance_transition": "instance_transition",
}


def runner_names() -> tuple[str, ...]:
    return _RUNNERS.names()


def method_names() -> tuple[str, ...]:
    """Return public method names owned by the central runner registry."""

    return tuple(sorted(_METHOD_RUNNERS))


def _set_nested_epoch(config: dict[str, Any], path: tuple[str, ...], epochs: int) -> None:
    current: dict[str, Any] = config
    for key in path[:-1]:
        value = current.get(key)
        if not isinstance(value, dict):
            raise ValueError(
                f"--epochs requires an existing {'.'.join(path[:-1])} mapping"
            )
        current = value
    current[path[-1]] = epochs


def apply_epoch_override(config: dict[str, Any], epochs: int) -> None:
    """Apply a user epoch target only where its lifecycle meaning is unambiguous."""

    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    runner = resolve_runner(config)
    method_value = config.get("method", "")
    if isinstance(method_value, Mapping):
        method_value = method_value.get("name", "")
    method = _normalize(method_value) if str(method_value).strip() else ""
    if method == "t_revision":
        _set_nested_epoch(config, ("t_revision", "revision", "epochs"), epochs)
        return
    if method == "upm":
        _set_nested_epoch(config, ("upm", "main", "epochs"), epochs)
        return
    if method == "dld":
        _set_nested_epoch(config, ("dld", "diffusion", "epochs"), epochs)
        return
    if method == "dividemix":
        _set_nested_epoch(config, ("dividemix", "training", "epochs"), epochs)
        return
    if method in {"dual_t", "pcse"}:
        raise ValueError(
            f"--epochs is ambiguous for staged method {method!r}; edit the explicit "
            "stage epoch field in the YAML instead"
        )
    if runner.name == "binary":
        raise ValueError("--epochs is not supported by the non-resumable binary runner")
    trainer = config.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("--epochs requires an existing trainer mapping")
    trainer["epochs"] = epochs


def resolve_runner(config: Mapping[str, Any]) -> RunnerSpec:
    if not isinstance(config, Mapping):
        raise TypeError("experiment configuration must be a mapping")

    execution = config.get("execution", {}) or {}
    if not isinstance(execution, Mapping):
        raise TypeError("execution configuration must be a mapping")
    explicit = str(execution.get("runner", "")).strip()

    method_value = config.get("method", "")
    if isinstance(method_value, Mapping):
        method_value = method_value.get("name", "")
    method = _normalize(method_value) if str(method_value).strip() else ""
    if method in _RENAMED_METHODS:
        raise ValueError(f"method {method!r} was renamed to {_RENAMED_METHODS[method]!r}")
    if method and method not in _METHOD_RUNNERS:
        suggestion = get_close_matches(method, sorted(_METHOD_RUNNERS), n=1)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(
            f"Unsupported training method: unknown method {method!r}{hint}; "
            "run 'lnl list experiments' to see runnable methods"
        )

    inferred = method
    for section, runner in _DEDICATED_SECTIONS.items():
        if section in config:
            if inferred and inferred != runner:
                raise ValueError(f"configuration selects both {inferred!r} and dedicated section {section!r}")
            inferred = runner
    algorithm = config.get("algorithm", {}) or {}
    if isinstance(algorithm, Mapping) and str(algorithm.get("name", "")).strip().lower() == "jocor":
        if inferred and inferred != "multi_model":
            raise ValueError("JoCoR configuration conflicts with another runner")
        inferred = "multi_model"

    selected = _normalize(explicit) if explicit else (inferred or "supervised")
    if explicit and inferred and selected != inferred:
        raise ValueError(
            f"execution.runner {selected!r} conflicts with configuration requiring {inferred!r}"
        )
    return _RUNNERS.get(selected)


__all__ = [
    "RunnerRegistry",
    "RunnerSpec",
    "apply_epoch_override",
    "create_runner_registry",
    "method_names",
    "resolve_runner",
    "runner_names",
]
