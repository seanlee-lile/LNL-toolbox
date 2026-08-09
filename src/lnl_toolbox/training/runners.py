from __future__ import annotations

"""Central lazy registry for user-facing experiment execution."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from difflib import get_close_matches
from importlib import import_module
from pathlib import Path
from typing import Any, Callable



Runner = Callable[[dict[str, Any], str | Path | None, str | Path | None], Path]


def _method_name(config: Mapping[str, Any], fallback: str) -> str:
    value = config.get("method", fallback)
    if isinstance(value, Mapping):
        value = value.get("name", fallback)
    return str(value or fallback)


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    name: str
    module: str
    function: str
    supports_resume: bool = True
    lifecycle: str = "single_stage"
    checkpoint_unit: str = "epoch"
    smoke_recipe: str | None = None

    def load(self) -> Callable[..., Path]:
        candidate = getattr(import_module(self.module), self.function)
        if not callable(candidate):
            raise TypeError(f"runner {self.module}.{self.function} is not callable")
        return candidate

    def build(self, *, method: str | None = None):
        """Build the public protocol object for this legacy runner."""

        from lnl_toolbox.training.adapters import adapter_class_for

        return adapter_class_for(self, method=method)(self, method=method)

    def invoke(
        self,
        config: dict[str, Any],
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> Path:
        if resume is not None and not self.supports_resume:
            raise ValueError(f"runner {self.name!r} does not support resume")
        result = self.build(method=_method_name(config, self.name)).fit(
            config=config,
            output_dir=output_dir,
            resume=resume,
        )
        return result.run_dir


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
    registry.add("binary", "lnl_toolbox.training.binary_experiment", "run_binary_experiment")
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
    registry.add("mc_ldce", "lnl_toolbox.training.mc_ldce_experiment", "run_mc_ldce_experiment")
    registry.add("cal", "lnl_toolbox.training.cal_experiment", "run_cal_experiment")
    registry.add("ca2c", "lnl_toolbox.training.ca2c_experiment", "run_ca2c_experiment")
    registry.add("l2rw", "lnl_toolbox.training.l2rw_experiment", "run_l2rw_experiment")
    registry.add("dld", "lnl_toolbox.training.dld_experiment", "run_dld_experiment")
    registry.add("cnlcu", "lnl_toolbox.training.cnlcu_experiment", "run_cnlcu_experiment")
    registry.add(
        "t_revision",
        "lnl_toolbox.training.t_revision_experiment",
        "run_t_revision_experiment",
    )
    registry.add("volmin", "lnl_toolbox.training.volminnet_experiment", "run_volminnet_experiment")
    registry.add("upm", "lnl_toolbox.training.upm_experiment", "run_upm_experiment")
    registry.add("lend", "lnl_toolbox.training.lend_experiment", "run_lend_experiment")
    registry.add("dividemix", "lnl_toolbox.training.dividemix_experiment", "run_dividemix_experiment")
    registry.add("volminnet", "lnl_toolbox.training.volminnet_experiment", "run_volminnet_experiment")
    profiles = {
        "supervised": ("single_stage", "epoch", "cifar10-symmetric-ce-smoke"),
        "clean": ("single_stage", "epoch", "cifar10-clean-smoke"),
        "multi_model": ("multi_model", "epoch", "jocor-cifar10-symmetric05-smoke"),
        "cwd": ("binary_fold", "epoch", "cwd-cifar10-smoke"),
        "fine": ("two_stage", "epoch", "fine-cifar100n-smoke"),
        "binary": ("single_stage", "epoch", "binary-risk-natarajan-1epoch"),
        "instance_transition": ("staged", "epoch", "pdl-cifar10-smoke"),
        "coteaching": ("multi_model", "epoch", "cifar10-coteaching-smoke"),
        "dual_t": ("staged", "epoch", "cifar10-dual-t-smoke"),
        "importance_reweighting": ("staged", "epoch", "importance-reweighting-binary-smoke"),
        "pcse": ("staged", "epoch", "pcse-multiclass-smoke"),
        "mc_ldce": ("staged", "epoch", "mc-ldce-cifar10-smoke"),
        "cal": ("staged", "epoch", "cal-cifar10-smoke"),
        "ca2c": ("staged", "epoch", "ca2c-cifar10-smoke"),
        "l2rw": ("staged", "step", "l2rw-cifar10-smoke"),
        "dld": ("staged", "epoch", "dld-cifar10-smoke"),
        "cnlcu": ("multi_model", "epoch", "cifar10-cnlcu-soft-smoke"),
        "t_revision": ("staged", "epoch", "cifar10-t-revision-smoke"),
        "volmin": ("staged", "epoch", "volmin-cifar10-smoke"),
        "upm": ("staged", "epoch", "upm-cifar10-smoke"),
        "lend": ("staged", "epoch", "lend-cifar10-smoke"),
        "dividemix": ("multi_model", "epoch", "cifar10-dividemix-smoke"),
        "volminnet": ("dual_optimizer", "epoch", "cifar10-volminnet-smoke"),
    }
    for name, (lifecycle, checkpoint_unit, smoke_recipe) in profiles.items():
        current = registry._specs[name]
        registry._specs[name] = replace(
            current,
            lifecycle=lifecycle,
            checkpoint_unit=checkpoint_unit,
            smoke_recipe=smoke_recipe,
        )
    return registry


_RUNNERS = create_runner_registry()
_METHOD_RUNNERS = frozenset(
    {
        "cnlcu",
        "coteaching",
        "dual_t",
        "importance_reweighting",
        "pcse",
        "mc_ldce",
        "cal",
        "ca2c",
        "l2rw",
        "dld",
        "t_revision",
        "volmin",
        "upm",
        "lend",
        "dividemix",
        "volminnet",
    }
)
_SUPPORTED_METHOD_ALIASES = frozenset(
    {
        "apl", "gce", "cdr", "loss_correction", "binary_risk", "natarajan",
        "jocor", "pdl", "cwd", "dss", "fine",
    }
)
_RUNNER_ALIASES = {
    "apl": "supervised",
    "gce": "supervised",
    "dss": "supervised",
    "cdr": "supervised",
    "loss_correction": "supervised",
    "binary_risk": "binary",
    "natarajan": "binary",
    "jocor": "multi_model",
    "pdl": "instance_transition",
    "cwd": "cwd",
    "dss": "supervised",
    "fine": "fine",
}
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
    if method == "dividemix":
        _set_nested_epoch(config, ("dividemix", "training", "epochs"), epochs)
        return
    if method == "upm":
        _set_nested_epoch(config, ("upm", "main", "epochs"), epochs)
        return
    if method == "dld":
        _set_nested_epoch(config, ("dld", "diffusion", "epochs"), epochs)
        return
    if method == "lend":
        _set_nested_epoch(config, ("lend", "training", "epochs"), epochs)
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
    if method and method not in _METHOD_RUNNERS and method not in _SUPPORTED_METHOD_ALIASES:
        suggestion = get_close_matches(method, sorted(_METHOD_RUNNERS), n=1)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(
            f"Unsupported training method: unknown method {method!r}{hint}; "
            "run 'lnl list experiments' to see runnable methods"
        )

    inferred = _RUNNER_ALIASES.get(method, method)
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
    selected = _RUNNER_ALIASES.get(selected, selected)
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
