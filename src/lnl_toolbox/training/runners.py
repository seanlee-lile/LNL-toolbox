from __future__ import annotations

"""Central lazy registry for user-facing experiment execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from lnl_toolbox.data.profile import Modality
from lnl_toolbox.training.compatibility import MethodRequirements
from lnl_toolbox.training.planning import (
    RunPlan,
    coteaching_plan,
    dld_plan,
    dividemix_plan,
    generic_plan,
    lend_plan,
    supervised_plan,
    upm_plan,
)


Runner = Callable[[dict[str, Any], str | Path | None, str | Path | None], Path]
Planner = Callable[[Mapping[str, Any], str, tuple[str, ...] | None], RunPlan]
RequirementsProvider = Callable[[Mapping[str, Any]], MethodRequirements]


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    name: str
    module: str
    function: str
    supports_resume: bool = True
    budget_path: tuple[str, ...] | None = ("trainer", "epochs")
    planner: Planner = generic_plan
    requirements_provider: RequirementsProvider | None = None

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

    def describe(self, config: Mapping[str, Any]) -> RunPlan:
        return self.planner(config, self.name, self.budget_path)

    def requirements(self, config: Mapping[str, Any] | None = None) -> MethodRequirements | None:
        if self.requirements_provider is None:
            return None
        return self.requirements_provider(config or {})

    def apply_training_budget(self, config: dict[str, Any], epochs: int) -> None:
        if epochs <= 0:
            raise ValueError("--epochs must be positive")
        if self.budget_path is None:
            raise ValueError(
                f"--epochs is ambiguous or unsupported for runner {self.name!r}; "
                "use --set with the explicit stage path"
            )
        current = config
        for key in self.budget_path[:-1]:
            value = current.get(key)
            if not isinstance(value, dict):
                raise ValueError(
                    "--epochs requires an existing "
                    + ".".join(self.budget_path[:-1])
                    + " mapping"
                )
            current = value
        leaf = self.budget_path[-1]
        if leaf not in current:
            raise ValueError(
                f"--epochs requires existing config path {'.'.join(self.budget_path)}"
            )
        current[leaf] = epochs


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
        budget_path: tuple[str, ...] | None = ("trainer", "epochs"),
        planner: Planner = generic_plan,
        requirements_provider: RequirementsProvider | None = None,
    ) -> None:
        key = _normalize(name)
        if key in self._specs:
            raise KeyError(f"runner {key!r} is already registered")
        self._specs[key] = RunnerSpec(
            key, module, function, supports_resume, budget_path, planner,
            requirements_provider,
        )

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


def _image_requirements(
    method: str,
    *,
    min_classes: int = 2,
    exact_classes: frozenset[int] = frozenset(),
    method_noise_prior: bool = False,
    method_noise_prior_paths: tuple[tuple[str, ...], ...] = (),
    clean_validation: bool = False,
) -> RequirementsProvider:
    def provide(_config: Mapping[str, Any]) -> MethodRequirements:
        return MethodRequirements(
            method=method,
            supported_modalities=frozenset({Modality.IMAGE}),
            min_classes=min_classes,
            exact_classes=exact_classes,
            requires_method_noise_prior=method_noise_prior,
            method_noise_prior_paths=method_noise_prior_paths,
            requires_clean_validation=clean_validation,
            validation_target="clean" if clean_validation else "noisy",
        )
    return provide


def _importance_reweighting_requirements(_config: Mapping[str, Any]) -> MethodRequirements:
    return MethodRequirements(
        method="importance_reweighting",
        supported_modalities=frozenset({Modality.TABULAR}),
        min_classes=2,
        max_classes=2,
        exact_classes=frozenset({2}),
        validation_target="noisy",
    )


def _pcse_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    pretraining = config.get("pretraining_stage", {}) or {}
    external = isinstance(pretraining, Mapping) and str(pretraining.get("mode", "train")) == "external_checkpoint"
    return MethodRequirements(
        method="pcse",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        min_classes=3,
        validation_target="noisy",
        required_pretrained_roles=("upm_main_best",) if external else (),
        pretrained_role_paths=(
            (("upm_main_best", ("pretraining_stage", "source", "adapter")),)
            if external else ()
        ),
    )


def _dld_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    dld = config.get("dld", {}) or {}
    feature = dld.get("feature_extractor", {}) if isinstance(dld, Mapping) else {}
    source = str(feature.get("source", "repository_frozen")) if isinstance(feature, Mapping) else "repository_frozen"
    return MethodRequirements(
        method="dld",
        supported_modalities=frozenset({Modality.IMAGE}),
        min_classes=2,
        validation_target="noisy",
        required_pretrained_roles=("upm_main_best",) if source == "external_checkpoint" else (),
        pretrained_role_paths=(
            (("upm_main_best", ("dld", "feature_extractor", "external", "adapter")),)
            if source == "external_checkpoint" else ()
        ),
    )


def create_runner_registry() -> RunnerRegistry:
    registry = RunnerRegistry()
    registry.add(
        "supervised",
        "lnl_toolbox.training.experiment",
        "run_supervised_experiment",
        planner=supervised_plan,
    )
    registry.add(
        "clean",
        "lnl_toolbox.training.clean_baseline",
        "run_clean_experiment",
        planner=supervised_plan,
    )
    registry.add("multi_model", "lnl_toolbox.training.multi_model_experiment", "run_multi_model_experiment")
    registry.add("cwd", "lnl_toolbox.training.cwd_experiment", "run_cwd_experiment")
    registry.add("fine", "lnl_toolbox.training.fine_experiment", "run_fine_experiment")
    registry.add(
        "binary",
        "lnl_toolbox.training.binary_experiment",
        "run_binary_experiment",
        supports_resume=False,
        budget_path=None,
    )
    registry.add(
        "instance_transition",
        "lnl_toolbox.training.instance_transition_experiment",
        "run_instance_transition_experiment",
        requirements_provider=_image_requirements("pdl"),
    )
    registry.add(
        "coteaching",
        "lnl_toolbox.training.coteaching_experiment",
        "run_coteaching_experiment",
        planner=coteaching_plan,
        requirements_provider=_image_requirements(
            "coteaching",
            method_noise_prior=True,
            method_noise_prior_paths=(("coteaching", "noise_rate"),),
        ),
    )
    registry.add(
        "dual_t",
        "lnl_toolbox.training.dual_t_experiment",
        "run_dual_t_experiment",
        budget_path=None,
    )
    registry.add(
        "importance_reweighting",
        "lnl_toolbox.training.importance_reweighting_experiment",
        "run_importance_reweighting_experiment",
        requirements_provider=_importance_reweighting_requirements,
    )
    registry.add(
        "pcse",
        "lnl_toolbox.training.pcse_experiment",
        "run_pcse_experiment",
        budget_path=None,
        requirements_provider=_pcse_requirements,
    )
    registry.add("mc_ldce", "lnl_toolbox.training.mc_ldce_experiment", "run_mc_ldce_experiment")
    registry.add("cal", "lnl_toolbox.training.cal_experiment", "run_cal_experiment")
    registry.add(
        "ca2c", "lnl_toolbox.training.ca2c_experiment", "run_ca2c_experiment",
        requirements_provider=_image_requirements("ca2c", clean_validation=True),
    )
    registry.add("l2rw", "lnl_toolbox.training.l2rw_experiment", "run_l2rw_experiment")
    registry.add(
        "volminnet",
        "lnl_toolbox.training.volminnet_experiment",
        "run_volminnet_experiment",
        requirements_provider=_image_requirements(
            "volminnet", min_classes=3, exact_classes=frozenset({10, 100})
        ),
    )
    registry.add(
        "upm",
        "lnl_toolbox.training.upm_experiment",
        "run_upm_experiment",
        budget_path=("upm", "main", "epochs"),
        planner=upm_plan,
        requirements_provider=_image_requirements("upm"),
    )
    registry.add(
        "dld",
        "lnl_toolbox.training.dld_experiment",
        "run_dld_experiment",
        budget_path=("dld", "diffusion", "epochs"),
        planner=dld_plan,
        requirements_provider=_dld_requirements,
    )
    registry.add(
        "dividemix",
        "lnl_toolbox.training.dividemix_experiment",
        "run_dividemix_experiment",
        budget_path=("dividemix", "training", "epochs"),
        planner=dividemix_plan,
        requirements_provider=_image_requirements(
            "dividemix",
            method_noise_prior=True,
            method_noise_prior_paths=(("noise", "rate"),),
        ),
    )
    registry.add(
        "lend",
        "lnl_toolbox.training.lend_experiment",
        "run_lend_experiment",
        budget_path=("lend", "training", "epochs"),
        planner=lend_plan,
        requirements_provider=_image_requirements("lend"),
    )
    registry.add(
        "cnlcu", "lnl_toolbox.training.cnlcu_experiment", "run_cnlcu_experiment",
        requirements_provider=_image_requirements(
            "cnlcu",
            method_noise_prior=True,
            method_noise_prior_paths=(("cnlcu", "noise_rate"),),
        ),
    )
    registry.add(
        "t_revision",
        "lnl_toolbox.training.t_revision_experiment",
        "run_t_revision_experiment",
        budget_path=("t_revision", "revision", "epochs"),
        requirements_provider=_image_requirements("t_revision"),
    )
    registry.add("volmin", "lnl_toolbox.training.volmin_experiment", "run_volmin_experiment")
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
        "lend",
        "pcse",
        "mc_ldce",
        "cal",
        "ca2c",
        "l2rw",
        "t_revision",
        "volmin",
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


def runner_specs() -> tuple[RunnerSpec, ...]:
    """Return the central registry's specs for discovery surfaces."""

    return tuple(_RUNNERS.get(name) for name in _RUNNERS.names())


def method_names() -> tuple[str, ...]:
    """Return public method names owned by the central runner registry."""

    return tuple(sorted(_METHOD_RUNNERS))


def apply_epoch_override(config: dict[str, Any], epochs: int) -> None:
    """Delegate the public budget override to the selected runner contract."""

    resolve_runner(config).apply_training_budget(config, epochs)


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
    "runner_specs",
]
