from __future__ import annotations

"""Central lazy registry for user-facing experiment execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.data.profile import Modality
from lnl_toolbox.training.compatibility import ConfigInputRequirement, MethodRequirements
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
RequirementsProvider = Callable[[Mapping[str, Any]], MethodRequirements | None]


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
        method_requirements = self.requirements(config)
        requirement_options = (
            {"requirements": method_requirements.data_requirements}
            if method_requirements is not None
            else {}
        )
        if self.supports_resume:
            return runner(config, output_dir, resume, **requirement_options)
        return runner(config, output_dir, **requirement_options)

    def describe(self, config: Mapping[str, Any]) -> RunPlan:
        return self.planner(config, self.name, self.budget_path)

    def requirements(self, config: Mapping[str, Any] | None = None) -> MethodRequirements | None:
        if self.requirements_provider is None:
            return None
        return self.requirements_provider(config or {})

    def data_requirements(self, config: Mapping[str, Any] | None = None) -> DataRequirements:
        requirements = self.requirements(config)
        if requirements is None:
            raise ValueError(f"runner {self.name!r} does not publish data requirements")
        return requirements.data_requirements

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
    extra_roles: tuple[DataRole, ...] = (),
    views: tuple[str, ...] = ("weak",),
) -> RequirementsProvider:
    def provide(_config: Mapping[str, Any]) -> MethodRequirements:
        return MethodRequirements(
            method=method,
            supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
            data_requirements=DataRequirements(
                roles=frozenset({
                    DataRole.TRAIN,
                    DataRole.CLEAN_VALIDATION if clean_validation else DataRole.NOISY_VALIDATION,
                    DataRole.TEST,
                    *extra_roles,
                }),
                views=views,
                validation_targets="clean" if clean_validation else "noisy",
                needs_noise_manifest=True,
            ),
            implemented_variant="current",
            min_classes=min_classes,
            exact_classes=exact_classes,
            requires_method_noise_prior=method_noise_prior,
            method_noise_prior_paths=method_noise_prior_paths,
        )
    return provide


def _config_input(
    code: str,
    *paths: tuple[str, ...],
    mode: str = "all",
    description: str,
    implementation_limit: str | None = None,
) -> ConfigInputRequirement:
    return ConfigInputRequirement(
        code=code,
        paths=tuple(paths),
        mode=mode,
        description=description,
        implementation_limit=implementation_limit,
    )


def _validation_target(config: Mapping[str, Any]) -> str:
    data = config.get("data", {}) or {}
    size = data.get("validation_size", data.get("num_val", 0))
    if int(size or 0) <= 0:
        return "none"
    return str((config.get("noise", {}) or {}).get("validation_targets", "clean")).strip().lower()


def _component_name(config: Mapping[str, Any], *path: str) -> str:
    value: Any = config
    for part in path:
        if not isinstance(value, Mapping):
            return ""
        value = value.get(part, {})
    if isinstance(value, Mapping):
        value = value.get("name", "")
    return str(value).strip().lower().replace("-", "_")


def _classification_data(
    *,
    validation: str = "noisy",
    extra_roles: tuple[DataRole, ...] = (),
    views: tuple[str, ...] = ("weak",),
    needs_manifest: bool = True,
    manifest_scope: str = "train_split",
    train_drop_last: bool | None = None,
    validation_size: int | None = None,
    split_strategy: str | None = None,
    subset_before_split: bool = False,
) -> DataRequirements:
    roles = {DataRole.TRAIN, DataRole.TEST, *extra_roles}
    if validation == "clean":
        roles.add(DataRole.CLEAN_VALIDATION)
    elif validation == "noisy":
        roles.add(DataRole.NOISY_VALIDATION)
    elif validation != "none":
        raise ValueError("validation must be clean, noisy, or none")
    return DataRequirements(
        roles=frozenset(roles),
        views=views,
        validation_targets="clean" if validation == "clean" else "noisy",
        needs_noise_manifest=needs_manifest,
        manifest_scope=manifest_scope,
        train_drop_last=train_drop_last,
        validation_size=validation_size,
        split_strategy=split_strategy,
        subset_before_split=subset_before_split,
    )


def _supervised_requirements(config: Mapping[str, Any]) -> MethodRequirements | None:
    risk = _component_name(config, "pipeline", "risk_corrector")
    weight = _component_name(config, "pipeline", "weight_provider")
    objective = _component_name(config, "pipeline", "objective_consumer")
    update = _component_name(config, "parameter_update")
    loss = _component_name(config, "loss")

    if risk in {"forward", "backward"}:
        estimator = _component_name(config, "pipeline", "transition_estimator")
        inputs = ()
        if estimator == "known":
            inputs = (_config_input(
                "requires_transition_matrix",
                ("pipeline", "transition_estimator", "matrix"),
                description="known-T loss correction requires a transition matrix",
            ),)
        return MethodRequirements(
            method="loss_correction",
            supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
            data_requirements=_classification_data(validation="noisy"),
            implemented_variant="forward_backward",
            required_config_inputs=inputs,
        )
    if weight == "mentornet":
        return MethodRequirements(
            method="mentornet",
            supported_modalities=frozenset({Modality.IMAGE}),
            data_requirements=_classification_data(validation="clean"),
            implemented_variant="artifact_weight_provider",
            implementation_limits=frozenset({"modality", "mentor_artifact"}),
            required_config_inputs=(_config_input(
                "requires_mentor_artifact",
                ("pipeline", "weight_provider", "artifact_path"),
                description="MentorNet requires a configured MentorArtifact",
                implementation_limit="mentor_artifact",
            ),),
        )
    if objective == "dss":
        return MethodRequirements(
            method="dss",
            supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
            data_requirements=_classification_data(validation="noisy"),
            implemented_variant="objective_consumer",
        )
    if update == "cdr":
        return MethodRequirements(
            method="cdr",
            supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
            data_requirements=_classification_data(validation="noisy"),
            implemented_variant="parameter_update_policy",
            requires_method_noise_prior=True,
            method_noise_prior_paths=(("parameter_update", "noise_rate"),),
        )
    if loss in {"apl", "gce"}:
        noise_config = config.get("noise", {}) or {}
        validation_target = str(
            noise_config.get("validation_targets", "clean")
        ).strip().lower()
        noise_name = str(noise_config.get("name", "clean")).strip().lower()
        requires_manifest = bool(noise_config.get("manifest")) or noise_name not in {
            "",
            "clean",
            "external",
        }
        return MethodRequirements(
            method=loss,
            supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
            data_requirements=_classification_data(
                validation=validation_target,
                needs_manifest=requires_manifest,
            ),
            implemented_variant="per_sample_loss",
            supports_native_noisy_labels=True,
        )
    validation_target = str(
        (config.get("noise", {}) or {}).get("validation_targets", "clean")
    ).strip().lower()
    return MethodRequirements(
        method="ce",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(
            validation=validation_target,
            needs_manifest=False,
        ),
        implemented_variant="per_sample_loss",
        supports_native_noisy_labels=True,
    )


def _binary_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    risk = _component_name(config, "risk")
    inputs = ()
    if risk in {"natarajan", "natarajan_unbiased"}:
        inputs = (_config_input(
            "requires_binary_noise_prior",
            ("risk", "rho_positive"),
            ("risk", "rho_negative"),
            description="Natarajan risk requires both class-conditional noise rates",
        ),)
    return MethodRequirements(
        method="binary",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(validation="none", needs_manifest=True),
        implemented_variant="binary_risk",
        min_classes=2,
        max_classes=2,
        exact_classes=frozenset({2}),
        required_config_inputs=inputs,
    )


def _l2rw_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    trusted = config.get("trusted_validation", {}) or {}
    source = (
        str(trusted.get("source", "")).strip().lower()
        if isinstance(trusted, Mapping) else ""
    )
    inputs = [_config_input(
        "requires_trusted_validation",
        ("trusted_validation", "source"),
        description="L2RW requires an explicit trusted-validation source",
    )]
    if source == "audited_manifest":
        inputs.append(_config_input(
            "requires_trusted_manifest",
            ("trusted_validation", "manifest"),
            description="audited L2RW supervision requires a trusted manifest",
        ))
    return MethodRequirements(
        method="l2rw",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(
            validation="clean",
            extra_roles=(DataRole.TRUSTED_VALIDATION,),
            train_drop_last=True if source == "official_generated" else None,
        ),
        implemented_variant="trusted_meta_reweighting",
        requires_clean_train_labels=source == "official_generated",
        required_config_inputs=tuple(inputs),
    )


def _cal_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    return MethodRequirements(
        method="cal",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(
            validation=_validation_target(config),
            needs_manifest=False,
        ),
        implemented_variant="proxy_label_sieve",
        supports_native_noisy_labels=True,
    )


def _cwd_requirements(_config: Mapping[str, Any]) -> MethodRequirements:
    return MethodRequirements(
        method="cwd",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(
            validation="none",
            extra_roles=(DataRole.TRAIN_EVAL,),
            manifest_scope="effective_train",
        ),
        implemented_variant="binary_protocol",
        implementation_limits=frozenset({"class_count"}),
        min_classes=2,
        max_classes=2,
        exact_classes=frozenset({2}),
        required_config_inputs=(_config_input(
            "requires_binary_noise_prior",
            ("noise", "rho_positive"), ("noise", "rho_negative"),
            description="CWD requires both class-conditional noise rates",
        ),),
    )


def _mc_ldce_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    transition = config.get("transition", {}) or {}
    estimator = str(transition.get("estimator", "")).strip().lower()
    inputs = ()
    if estimator and estimator != "paper_volmin":
        inputs = (_config_input(
            "requires_transition_source",
            ("transition", "artifact"), ("transition", "matrix"),
            mode="any",
            description="MC-LDCE requires a transition artifact or matrix",
        ),)
    return MethodRequirements(
        method="mc_ldce",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(
            validation=_validation_target(config),
            extra_roles=(DataRole.TRAIN_EVAL,),
        ),
        implemented_variant="centroid_decomposition",
        implementation_limits=frozenset({"class_count_evidence_unresolved"}),
        min_classes=3,
        required_config_inputs=inputs,
    )


def _ca2c_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    return MethodRequirements(
        method="ca2c",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(
            validation=_validation_target(config),
            views=("weak", "strong")
            if bool((config.get("data", {}) or {}).get("strong_augment", False))
            else ("weak",),
        ),
        implemented_variant="asymmetric_colearning",
    )


def _fine_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    official_strong = bool((config.get("data", {}) or {}).get("strong_policy"))
    return MethodRequirements(
        method="fine",
        supported_modalities=frozenset({Modality.IMAGE}),
        data_requirements=_classification_data(
            validation=_validation_target(config),
            views=("weak", "strong"),
            manifest_scope="effective_train",
        ),
        implemented_variant="sed_fine",
        implementation_limits=(
            frozenset({"modality"})
            if official_strong
            else frozenset({"modality", "strong_view"})
        ),
    )


def _jocor_requirements(config: Mapping[str, Any]) -> MethodRequirements | None:
    if _component_name(config, "algorithm") != "jocor":
        return None
    return MethodRequirements(
        method="jocor",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(
            validation=str((config.get("noise", {}) or {}).get("validation_targets", "clean")),
            manifest_scope="effective_train",
        ),
        implemented_variant="joint_coregularization",
        requires_method_noise_prior=True,
        method_noise_prior_paths=(("noise", "rate"),),
    )


def _importance_reweighting_requirements(_config: Mapping[str, Any]) -> MethodRequirements:
    return MethodRequirements(
        method="importance_reweighting",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(validation="noisy"),
        implemented_variant="binary_asymmetric_rcn",
        implementation_limits=frozenset({"class_count"}),
        min_classes=2,
        max_classes=2,
        exact_classes=frozenset({2}),
    )


def _pcse_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    pretraining = config.get("pretraining_stage", {}) or {}
    external = isinstance(pretraining, Mapping) and str(pretraining.get("mode", "train")) == "external_checkpoint"
    return MethodRequirements(
        method="pcse",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(
            validation="noisy",
            extra_roles=(DataRole.TRAIN_EVAL,),
        ),
        implemented_variant="per_class_statistics",
        implementation_limits=frozenset({"class_count"}),
        min_classes=3,
        required_pretrained_roles=("upm_main_best",) if external else (),
        pretrained_role_paths=(
            (("upm_main_best", ("pretraining_stage", "source", "adapter")),)
            if external else ()
        ),
    )


def _volminnet_requirements(_config: Mapping[str, Any]) -> MethodRequirements:
    return MethodRequirements(
        method="volminnet",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(validation="noisy"),
        implemented_variant="fixed_diagonal_sigmoid_offdiag",
        implementation_limits=frozenset({"class_count"}),
        min_classes=3,
    )


def _dld_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    dld = config.get("dld", {}) or {}
    feature = dld.get("feature_extractor", {}) if isinstance(dld, Mapping) else {}
    source = str(feature.get("source", "repository_frozen")) if isinstance(feature, Mapping) else "repository_frozen"
    return MethodRequirements(
        method="dld",
        supported_modalities=frozenset({Modality.IMAGE}),
        data_requirements=_classification_data(
            validation="noisy",
            views=("weak", "strong"),
        ),
        implemented_variant="two_view_diffusion",
        implementation_limits=frozenset({"modality"}),
        min_classes=2,
        required_pretrained_roles=("upm_main_best",) if source == "external_checkpoint" else (),
        pretrained_role_paths=(
            (("upm_main_best", ("dld", "feature_extractor", "external", "adapter")),)
            if source == "external_checkpoint" else ()
        ),
    )


def _pdl_requirements(config: Mapping[str, Any]) -> MethodRequirements:
    official = is_official_pdl_config(config)
    validation_size = int(
        (config.get("warmup", {}) or {}).get("noisy_validation_size", 0)
    ) or None
    return MethodRequirements(
        method="pdl",
        supported_modalities=frozenset({Modality.IMAGE, Modality.TABULAR}),
        data_requirements=_classification_data(
            validation="noisy",
            validation_size=validation_size,
            split_strategy="numpy_choice_complement" if official else None,
            subset_before_split=True,
        ),
        implemented_variant="part_dependent_transition",
    )


def is_official_pdl_config(config: Mapping[str, Any]) -> bool:
    """Return the canonical predicate for the official PDL phase variant."""
    return (
        _component_name(config, "algorithm", "correction") in {"pdl", "pdl_revision"}
        or "phases" in config
    )


def create_runner_registry() -> RunnerRegistry:
    registry = RunnerRegistry()
    registry.add(
        "supervised",
        "lnl_toolbox.training.experiment",
        "run_supervised_experiment",
        planner=supervised_plan,
        requirements_provider=_supervised_requirements,
    )
    registry.add(
        "clean",
        "lnl_toolbox.training.clean_baseline",
        "run_clean_experiment",
        planner=supervised_plan,
    )
    registry.add(
        "multi_model", "lnl_toolbox.training.multi_model_experiment",
        "run_multi_model_experiment", requirements_provider=_jocor_requirements,
    )
    registry.add(
        "cwd", "lnl_toolbox.training.cwd_experiment", "run_cwd_experiment",
        requirements_provider=_cwd_requirements,
    )
    registry.add(
        "fine", "lnl_toolbox.training.fine_experiment", "run_fine_experiment",
        requirements_provider=_fine_requirements,
    )
    registry.add(
        "binary",
        "lnl_toolbox.training.binary_experiment",
        "run_binary_experiment",
        supports_resume=False,
        budget_path=None,
        requirements_provider=_binary_requirements,
    )
    registry.add(
        "instance_transition",
        "lnl_toolbox.training.instance_transition_experiment",
        "run_instance_transition_experiment",
        requirements_provider=_pdl_requirements,
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
        requirements_provider=_image_requirements("dual_t"),
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
    registry.add(
        "mc_ldce", "lnl_toolbox.training.mc_ldce_experiment", "run_mc_ldce_experiment",
        requirements_provider=_mc_ldce_requirements,
    )
    registry.add(
        "cal", "lnl_toolbox.training.cal_experiment", "run_cal_experiment",
        requirements_provider=_cal_requirements,
    )
    registry.add(
        "ca2c", "lnl_toolbox.training.ca2c_experiment", "run_ca2c_experiment",
        requirements_provider=_ca2c_requirements,
    )
    registry.add(
        "l2rw", "lnl_toolbox.training.l2rw_experiment", "run_l2rw_experiment",
        requirements_provider=_l2rw_requirements,
    )
    registry.add(
        "volminnet",
        "lnl_toolbox.training.volminnet_experiment",
        "run_volminnet_experiment",
        requirements_provider=_volminnet_requirements,
    )
    registry.add(
        "upm",
        "lnl_toolbox.training.upm_experiment",
        "run_upm_experiment",
        budget_path=("upm", "main", "epochs"),
        planner=upm_plan,
        requirements_provider=_image_requirements("upm", extra_roles=(DataRole.TRAIN_EVAL,)),
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
            extra_roles=(DataRole.TRAIN_EVAL,),
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
        requirements_provider=_image_requirements(
            "t_revision", extra_roles=(DataRole.TRAIN_EVAL,)
        ),
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


def resolve_data_requirements(
    config: Mapping[str, Any], *, expected_runner: str | None = None
) -> DataRequirements:
    """Resolve runtime data requirements in the runner layer, never DataService."""

    spec = resolve_runner(config)
    expected = None if expected_runner is None else _normalize(expected_runner)
    if spec.name == "clean" and expected == "supervised":
        # ``clean`` is a compatibility wrapper around the supervised engine.
        # It deliberately owns no second requirement provider.
        return _RUNNERS.get("supervised").data_requirements(config)
    if expected is not None and spec.name != expected:
        raise ValueError(
            f"runner {spec.name!r} does not match expected runner {expected_runner!r}"
        )
    return spec.data_requirements(config)


__all__ = [
    "RunnerRegistry",
    "RunnerSpec",
    "apply_epoch_override",
    "create_runner_registry",
    "method_names",
    "resolve_data_requirements",
    "resolve_runner",
    "runner_names",
    "runner_specs",
]
