from __future__ import annotations

"""Discoverable experiment recipes and paper-to-implementation metadata."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata, resources
import json
import math
from pathlib import Path
import pickle
from typing import Any

from lnl_toolbox.cli import repository_root
from lnl_toolbox.core.config_schema import normalize_experiment_config
from lnl_toolbox.training.runners import RunnerSpec, resolve_runner


_MENTORNET_TEACHER_CONFIG = Path(
    "configs/experiment/mentornet_dd_teacher_cifar10_symmetric04.yaml"
)
_MENTORNET_SMOKE_RECIPE = "mentornet-dd-cifar100-symmetric04-smoke"


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    id: str
    config_path: Path
    profile: str
    runner: str
    dataset: str
    noise: str
    method: str
    epochs: int | None
    implementation_status: str
    configuration_fidelity: str
    reproduction_status: str
    availability: str
    visibility: str
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class PaperConfig:
    recipe_id: str
    profile: str
    variant: str
    configuration_fidelity: str
    implementation_status: str
    reproduction_status: str
    availability: str


@dataclass(frozen=True, slots=True)
class PaperSpec:
    id: str
    acronym: str
    title: str
    venue: str
    year: int
    source_url: str
    summary: str
    mechanism: str
    lifecycle: tuple[str, ...]
    limitations: tuple[str, ...]
    configs: tuple[PaperConfig, ...]
    concept_to_config: tuple[Mapping[str, str], ...]
    implementation_paths: tuple[str, ...]
    implementation_status: str
    reproduction_status: str
    availability: str


def load_yaml(path: Path, *, check_parameters: bool = False) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required; install training dependencies with "
            "python -m pip install -e \".[train]\""
        ) from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration must contain a YAML mapping: {path}")
    if check_parameters:
        errors = _config_parameter_errors(value)
        if errors:
            raise ValueError("\n".join(errors))
    return normalize_experiment_config(dict(value))


def _profile(path: Path) -> str:
    name = path.stem.lower()
    if "reproduction" in name or path.parent.name == "reproduction":
        return "reproduction"
    if "smoke" in name:
        return "smoke"
    return "experiment"


def _recipe_id(path: Path) -> str:
    return path.stem.lower().replace("_", "-")


def _display_method(config: Mapping[str, Any], runner: str) -> str:
    method = config.get("method", "")
    if isinstance(method, Mapping):
        method = method.get("name", "")
    if str(method).strip():
        return str(method)
    if runner == "instance_transition":
        transition = config.get("instance_transition", {}) or {}
        if isinstance(transition, Mapping) and str(transition.get("name", "")).strip():
            return str(transition["name"])
    algorithm = config.get("algorithm", {}) or {}
    if isinstance(algorithm, Mapping) and str(algorithm.get("name", "")).strip():
        return str(algorithm["name"])
    pipeline = config.get("pipeline", {}) or {}
    if isinstance(pipeline, Mapping):
        for key in ("weight_provider", "objective_consumer", "risk_corrector"):
            component = pipeline.get(key, {}) or {}
            if isinstance(component, Mapping) and str(component.get("name", "")).strip():
                return str(component["name"])
    for key, defaults in (("loss", {"ce"}), ("parameter_update", {"standard"}), ("selector", {"all"})):
        component = config.get(key, {}) or {}
        if isinstance(component, Mapping):
            name = str(component.get("name", "")).strip()
            if name and name not in defaults:
                return name
    return runner


def _recipe_manifest() -> tuple[
    tuple[str, ...], frozenset[str], Mapping[str, Mapping[str, str]]
]:
    text = resources.files("lnl_toolbox.cli").joinpath(
        "data/recipe_catalog.json"
    ).read_text(encoding="utf-8")
    raw = json.loads(text)
    paths = tuple(str(value) for value in raw["recipes"])
    if len(paths) != len(set(paths)):
        raise ValueError("built-in recipe manifest contains duplicate paths")
    public_items = raw.get("public", ())
    public = {
        str(item["path"]): {
            "label": str(item["label"]),
            "description": str(item["description"]),
        }
        for item in public_items
    }
    unknown = set(public).difference(paths)
    if unknown:
        raise ValueError(
            f"public recipe manifest references unknown paths: {sorted(unknown)}"
        )
    return (
        paths,
        frozenset(str(value) for value in raw.get("conditional", ())),
        public,
    )


def _installed_recipe_path(relative: str) -> Path:
    try:
        distribution = metadata.distribution("lnl-toolbox")
    except metadata.PackageNotFoundError as exc:
        raise FileNotFoundError(
            f"built-in recipe is unavailable outside a source checkout: {relative}"
        ) from exc
    installed_suffix = (Path("share/lnl-toolbox") / relative).as_posix()
    for entry in distribution.files or ():
        if entry.as_posix().endswith(installed_suffix):
            return Path(distribution.locate_file(entry)).resolve()
    raise FileNotFoundError(f"packaged built-in recipe is missing: {relative}")


def _recipe_path(relative: str, project: Path) -> Path:
    source = (project / relative).resolve()
    if source.is_file():
        return source
    installed = _installed_recipe_path(relative)
    if not installed.is_file():
        raise FileNotFoundError(f"packaged built-in recipe is missing: {relative}")
    return installed


def _implementation_status(method: str, runner: str) -> str:
    if method in {
        "cnlcu", "coteaching", "dividemix", "dss", "dual_t", "fine",
        "importance_reweighting", "jocor", "pcse", "pdl", "t_revision",
        "upm", "volminnet", "lend",
        "cwd",
    }:
        return "user_ready"
    if method == "mentornet":
        return "workflow"
    if runner in {"clean", "supervised"}:
        return "component"
    return "workflow"


def _configuration_fidelity(path: Path, profile: str) -> str:
    name = path.stem.lower()
    if profile == "smoke":
        return "smoke"
    if "reproduction" in name or path.parent.name == "reproduction":
        return "paper_oriented"
    return "engineering"


def discover_recipes(
    root: Path | None = None,
    *,
    include_conditional: bool = False,
    public_only: bool = False,
) -> tuple[RecipeSpec, ...]:
    project = (root or repository_root()).resolve()
    manifest_paths, conditional_paths, public_recipes = _recipe_manifest()
    recipes: list[RecipeSpec] = []
    seen: set[str] = set()
    for relative in manifest_paths:
        if public_only and relative not in public_recipes:
            continue
        conditional = relative in conditional_paths
        if conditional and not include_conditional:
            continue
        path = _recipe_path(relative, project)
        recipe_id = _recipe_id(path)
        if recipe_id in seen:
            recipe_id = f"{path.parent.name}-{recipe_id}"
        seen.add(recipe_id)
        config = load_yaml(path)
        data = config.get("data")
        # Auxiliary stage configs (for example Mentor training) are not
        # standalone experiment recipes and remain available to their CLI.
        if not isinstance(data, Mapping):
            continue
        runner = resolve_runner(config).name
        noise = config.get("noise", {}) or {}
        trainer = config.get("trainer", {}) or {}
        recipes.append(
            RecipeSpec(
                id=recipe_id,
                config_path=path.resolve(),
                profile=_profile(path),
                runner=runner,
                dataset=str(data.get("name", "unknown")),
                noise=str(noise.get("name", "clean")) if noise else "clean",
                method=_display_method(config, runner),
                epochs=int(trainer["epochs"]) if "epochs" in trainer else None,
                implementation_status=_implementation_status(
                    _display_method(config, runner), runner
                ),
                configuration_fidelity=str(
                    config.get(
                        "configuration_fidelity",
                        _configuration_fidelity(path, _profile(path)),
                    )
                ),
                reproduction_status="not_run",
                availability="conditional" if conditional else "runnable",
                visibility="public" if relative in public_recipes else "internal",
                label=public_recipes.get(relative, {}).get("label", recipe_id),
                description=public_recipes.get(relative, {}).get("description", ""),
            )
        )
    return tuple(recipes)


def recipe_by_id(recipe_id: str, root: Path | None = None) -> RecipeSpec:
    key = recipe_id.strip().lower().replace("_", "-")
    recipes = {
        item.id: item
        for item in discover_recipes(root, include_conditional=True)
    }
    try:
        return recipes[key]
    except KeyError as exc:
        from difflib import get_close_matches

        suggestion = get_close_matches(key, sorted(recipes), n=1)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(f"unknown recipe {recipe_id!r}{hint}") from exc


def load_papers(root: Path | None = None) -> tuple[PaperSpec, ...]:
    project = (root or repository_root()).resolve()
    raw = json.loads(
        resources.files("lnl_toolbox").joinpath("paper_catalog.json").read_text(
            encoding="utf-8"
        )
    )
    recipes = {
        item.id: item
        for item in discover_recipes(project, include_conditional=True)
    }
    papers: list[PaperSpec] = []
    for item in raw:
        parsed_configs: list[PaperConfig] = []
        for value in item["configs"]:
            recipe = recipes.get(value["recipe_id"])
            if recipe is None:
                raise ValueError(
                    f"paper {item['id']!r} references unknown built-in recipe "
                    f"{value['recipe_id']!r}"
                )
            parsed_configs.append(
                PaperConfig(
                    recipe_id=value["recipe_id"],
                    profile=value["profile"],
                    variant=value["variant"],
                    configuration_fidelity=str(
                        value.get(
                            "configuration_fidelity",
                            value.get("fidelity", recipe.configuration_fidelity),
                        )
                    ),
                    implementation_status=str(
                        value.get("implementation_status", recipe.implementation_status)
                    ),
                    reproduction_status=str(
                        value.get("reproduction_status", "not_run")
                    ),
                    availability=str(value.get("availability", recipe.availability)),
                )
            )
        configs = tuple(parsed_configs)
        papers.append(
            PaperSpec(
                id=item["id"],
                acronym=item["acronym"],
                title=item["title"],
                venue=item["venue"],
                year=int(item["year"]),
                source_url=item["source_url"],
                summary=item["summary"],
                mechanism=item["mechanism"],
                lifecycle=tuple(item["lifecycle"]),
                limitations=tuple(item["limitations"]),
                configs=configs,
                concept_to_config=tuple(item["concept_to_config"]),
                implementation_paths=tuple(item["implementation_paths"]),
                implementation_status=str(
                    item.get(
                        "implementation_status",
                        max(
                            (value.implementation_status for value in configs),
                            default="component",
                        ),
                    )
                ),
                reproduction_status=str(item.get("reproduction_status", "not_run")),
                availability=str(
                    item.get(
                        "availability",
                        "conditional"
                        if any(value.availability == "conditional" for value in configs)
                        else "runnable",
                    )
                ),
            )
        )
    return tuple(sorted(papers, key=lambda value: (value.year, value.id)))


def paper_by_id(paper_id: str, root: Path | None = None) -> PaperSpec:
    key = paper_id.strip().lower()
    papers = {item.id: item for item in load_papers(root)}
    aliases = {item.acronym.lower(): item for item in papers.values()}
    if key in papers:
        return papers[key]
    if key in aliases:
        return aliases[key]
    raise ValueError(
        f"unknown paper {paper_id!r}; valid papers: " + ", ".join(sorted(papers))
    )


def select_paper_config(
    paper: PaperSpec,
    *,
    profile: str | None = None,
    variant: str | None = None,
    root: Path | None = None,
) -> tuple[PaperConfig, RecipeSpec]:
    candidates = list(paper.configs)
    if not candidates:
        raise ValueError(
            f"paper {paper.id!r} has no built-in runnable recipe; "
            f"availability={paper.availability}"
        )
    if profile:
        candidates = [item for item in candidates if item.profile == profile]
    if variant:
        candidates = [item for item in candidates if item.variant == variant]
    if not candidates:
        choices = ", ".join(
            f"{item.profile}/{item.variant}" for item in paper.configs
        )
        raise ValueError(f"no matching config; valid profile/variant pairs: {choices}")
    variants = sorted({item.variant for item in candidates})
    if len(candidates) > 1 and not variant:
        raise ValueError("multiple configs match; choose --variant from: " + ", ".join(variants))
    selected = candidates[0]
    return selected, recipe_by_id(selected.recipe_id, root)


def default_paper_config(
    paper: PaperSpec,
    *,
    root: Path | None = None,
) -> tuple[PaperConfig, RecipeSpec]:
    """Return the single formal Web/CLI default for one paper.

    Catalog order is intentional when a paper publishes several reproduction
    variants. Smoke profiles are never selected as the formal default.
    """

    candidates = [item for item in paper.configs if item.profile == "reproduction"]
    if not candidates:
        raise ValueError(f"paper {paper.id!r} has no formal reproduction recipe")
    selected = candidates[0]
    return selected, recipe_by_id(selected.recipe_id, root)


def load_recipe_config(recipe: RecipeSpec, *, check_parameters: bool = False) -> dict[str, Any]:
    """Load one explicit built-in recipe without scanning user configuration."""

    return load_yaml(recipe.config_path, check_parameters=check_parameters)


def find_project_root(config_path: Path | None = None, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    starts = [config_path.resolve().parent] if config_path is not None else []
    starts.append(Path.cwd().resolve())
    for start in starts:
        for candidate in (start, *start.parents):
            pyproject = candidate / "pyproject.toml"
            if pyproject.is_file() and "name = \"lnl-toolbox\"" in pyproject.read_text(encoding="utf-8"):
                return candidate
    return config_path.resolve().parent if config_path is not None else Path.cwd().resolve()


def resolve_config_paths(config: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    for section, key in (("data", "root"), ("data", "path"), ("noise", "manifest")):
        value = resolved.get(section)
        if isinstance(value, Mapping) and value.get(key):
            updated = dict(value)
            path = Path(str(updated[key])).expanduser()
            updated[key] = str(path if path.is_absolute() else (project_root / path).resolve())
            resolved[section] = updated
    if resolved.get("output_root"):
        path = Path(str(resolved["output_root"])).expanduser()
        resolved["output_root"] = str(path if path.is_absolute() else (project_root / path).resolve())
    pipeline = resolved.get("pipeline")
    if isinstance(pipeline, Mapping):
        updated_pipeline = dict(pipeline)
        provider = updated_pipeline.get("weight_provider")
        if isinstance(provider, Mapping) and provider.get("artifact_path"):
            updated_provider = dict(provider)
            path = Path(str(updated_provider["artifact_path"])).expanduser()
            updated_provider["artifact_path"] = str(
                path if path.is_absolute() else (project_root / path).resolve()
            )
            updated_pipeline["weight_provider"] = updated_provider
            resolved["pipeline"] = updated_pipeline
    return resolved


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _quoted_command_path(path: Path, project_root: Path) -> str:
    value = _display_path(path, project_root)
    return f'"{value}"' if any(character.isspace() for character in value) else value


def mentornet_preparation_status(
    config: Mapping[str, Any],
    project_root: Path | None = None,
    *,
    student_recipe: str | None = None,
) -> dict[str, Any] | None:
    """Describe the explicit offline Mentor preparation required by a Student run."""

    pipeline = config.get("pipeline", {}) or {}
    if not isinstance(pipeline, Mapping):
        return None
    provider = pipeline.get("weight_provider", {}) or {}
    if (
        not isinstance(provider, Mapping)
        or str(provider.get("name", "")).strip().lower() != "mentornet"
    ):
        return None
    root = (project_root or find_project_root()).resolve()
    artifact = Path(str(provider.get("artifact_path", ""))).expanduser()
    artifact = artifact if artifact.is_absolute() else (root / artifact).resolve()
    artifact_ready = False
    artifact_error: str | None = None
    if artifact.is_file():
        try:
            from lnl_toolbox.training.mentor_artifacts import MentorArtifact

            MentorArtifact.load(artifact)
            artifact_ready = True
        except (
            EOFError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            artifact_error = str(exc)

    teacher_config = root / _MENTORNET_TEACHER_CONFIG
    if not teacher_config.is_file():
        try:
            teacher_config = _installed_recipe_path(
                _MENTORNET_TEACHER_CONFIG.as_posix()
            )
        except FileNotFoundError:
            pass
    feature_data = root / "data/mentornet/cifar10-symmetric04-seed20260729/mentor_features.npz"
    if teacher_config.is_file():
        teacher = load_yaml(teacher_config)
        configured_feature = Path(str(teacher.get("feature_data", feature_data)))
        feature_data = (
            configured_feature
            if configured_feature.is_absolute()
            else (root / configured_feature).resolve()
        )
    expected_artifact = feature_data.parent / "mentor_artifact.pt"
    preparation_available = (
        teacher_config.is_file()
        and artifact.resolve() == expected_artifact.resolve()
    )
    recipe_id = student_recipe or (
        _MENTORNET_SMOKE_RECIPE if preparation_available else None
    )
    commands: dict[str, str] = {}
    if preparation_available:
        teacher_arg = _quoted_command_path(teacher_config, root)
        output_dir_arg = _quoted_command_path(feature_data.parent, root)
        artifact_arg = _quoted_command_path(artifact, root)
        commands = {
            "prepare": (
                f"lnl mentor prepare --config {teacher_arg} "
                f"--output-dir {output_dir_arg}"
            ),
            "train": (
                f"lnl mentor train --config {teacher_arg} --output {artifact_arg}"
            ),
        }
        if recipe_id:
            commands["student"] = (
                f"lnl run --recipe {recipe_id} --check-data"
            )
    return {
        "status": "ready" if artifact_ready else "not_ready",
        "artifact_ready": artifact_ready,
        "artifact_path": str(artifact),
        "artifact_error": artifact_error,
        "feature_ready": feature_data.is_file(),
        "feature_path": str(feature_data),
        "teacher_config": str(teacher_config),
        "preparation_available": preparation_available,
        "student_recipe": recipe_id,
        "commands": commands,
    }


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} configuration must be a mapping")
    return value


def _validate_dedicated_runner(config: Mapping[str, Any], runner: str) -> None:
    if runner == "multi_model":
        algorithm = _require_mapping(config, "algorithm")
        if str(algorithm.get("name", "")).strip().lower() != "jocor":
            raise ValueError("multi_model built-in workflow requires algorithm.name: jocor")
        models = config.get("models")
        if not isinstance(models, list) or len(models) != 2:
            raise ValueError("JoCoR requires exactly two model configurations")
        selector = _require_mapping(config, "selector")
        if str(selector.get("name", "")).strip().lower() != "small_loss":
            raise ValueError("JoCoR requires selector.name: small_loss")
    elif runner == "cwd":
        cwd = _require_mapping(config, "cwd")
        pinv_rcond = cwd.get("pinv_rcond")
        if pinv_rcond is not None and (
            not math.isfinite(float(pinv_rcond)) or float(pinv_rcond) < 0.0
        ):
            raise ValueError("cwd.pinv_rcond must be finite and non-negative")
        # Keep accepting the old smoke-only key while new reproduction YAMLs
        # use the paper's explicit Moore-Penrose default above.
        if "ridge" in cwd and float(cwd.get("ridge", 0.0)) < 0.0:
            raise ValueError("cwd.ridge must be non-negative")
        data = _require_mapping(config, "data")
        folds = int(data.get("folds", 0))
        fold_index = int(data.get("fold_index", -1))
        if folds < 2 or not 0 <= fold_index < folds:
            raise ValueError("CWD requires 0 <= data.fold_index < data.folds")
    elif runner == "fine":
        fine = _require_mapping(config, "fine")
        if int(fine.get("warmup_epochs", 0)) <= 0:
            raise ValueError("fine.warmup_epochs must be positive")
    elif runner == "instance_transition":
        transition = _require_mapping(config, "instance_transition")
        if str(transition.get("name", "")).strip().lower() != "pdl":
            raise ValueError("instance_transition runner currently requires name: pdl")
        algorithm = _require_mapping(config, "algorithm")
        if str(algorithm.get("correction", "")).strip().lower() not in {
            "forward", "pdl", "pdl_revision"
        }:
            raise ValueError(
                "PDL workflow requires algorithm.correction: forward, pdl, or pdl_revision"
            )


def _config_parameter_errors(config: Mapping[str, Any]) -> list[str]:
    """Collect input errors before normalization can coerce or hide bad values.

    Rules describe supported input domains, not paper defaults/search grids.
    Common leaf rules also cover optimizers and schedulers in nested stages.
    This intentionally has no dependency on Web code or machine-local data.
    """
    def number(value: Any) -> bool:
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    rules = {
        "integer": ("an integer >= 0", lambda v: type(v) is int and v >= 0),
        "positive_integer": ("an integer > 0", lambda v: type(v) is int and v > 0),
        "number": ("a finite number", number),
        "nonnegative": ("a finite number >= 0", lambda v: number(v) and v >= 0),
        "positive": ("a finite number > 0", lambda v: number(v) and v > 0),
        "probability": ("a finite number in [0, 1]", lambda v: number(v) and 0 <= v <= 1),
        "open_probability": ("a finite number in (0, 1)", lambda v: number(v) and 0 < v < 1),
        "decay": ("a finite number in [0, 1)", lambda v: number(v) and 0 <= v < 1),
        "seed": ("an integer in [0, 4294967295]", lambda v: type(v) is int and 0 <= v < 2**32),
        "half_probability": ("a finite number in (0, 0.5)", lambda v: number(v) and 0 < v < 0.5),
        "boolean": ("a boolean", lambda v: type(v) is bool),
        "text": ("a non-empty string without control characters", lambda v: isinstance(v, str) and bool(v.strip()) and all(ord(c) >= 32 for c in v)),
        "milestones": ("a list of strictly increasing positive integers", lambda v: isinstance(v, list) and all(type(x) is int and x > 0 for x in v) and v == sorted(set(v))),
        "rates": ("null or a list of positive finite learning rates", lambda v: v is None or isinstance(v, list) and all(number(x) and x > 0 for x in v)),
        "numbers": ("a list of finite numbers", lambda v: isinstance(v, list) and all(number(x) for x in v)),
        "vector": ("a non-empty list of finite numbers", lambda v: isinstance(v, list) and bool(v) and all(number(x) for x in v)),
    }
    leaf_groups = {
        "integer": "seed num_workers max_steps num_clean num_val validation_size test_size train_size fold_index strong_magnitude patience warmup_epochs gradual_epochs burn_in_epoch fixed_label noisy_validation_size decay_start start_epoch update_start_epoch correction_epochs revision_epochs stem_padding epochs",
        "positive_integer": "batch_size candidate_k window_size window_epochs augmentations rampup_epochs hidden_dim time_dim timesteps base_width k_neighbors report_last_epochs anchor_candidates basis_epochs num_parts representation_iterations steps k num_residual_units end_epoch t_max update_interval_epochs folds total_epochs num_classes width input_channels hidden_width input_dim step_size",
        "nonnegative": "lambda lambda_volume robust_weight confidence_weight lambda_r lambda_u confidence_penalty_weight min_delta learning_rate lr initial_lr final_lr eta_min model_lr transition_lr weight_decay gamma l1_decay max_grad_norm prior_decay decay bandwidth beta coefficient basis_loss_threshold alpha delta leakiness",
        "positive": "mixup_alpha temperature basis_learning_rate warmup_lr eps epsilon scale width_multiplier",
        "probability": "rate rho_negative rho_positive high_noise_rate maximum_threshold momentum_scr momentum_scs ema_momentum dropout percentile beta1_after beta1_before initial_value start end noise_rate initial_flip_mass max_flip_mass batch_norm_momentum",
        "decay": "beta1 beta2 rho_positive rho_negative",
        "boolean": "dynamic_centroid augment strong_augment enabled allow_test_selection nesterov normalize_features classifier_bias bias ccs mda fixed_epoch_after_burn_in drop_last pin_memory download",
        "number": "momentum initial_weight lower_threshold upper_threshold",
        "milestones": "milestones step_milestones",
        "rates": "lr_values",
        "numbers": "values mean std anchor_percentages",
        "text": "path root artifact_path monitor",
    }
    leaf_rules = {name: kind for kind, names in leaf_groups.items() for name in names.split()}
    # Exact overrides distinguish parameters sharing a spelling but not a domain.
    exact = {
        "trainer.epochs": "positive_integer",
        "seed": "seed",
        "evaluation.report_last_epochs": "integer",
        "data.normalization.mean": "vector", "data.normalization.std": "vector",
        "instance_transition.anchor_percentages": "vector",
        "fine.warmup_epochs": "positive_integer",
        "loss.alpha": "positive", "loss.beta": "positive",
        "algorithm.lambda": "probability", "ca2c.lambda": "probability",
        "dld.precorrection.delta": "positive",
        "cnlcu.uncertainty.sigma_squared": "open_probability",
        "dividemix.gmm.threshold": "open_probability",
        "dividemix.mixmatch.temperature": "positive",
        "dld.diffusion.ema.decay": "decay",
        "lend.dilution.alpha": "open_probability",
        "lend.graph.gamma": "positive",
        "lend.history.beta": "probability",
        "pipeline.objective_consumer.alpha": "open_probability",
        "pipeline.objective_consumer.prior_decay": "decay",
        "pipeline.weight_provider.decay": "probability",
        "meta.virtual_learning_rate": "positive",
        "posterior_stage.bandwidth": "positive",
        "transition_stage.parameterization.initial_flip_mass": "positive",
        "transition_stage.parameterization.max_flip_mass": "half_probability",
    }
    enums = {
        "algorithm.correction": "forward pdl pdl_revision",
        "cnlcu.variant": "soft hard", "cwd.variant": "binary_scalar multiclass",
        "data.preprocessing": "standard tensor_only gce2018 l2rw",
        "data.split_strategy": "random stratified classwise_legacy numpy_choice_complement",
        "data.validation_split.strategy": "random stratified classwise_legacy numpy_choice_complement",
        "warmup.split_strategy": "random stratified classwise_legacy numpy_choice_complement",
        "data.split_rng": "default_rng numpy_legacy",
        "data.validation_split.rng": "default_rng numpy_legacy",
        "data.strong_policy": "official_cifar10 torchvision",
        "dividemix.gmm.loss_history.name": "official_auto current_epoch",
        "dld.feature_extractor.source": "repository_frozen_model external_checkpoint",
        "early_stopping.mode": "min max",
        "early_stopping.monitor": "selection_accuracy selection_loss validation_accuracy validation_loss train_accuracy train_loss test_accuracy test_loss",
        "evaluation.selection_split": "validation test",
        "evaluation.primary": "accuracy ensemble_accuracy mean_peer_accuracy",
        "lend.graph.metric": "inner_product cosine euclidean",
        "mc_ldce.feature_mode": "fixed", "mc_ldce.transition_model": "separate",
        "model.initialization": "kaiming torch_default",
        "noise.mode": "generated external clean",
        "noise.name": "symmetric pairflip class_conditional binary_asymmetric_rcn asymmetric_rcn external_torch official_uniform_flip pdl clean none native real_world",
        "noise.sampling": "global per_class transition",
        "parameter_update.compatibility_mode": "paper official_code",
        "parameter_update.critical_scope": "all_trainable matrix_and_convolution_weights",
        "pretraining_stage.mode": "train external_checkpoint",
        "selector.keep_rate.name": "linear constant",
        "transition.estimator": "paper_volmin dual_t known_smoke",
        "transition.parameterization.name": "sigmoid_off_diagonal",
    }
    for stage in (
        "dividemix.training", "dividemix.warmup", "dld.diffusion", "lend.training",
        "ensemble_stage", "final_stage", "posterior_stage", "transition_stage",
        "t_revision.stage1", "t_revision.classifier_initialization", "t_revision.revision",
        "upm.stage1", "upm.main",
    ):
        exact[stage + ".epochs"] = "positive_integer"
    method = config.get("method", "")
    if isinstance(method, Mapping):
        method = method.get("name", "")
    method = str(method).strip().lower().replace("-", "_")
    if not method:
        execution = config.get("execution", {})
        if isinstance(execution, Mapping):
            method = str(execution.get("runner", "")).strip().lower()
        algorithm = config.get("algorithm", {})
        if method == "multi_model" or (
            isinstance(algorithm, Mapping)
            and str(algorithm.get("name", "")).strip().lower() == "jocor"
        ):
            method = "jocor"
        elif method == "binary":
            method = "binary_risk"
    if method in {"coteaching", "cnlcu"}:
        exact["noise.rate"] = "decay"
        exact[method + ".noise_rate"] = "decay"
        exact["data.validation_size"] = "positive_integer"
    if method == "t_revision":
        exact["data.validation_size"] = "positive_integer"
    if method == "binary_risk":
        enums["optimizer.name"] = "sgd"
        enums["model.name"] = "linear mlp"
    if method == "cwd":
        enums["optimizer.name"] = "adam"
        enums["model.name"] = "resnet34 cifar_resnet34 tiny_cnn feature_mlp"
    model_names = "tiny_cnn cifar_cnn8 cnlcu_cnn9 resnet14 resnet32 resnet18 resnet34 resnet50 resnet101 preact_resnet18 mentor_wide_resnet feature_mlp pcse_mlp mlp linear cifar_six_conv cifar_resnet34 fine_seven_cnn ca2c_seven_cnn mc_ldce_cnn l2rw_resnet32 torchvision_resnet34"
    errors: dict[str, str] = {}
    flat: dict[str, Any] = {}

    def fail(path: str, expected: str) -> None:
        errors.setdefault(path, f"invalid config for {path}, expected {expected}")

    def walk(value: Mapping[str, Any], prefix: str = "") -> None:
        for key, current in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat[path] = current
            kind = exact.get(path, leaf_rules.get(str(key)))
            choices = enums.get(path)
            parts = path.split(".")
            if choices is not None:
                pass
            elif key == "name" and "optimizer" in parts:
                choices = "sgd adam adamw"
            elif key == "name" and "scheduler" in parts:
                choices = "none cosine multistep linear_after"
                if method == "jocor":
                    choices = "none linear_decay"
                elif method == "cal":
                    choices = "none step multistep"
            elif key == "name" and ("model" in parts or prefix.startswith("models.")):
                choices = model_names
            elif key == "name" and "loss" in parts:
                choices = "ce cross_entropy nce mae gce rce apl nce_rce nce_mae"
            if choices is not None:
                if not isinstance(current, str) or current.strip().lower() not in choices.split():
                    fail(path, "one of: " + ", ".join(choices.split()))
            elif kind is not None:
                expected, predicate = rules[kind]
                if not predicate(current):
                    fail(path, expected)
            if isinstance(current, Mapping):
                # data.preprocessing also accepts a structured tabular protocol.
                if path == "data.preprocessing":
                    errors.pop(path, None)
                walk(current, path)
            elif key == "models":
                if not isinstance(current, list) or len(current) != 2 or any(not isinstance(x, Mapping) for x in current):
                    fail(path, "a list of two model mappings")
                else:
                    for i, model in enumerate(current):
                        walk(model, f"models.{i}")

    walk(config)
    # List-valued structures need element and shape checks as well as a list type.
    for path in ("noise.transition_matrix", "pipeline.transition_estimator.matrix"):
        if path in flat:
            matrix = flat[path]
            classes = flat.get("data.num_classes")
            if not (isinstance(matrix, list) and matrix and all(
                isinstance(row, list) and len(row) == len(matrix)
                and all(number(x) and 0 <= x <= 1 for x in row)
                and math.isclose(sum(row), 1, abs_tol=1e-6) for row in matrix
            ) and (classes is None or len(matrix) == classes)):
                fail(path, "a square probability matrix matching data.num_classes, with each row summing to 1")
    special_lists = {
        "optimizer.betas": ("two finite numbers in [0, 1)", lambda v: isinstance(v, list) and len(v) == 2 and all(number(x) and 0 <= x < 1 for x in v)),
        "feature_stage.layers": ("at least two distinct layer mappings with name and pooling (global_average or flatten)", lambda v: isinstance(v, list) and len(v) >= 2 and all(isinstance(x, Mapping) and isinstance(x.get("name"), str) and x["name"].strip() and isinstance(x.get("pooling", "global_average"), str) and x.get("pooling", "global_average") in {"global_average", "flatten"} for x in v) and len({x["name"] for x in v}) == len(v)),
        "pipeline.weight_provider.dropout_schedule": ("a list of [dropout in [0, 1], positive integer duration] pairs", lambda v: isinstance(v, list) and bool(v) and all(isinstance(x, list) and len(x) == 2 and number(x[0]) and 0 <= x[0] <= 1 and type(x[1]) is int and x[1] > 0 for x in v)),
    }
    for path, (expected, predicate) in special_lists.items():
        if path in flat and not predicate(flat[path]):
            fail(path, expected)
    for path, current in flat.items():
        if path.endswith(".momentum") and "optimizer" in path.split("."):
            if not number(current) or current < 0:
                fail(path, "a finite number >= 0")
        if path.endswith(".std") and path not in errors and any(x <= 0 for x in current):
            fail(path, "a list of positive finite numbers")
        if path.endswith(".anchor_percentages") and path not in errors and any(not 0 <= x <= 100 for x in current):
            fail(path, "a list of finite numbers in [0, 100]")
        if path.endswith(".lr_values") and current is not None and path not in errors:
            owner = path.rsplit(".", 1)[0]
            milestones = flat.get(owner + ".step_milestones", flat.get(owner + ".milestones", []))
            if not isinstance(milestones, list) or len(current) != len(milestones):
                fail(path, "one positive finite learning rate per milestone")
            scheduler_name = str(flat.get(owner + ".name", "")).strip().lower()
            # L2RW applies its step schedule directly in the update loop;
            # its epoch scheduler correctly remains disabled.
            l2rw_step_schedule = (
                method == "l2rw" and owner == "scheduler"
                and owner + ".step_milestones" in flat
                and scheduler_name == "none"
            )
            if scheduler_name not in {"multistep", "step"} and not l2rw_step_schedule:
                fail(path, "a multistep or step scheduler when explicit learning rates are provided")
        if path.endswith(".nesterov") and current is True:
            owner = path.rsplit(".", 1)[0]
            momentum = flat.get(owner + ".momentum", 0.9)
            if number(momentum) and momentum <= 0:
                fail(owner + ".momentum", "a finite number > 0 when nesterov is enabled")
    q = flat.get("loss.q")
    if "loss.q" in flat and (not number(q) or not 0 < q <= 1):
        fail("loss.q", "a finite number in (0, 1]")
    classes, k = flat.get("data.num_classes"), flat.get("ca2c.candidate_k")
    if type(classes) is int and type(k) is int and not 0 < k < classes:
        fail("ca2c.candidate_k", "an integer > 0 and < data.num_classes")
    batch, k = flat.get("loader.batch_size"), flat.get("lend.graph.k")
    if type(batch) is int and type(k) is int and batch <= k:
        fail("loader.batch_size", "an integer > lend.graph.k")
    initial = flat.get("transition_stage.parameterization.initial_flip_mass")
    maximum = flat.get("transition_stage.parameterization.max_flip_mass")
    if number(initial) and number(maximum) and initial >= maximum:
        fail("transition_stage.parameterization.initial_flip_mass", "a positive finite number < transition_stage.parameterization.max_flip_mass")
    for owner in ("data.normalization",):
        mean, std = flat.get(owner + ".mean"), flat.get(owner + ".std")
        if isinstance(mean, list) and isinstance(std, list) and len(mean) != len(std):
            fail(owner + ".std", "one positive finite standard deviation per mean value")
    lower, upper = flat.get("sieve.lower_threshold"), flat.get("sieve.upper_threshold")
    if number(lower) and number(upper) and lower > upper:
        fail("sieve.upper_threshold", "a finite number >= sieve.lower_threshold")
    folds, index = flat.get("data.folds"), flat.get("data.fold_index")
    if type(folds) is int and folds < 2:
        fail("data.folds", "an integer >= 2")
    if type(folds) is int and type(index) is int and not 0 <= index < folds:
        fail("data.fold_index", "an integer >= 0 and < data.folds")
    timesteps, steps = flat.get("dld.diffusion.timesteps"), flat.get("dld.inference.steps")
    if type(timesteps) is int and type(steps) is int and timesteps < steps:
        fail("dld.diffusion.timesteps", "an integer >= dld.inference.steps")
    for path, current in flat.items():
        if path.endswith(".name") and "scheduler" in path.split(".") and current in ("linear_after", "linear_decay"):
            owner = path.rsplit(".", 1)[0]
            stage = owner.rsplit(".scheduler", 1)[0]
            budget = flat.get(stage + ".epochs", flat.get("trainer.epochs"))
            start = flat.get(owner + ".start_epoch")
            end = flat.get(owner + ".end_epoch", budget)
            if type(start) is int and type(end) is int and type(budget) is int and not 0 <= start < end <= budget:
                fail(owner + ".start_epoch", "0 <= start_epoch < end_epoch <= training epochs")
        if path.endswith(".confidence_schedule.values") and path not in errors:
            milestones = flat.get(path.rsplit(".", 1)[0] + ".milestones")
            if isinstance(milestones, list) and len(current) != len(milestones):
                fail(path, "one finite value per confidence milestone")
    if method == "importance_reweighting" and flat.get("data.name") != "uci_statlog_heart":
        for name in ("train_size", "validation_size", "test_size"):
            if name == "validation_size" and flat.get("data.validation_split.source") == "training_pool":
                continue  # Hold-out counts do not inherit the synthetic generator's parity constraint.
            value = flat.get("data." + name)
            if type(value) is int and (value < 2 or value % 2):
                fail("data." + name, "an even integer >= 2")
    for prefix in ("noise", "risk"):
        positive, negative = flat.get(prefix + ".rho_positive"), flat.get(prefix + ".rho_negative")
        if number(positive) and number(negative) and positive + negative >= 1:
            for name in ("rho_positive", "rho_negative"):
                fail(prefix + "." + name, "rates with rho_positive + rho_negative < 1")
    return list(errors.values())


def validate_config(config: Mapping[str, Any], *, check_data: bool = False) -> RunnerSpec:
    errors = _config_parameter_errors(config)
    if errors:
        raise ValueError("\n".join(errors))
    runner = resolve_runner(config)
    config = normalize_experiment_config(config)
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("configuration requires a data mapping")
    if check_data:
        data_path = data.get("root") or data.get("path")
        if data_path and not Path(str(data_path)).exists():
            raise ValueError(f"data path does not exist: {data_path}")
    if runner.name == "clean" and config.get("noise"):
        raise ValueError("clean runner rejects noise configuration")
    trainer = config.get("trainer", {}) or {}
    if trainer and not isinstance(trainer, Mapping):
        raise ValueError("trainer configuration must be a mapping")
    if isinstance(trainer, Mapping) and "epochs" in trainer and int(trainer["epochs"]) <= 0:
        raise ValueError("trainer.epochs must be positive")
    _validate_dedicated_runner(config, runner.name)
    if runner.name in {"supervised", "clean"}:
        model = config.get("model", {}) or {}
        if not isinstance(model, Mapping):
            raise ValueError("model configuration must be a mapping")
        model_name = str(model.get("name", "preact_resnet18")).lower()
        supported_models = {
            "tiny_cnn", "cifar_cnn8", "resnet14", "resnet32", "resnet18", "resnet34",
            "resnet50", "resnet101", "mentor_wide_resnet", "preact_resnet18"
        }
        if model_name not in supported_models:
            raise ValueError(
                f"unsupported model {model_name!r}; valid models: "
                + ", ".join(sorted(supported_models))
            )
        optimizer = config.get("optimizer", {}) or {}
        if optimizer and str(optimizer.get("name", "sgd")).lower() not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"unsupported optimizer: {optimizer.get('name')}")
        scheduler = config.get("scheduler", {}) or {}
        if scheduler and str(scheduler.get("name", "none")).lower() not in {"none", "cosine", "multistep"}:
            raise ValueError(f"unsupported scheduler: {scheduler.get('name')}")
        if isinstance(scheduler, Mapping) and scheduler.get("lr_values") is not None:
            milestones = scheduler.get("milestones", [])
            values = scheduler["lr_values"]
            if str(scheduler.get("name", "none")).lower() != "multistep":
                raise ValueError("scheduler.lr_values requires a multistep scheduler")
            if not isinstance(milestones, list) or not isinstance(values, list) or len(values) != len(milestones):
                raise ValueError("scheduler.lr_values must contain one value per milestone")
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in milestones) or milestones != sorted(set(milestones)):
                raise ValueError("scheduler.milestones must be sorted, distinct positive integers")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("scheduler.lr_values must contain positive finite numbers")
        try:
            from lnl_toolbox.plugins.builtin import create_builtin_catalog

            catalog = create_builtin_catalog()
            for key, kind, default in (
                ("loss", "loss", "ce"),
                ("selector", "batch_selector", "all"),
                ("parameter_update", "parameter_update_policy", "standard"),
                ("pipeline", "pipeline", ""),
            ):
                value = config.get(key, {}) or {}
                if not isinstance(value, Mapping):
                    raise ValueError(f"{key} configuration must be a mapping")
                name = str(value.get("name", default)).strip().lower()
                if name:
                    catalog.get(kind, name)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
    method = config.get("method", "")
    if isinstance(method, Mapping):
        method = method.get("name", "")
    method_name = str(method).strip().lower()
    validators = {
        "coteaching": (
            "lnl_toolbox.algorithms.coteaching.config", "CoTeachingConfig"
        ),
        "cnlcu": ("lnl_toolbox.algorithms.cnlcu.config", "CNLCUConfig"),
        "dual_t": ("lnl_toolbox.algorithms.dual_t.config", "DualTConfig"),
        "dld": ("lnl_toolbox.algorithms.dld.config", "DLDConfig"),
        "dividemix": ("lnl_toolbox.algorithms.dividemix.config", "DivideMixConfig"),
        "importance_reweighting": (
            "lnl_toolbox.algorithms.importance_reweighting.config",
            "ImportanceReweightingConfig",
        ),
        "pcse": ("lnl_toolbox.algorithms.pcse.config", "PCSEConfig"),
        "t_revision": (
            "lnl_toolbox.algorithms.t_revision.config", "TRevisionConfig"
        ),
        "lend": ("lnl_toolbox.algorithms.lend.config", "LENDConfig"),
        "upm": ("lnl_toolbox.algorithms.upm.config", "UPMConfig"),
        "volminnet": (
            "lnl_toolbox.algorithms.volminnet.config", "VolMinNetConfig"
        ),
    }
    if method_name in validators:
        from importlib import import_module

        module_name, class_name = validators[method_name]
        parsed_method_config = getattr(
            import_module(module_name), class_name
        ).from_mapping(config)
        if (
            method_name == "pcse"
            and parsed_method_config.pretraining.mode == "external_checkpoint"
        ):
            from lnl_toolbox.training.experiment import build_model
            from lnl_toolbox.training.pcse_pretrained import (
                load_pretrained_classifier_source,
            )

            num_classes = int(_require_mapping(config, "data")["num_classes"])
            model = build_model(
                dict(parsed_method_config.pretraining.model), num_classes
            )
            source = load_pretrained_classifier_source(
                parsed_method_config.pretraining.source,
                model,
                num_classes=num_classes,
            )
            source.assert_unchanged()
    pipeline = config.get("pipeline", {}) or {}
    if isinstance(pipeline, Mapping):
        provider = pipeline.get("weight_provider", {}) or {}
        if isinstance(provider, Mapping) and str(provider.get("name", "")).lower() == "mentornet":
            status = mentornet_preparation_status(config)
            assert status is not None
            if not status["artifact_ready"]:
                guidance = ""
                commands = status["commands"]
                if commands:
                    guidance = (
                        f"; Step 1 prepare Mentor features: {commands['prepare']}"
                        f"; Step 2 train MentorArtifact: {commands['train']}"
                        f"; Step 3 run Student: {commands['student']}"
                    )
                invalid = (
                    f"; artifact validation failed: {status['artifact_error']}"
                    if status["artifact_error"]
                    else ""
                )
                raise ValueError(
                    "MentorNet recipe is conditional; MentorArtifact: NOT READY: "
                    f"{status['artifact_path']}{invalid}{guidance}"
                )
    return runner


__all__ = [
    "PaperConfig",
    "PaperSpec",
    "RecipeSpec",
    "discover_recipes",
    "find_project_root",
    "load_papers",
    "load_recipe_config",
    "load_yaml",
    "mentornet_preparation_status",
    "default_paper_config",
    "paper_by_id",
    "recipe_by_id",
    "resolve_config_paths",
    "select_paper_config",
    "validate_config",
]
