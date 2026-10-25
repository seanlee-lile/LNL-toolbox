from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Mapping

from lnl_toolbox.catalog import (
    PaperSpec,
    RecipeSpec,
    load_recipe_config,
    recipe_by_id,
)
from lnl_toolbox.noise.quickstart_catalog import build_noise_config
from lnl_toolbox.data.registry import training_pool_size

from .models import QuickStartNoiseSelection


@dataclass(frozen=True, slots=True)
class MethodTemplate:
    paper_id: str
    recipe_id: str
    recipe: RecipeSpec
    config: Mapping[str, Any]


def method_template_for_paper(paper: PaperSpec) -> MethodTemplate:
    selected = next(
        (item for item in paper.configs if item.profile == "reproduction"),
        None,
    )
    if selected is None:
        raise ValueError(f"paper {paper.id!r} has no formal reproduction recipe")
    recipe = _cached_recipe_by_id(selected.recipe_id)
    return MethodTemplate(
        paper.id, selected.recipe_id, recipe, _cached_recipe_config(selected.recipe_id)
    )


@lru_cache(maxsize=None)
def _cached_recipe_by_id(recipe_id: str) -> RecipeSpec:
    return recipe_by_id(recipe_id)


def _cached_recipe_config(recipe_id: str) -> dict[str, Any]:
    recipe = _cached_recipe_by_id(recipe_id)
    stat = recipe.config_path.stat()
    return deepcopy(_recipe_config_revision(
        recipe_id, str(recipe.config_path), stat.st_mtime_ns, stat.st_size
    ))


@lru_cache(maxsize=128)
def _recipe_config_revision(recipe_id: str, config_path: str,
                            mtime_ns: int, size: int) -> dict[str, Any]:
    # Cache parsed YAML, but never reuse a plan template after its source changed.
    return load_recipe_config(_cached_recipe_by_id(recipe_id))


def _get(config: Mapping[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def _set(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _unset(config: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = config
    for key in path[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(key)
    if isinstance(current, dict):
        current.pop(path[-1], None)


def _synthetic_transition_matrix(
    selection: QuickStartNoiseSelection, num_classes: int,
) -> list[list[float]] | None:
    if selection.kind != "synthetic" or num_classes < 2:
        return None
    if selection.key == "class_conditional" and selection.transition_matrix:
        if len(selection.transition_matrix) != num_classes:
            raise ValueError("类别转移矩阵大小必须与当前数据集类别数相同")
        return [list(row) for row in selection.transition_matrix]
    if selection.key in {"symmetric", "pairflip"} and selection.rate is not None:
        rate = float(selection.rate)
        if selection.key == "symmetric":
            return [
                [1.0 - rate if row == col else rate / (num_classes - 1)
                 for col in range(num_classes)]
                for row in range(num_classes)
            ]
        return [
            [1.0 - rate if row == col else rate if col == (row + 1) % num_classes else 0.0
             for col in range(num_classes)]
            for row in range(num_classes)
        ]
    if selection.key == "binary_asymmetric_rcn" and num_classes == 2:
        if selection.rho_positive is not None and selection.rho_negative is not None:
            positive, negative = float(selection.rho_positive), float(selection.rho_negative)
            return [[1.0 - negative, negative], [positive, 1.0 - positive]]
    return None


def _binary_flip_rates(
    selection: QuickStartNoiseSelection, num_classes: int,
) -> tuple[float, float] | None:
    if selection.kind != "synthetic" or num_classes != 2:
        return None
    if selection.key == "binary_asymmetric_rcn":
        if selection.rho_positive is not None and selection.rho_negative is not None:
            return float(selection.rho_positive), float(selection.rho_negative)
    if selection.key in {"symmetric", "pairflip"} and selection.rate is not None:
        return float(selection.rate), float(selection.rate)
    return None


def _paper_method_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(config))


def configure_validation_split(config: dict[str, Any]) -> dict[str, Any]:
    """Make the Web's split count authoritative without changing stored recipes."""
    data = config.setdefault("data", {})
    if "folds" in data:
        return config  # CWD's held-out fold is a distinct experiment protocol.
    data.setdefault("validation_split", {})["source"] = "training_pool"
    if "num_val" not in data and _get(config, ("warmup", "noisy_validation_size")) is None:
        data.setdefault("validation_size", 0)
    return config


SPLIT_COUNT_PATHS = (
    ("data", "validation_size"),
    ("data", "num_val"),
    ("data", "num_clean"),
    ("warmup", "noisy_validation_size"),
)


def reference_train_size(config: Mapping[str, Any]) -> int | None:
    """Return a known source train count, never a guessed count for custom data."""

    data = config.get("data", {}) or {}
    if not isinstance(data, Mapping):
        return None
    name = str(data.get("name", "")).lower().replace("-", "_")
    if name.startswith("synthetic_"):
        size = data.get("train_size")
        return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else None
    return {
        "cifar10": 50_000, "cifar100": 50_000,
        "cifar10n": 50_000, "cifar100n": 50_000,
        "mnist": 60_000, "fashion_mnist": 60_000,
        "cifar10_airplane_automobile": 10_000,
    }.get(name)


def adapt_split_counts(config: dict[str, Any], original: Mapping[str, Any], train_count: int) -> None:
    """Preserve formal split fractions when a known corpus is replaced by a subset."""

    reference_count = reference_train_size(original)
    if not reference_count or train_count <= 0 or train_count == reference_count:
        return
    for path in SPLIT_COUNT_PATHS:
        value = _get(original, path)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            continue
        adjusted = 0 if value == 0 else max(1, math.floor(value * train_count / reference_count + 0.5))
        if path == ("data", "num_clean") and value > 0:
            classes = int(config.get("data", {}).get("num_classes", 0))
            if classes > 0:
                adjusted = max(classes, math.ceil(adjusted / classes) * classes)
        _set(config, path, min(adjusted, train_count - 1))


def _class_count_is_supported(requirements: object, num_classes: int) -> bool:
    if requirements is None:
        return True
    exact = frozenset(getattr(requirements, "exact_classes", ()))
    minimum = getattr(requirements, "min_classes", None)
    maximum = getattr(requirements, "max_classes", None)
    return not (
        (exact and num_classes not in exact)
        or (minimum is not None and num_classes < int(minimum))
        or (maximum is not None and num_classes > int(maximum))
    )


def _rebind_class_dependent_values(
    value: dict[str, Any],
    *,
    num_classes: int,
    discard_transition_matrices: bool,
    path: tuple[str, ...] = (),
) -> None:
    """Bind generic-template scalars and discard stale formal artifacts."""

    for key in tuple(value):
        child = value[key]
        child_path = path + (str(key),)
        key_name = str(key).strip().lower()
        if key_name == "num_classes" and isinstance(child, int) and not isinstance(child, bool):
            value[key] = num_classes
            continue
        if (
            key_name in {"matrix", "transition_matrix"}
            and "transition" in ".".join(child_path).lower()
            and isinstance(child, (list, tuple))
        ):
            square = len(child) == num_classes and all(
                isinstance(row, (list, tuple)) and len(row) == num_classes
                for row in child
            )
            if discard_transition_matrices or not square:
                del value[key]
            continue
        if isinstance(child, dict):
            _rebind_class_dependent_values(
                child,
                num_classes=num_classes,
                discard_transition_matrices=discard_transition_matrices,
                path=child_path,
            )


def configure_loss_correction_transition(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the paper's known-T or two-stage estimated-T configuration."""
    pipeline = config.get("pipeline", {}) or {}
    if pipeline.get("pretraining_protocol") != "loss_correction":
        return config
    model = config.get("model", {}) or {}
    if model.get("name") != "resnet32" and model.get("variant") == "loss_correction":
        model.pop("variant")
    estimator = pipeline.get("transition_estimator", {}) or {}
    name = str(estimator.get("name", "known"))
    if name == "known":
        if estimator.get("matrix") is None:
            estimator["matrix"] = deepcopy((config.get("noise", {}) or {}).get("transition_matrix"))
        estimator.pop("percentile", None)
        pipeline["warmup_epochs"] = 0
    elif name == "anchor":
        estimator.pop("matrix", None)
        if "percentile" not in estimator:
            estimator["percentile"] = None if (config.get("data", {}) or {}).get("name") == "cifar100" else 97.0
        if int(pipeline.get("warmup_epochs", 0)) <= 0:
            pipeline["warmup_epochs"] = int((config.get("trainer", {}) or {}).get("epochs", 120))
    else:
        raise ValueError("Loss Correction T source must be known or anchor")
    pipeline["transition_estimator"] = estimator
    config["pipeline"] = pipeline
    return config


def adapt_method_template(
    base_config: Mapping[str, Any],
    *,
    dataset_alias: str,
    dataset_profile: Mapping[str, object],
    noise_selection: QuickStartNoiseSelection,
    data_service,
    method_inputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Apply a local data source and one beginner noise choice to a template."""

    candidate = data_service.apply(_paper_method_config(base_config), dataset_alias)
    data = dict(candidate.get("data", {}) or {})
    data["name"] = str(dataset_profile.get("adapter") or data.get("name", ""))
    if not data["name"].startswith("synthetic_"):
        # These sizes generate synthetic samples; real adapters load their
        # registered splits and do not consume these inherited recipe fields.
        data.pop("train_size", None)
        data.pop("test_size", None)
    candidate["data"] = data
    source_dataset = str(_get(base_config, ("data", "name"), "")).replace("_", "").lower()
    selected_dataset = data["name"].replace("_", "").lower()
    if (_get(candidate, ("loss", "name")) == "gce"
            and source_dataset != selected_dataset
            and selected_dataset in {"cifar10", "cifar100"}):
        # GCE paper, Section 4.2: dataset-specific budgets and LR milestones.
        epochs, milestones = (150, [80, 120]) if selected_dataset == "cifar100" else (120, [40, 80])
        _set(candidate, ("trainer", "epochs"), epochs)
        _set(candidate, ("scheduler", "milestones"), milestones)
    if (_get(candidate, ("execution", "runner")) == "volminnet"
            and source_dataset != selected_dataset and selected_dataset == "cifar100"):
        # Paper Section 6 takes precedence over the author's ResNet-34 code.
        _set(candidate, ("volminnet", "model"), {"name": "resnet32", "base_width": 16})
        # Adam defaults; do not inherit the CIFAR-10 SGD fields.
        _set(candidate, ("volminnet", "optimizer", "transition"),
             {"name": "adam", "lr": 0.001, "weight_decay": 0.0})
    if (_get(candidate, ("execution", "runner")) == "t_revision"
            and source_dataset != selected_dataset
            and selected_dataset in {"cifar10", "cifar100"}):
        # Paper Section 4 specifies the network; author main.py supplies Stage 1 budgets.
        _set(candidate, ("t_revision", "stage1", "model", "name"),
             "resnet34" if selected_dataset == "cifar100" else "resnet18")
        _set(candidate, ("t_revision", "stage1", "epochs"),
             15 if selected_dataset == "cifar100" else 20)
    if (_get(candidate, ("execution", "runner")) == "dual_t"
            and source_dataset != selected_dataset
            and selected_dataset in {"cifar10", "cifar100"}):
        # Dual-T paper Section 4.1: ResNet-18 for CIFAR-10, ResNet-34 for CIFAR-100.
        for stage in ("posterior_stage", "final_stage"):
            _set(candidate, (stage, "model", "name"),
                 "resnet34" if selected_dataset == "cifar100" else "resnet18")
    train_count = training_pool_size(str(dataset_profile.get("adapter", "")), dict(dataset_profile.get("sample_counts_by_split") or {}))
    adapt_split_counts(candidate, base_config, train_count)
    original_noise = dict(candidate.get("noise", {}) or {})
    original_noise_name = str(original_noise.get("name", "clean")).strip().lower()
    selected_noise_name = str(noise_selection.key).strip().lower()
    noise_protocol_changed = (
        original_noise_name != selected_noise_name
        or (noise_selection.rate is not None and original_noise.get("rate") != noise_selection.rate)
        or (noise_selection.rho_positive is not None
            and original_noise.get("rho_positive") != noise_selection.rho_positive)
        or (noise_selection.rho_negative is not None
            and original_noise.get("rho_negative") != noise_selection.rho_negative)
        or (noise_selection.transition_matrix is not None
            and original_noise.get("transition_matrix") != [list(row) for row in noise_selection.transition_matrix])
        or (noise_selection.path is not None
            and original_noise.get("path") != noise_selection.path)
    )
    if noise_selection.kind in {"clean", "native"} or selected_noise_name in {"clean", "native"}:
        # Label selection is independent of artificial corruption. Keep the
        # paper's validation target contract, but discard generation inputs and
        # external manifests belonging to the template's original dataset.
        candidate["noise"] = {}
        if noise_selection.kind == "clean" or selected_noise_name == "clean":
            candidate["noise"]["name"] = "clean"
        else:
            candidate["noise"]["name"] = "native"
        if "validation_targets" in original_noise:
            candidate["noise"]["validation_targets"] = original_noise["validation_targets"]
    else:
        noise = build_noise_config(
            noise_selection.key,
            rate=noise_selection.rate,
            seed=noise_selection.seed,
            rho_positive=noise_selection.rho_positive,
            rho_negative=noise_selection.rho_negative,
            transition_matrix=noise_selection.transition_matrix,
            path=noise_selection.path,
        )
        if noise is None:
            raise ValueError(
                f"noise choice {noise_selection.key!r} needs additional method/data inputs"
            )
        if original_noise_name == selected_noise_name:
            # Preserve the selected paper's sampling/RNG and label-file keys.
            noise = {**original_noise, **noise}
        if noise_selection.key == "external_torch" and noise_selection.rate is None:
            noise.pop("rate", None)
        if "validation_targets" in original_noise:
            noise["validation_targets"] = original_noise["validation_targets"]
        candidate["noise"] = noise

    inputs = dict(method_inputs or {})
    # Use the existing runner declaration to set a known synthetic prior.  No
    # paper-specific method branch is needed here.
    from lnl_toolbox.training.runners import resolve_runner

    runner = resolve_runner(candidate)
    if runner.name == "importance_reweighting" and selected_dataset.startswith("cifar"):
        # Low-dimensional KDE/KLIEP recipes are not image posterior models.
        # Record the image adaptation in the actual config shown by Quick Start.
        data["dimension"] = 3 * 32 * 32
        candidate["data"] = data
        if candidate.get("posterior_stage", {}).get("name") in {"kde", "kliep"}:
            candidate["posterior_stage"] = {
                "name": "neural", "epochs": int(candidate["trainer"]["epochs"]),
                "early_stopping": False, "rate_estimator": "paper_raw_min",
                "model": {"name": "resnet18", "base_width": 16},
                "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001},
                "scheduler": {"name": "none"},
            }
            candidate["model"] = {"name": "resnet18", "base_width": 16, "num_classes": 2}
            candidate["configuration_fidelity"] = "engineering"
    requirements = runner.requirements(candidate)
    num_classes = int(dataset_profile.get("num_classes", 0) or 0)
    if num_classes > 1 and _class_count_is_supported(requirements, num_classes):
        data["num_classes"] = num_classes
        candidate["data"] = data
        _rebind_class_dependent_values(
            candidate,
            num_classes=num_classes,
            discard_transition_matrices=noise_protocol_changed,
        )
        if noise_selection.kind == "synthetic" and selected_noise_name == "class_conditional":
            # Cleanup removes inherited matrices, not this run's explicit input.
            candidate["noise"]["transition_matrix"] = _synthetic_transition_matrix(noise_selection, num_classes)
    if requirements is not None:
        if requirements.requires_method_noise_prior:
            known_rate = 0.0 if noise_selection.kind == "clean" else noise_selection.rate
            prior = inputs.get("noise_rate_prior", known_rate)
            for path in requirements.method_noise_prior_paths:
                if prior is None:
                    _unset(candidate, path)
                else:
                    _set(candidate, path, float(prior))
            if prior is not None and requirements.method in {"coteaching", "cnlcu"}:
                # Their keep-rate endpoint is defined by the supplied prior;
                # retaining the old paper endpoint makes an adapted run invalid.
                _set(candidate, (requirements.method, "remember_schedule", "end"), 1.0 - float(prior))
        binary_rates = _binary_flip_rates(noise_selection, num_classes)
        transition = _synthetic_transition_matrix(noise_selection, num_classes)
        for requirement in requirements.required_config_inputs:
            if requirement.code == "requires_binary_noise_prior":
                for path in requirement.paths:
                    if path[0] == "noise":
                        continue
                    if binary_rates is None:
                        _unset(candidate, path)
                    elif path[-1] in {"rho_positive", "rho_negative"}:
                        _set(candidate, path, binary_rates[path[-1] == "rho_negative"])
            elif requirement.code == "requires_transition_matrix":
                for path in requirement.paths:
                    if transition is not None:
                        _set(candidate, path, transition)
    for path_text, value in inputs.items():
        if path_text == "noise_rate_prior":
            continue
        path = tuple(str(part) for part in str(path_text).split("."))
        if path:
            _set(candidate, path, value)
    if noise_selection.seed is not None:
        candidate["seed"] = int(noise_selection.seed)
    # Loss Correction automatically uses T only when the selected noise
    # generator (or a supplied matrix) actually provides it.
    pipeline = candidate.get("pipeline", {}) or {}
    if pipeline.get("pretraining_protocol") == "loss_correction":
        known_matrix = inputs.get("pipeline.transition_estimator.matrix")
        if noise_selection.kind == "clean":
            known_matrix = [[float(row == col) for col in range(num_classes)] for row in range(num_classes)]
        if known_matrix is None:
            known_matrix = _synthetic_transition_matrix(noise_selection, num_classes)
        pipeline["transition_estimator"] = (
            {"name": "known", "matrix": known_matrix} if known_matrix is not None
            else {"name": "anchor"}
        )
        candidate["pipeline"] = pipeline
    estimator = pipeline.get("transition_estimator", {}) or {}
    if (
        (noise_selection.kind == "native" or pipeline.get("pretraining_protocol") == "loss_correction")
        and estimator.get("name") == "known"
        and "matrix" not in estimator
        and not any(str(path).startswith("pipeline.transition_estimator") for path in inputs)
    ):
        pipeline["transition_estimator"] = {"name": "anchor"}
        if int(pipeline.get("warmup_epochs", 0)) <= 0:
            pipeline["warmup_epochs"] = int((candidate.get("trainer", {}) or {}).get("epochs", 1))
        candidate["pipeline"] = pipeline
    if "pipeline.warmup_epochs" in inputs and int(inputs["pipeline.warmup_epochs"]) <= 0 and pipeline.get("transition_estimator", {}).get("name") == "anchor":
        raise ValueError("Unknown T requires positive CE pretraining epochs")
    configure_loss_correction_transition(candidate)
    from lnl_toolbox.core.config_schema import synchronize_experiment_seed

    return synchronize_experiment_seed(configure_validation_split(candidate))


def find_exact_reproduction(
    paper: PaperSpec,
    *,
    dataset_adapter: str,
    noise_selection: QuickStartNoiseSelection,
    recipes: Mapping[str, RecipeSpec] | None = None,
) -> str | None:
    """Find a formal recipe whose dataset adapter and noise choice match."""

    wanted_noise = str(noise_selection.key).strip().lower()
    for item in paper.configs:
        if item.profile != "reproduction":
            continue
        config = (
            load_recipe_config(recipes[item.recipe_id])
            if recipes is not None
            else _cached_recipe_config(item.recipe_id)
        )
        data_name = str(_get(config, ("data", "name"), "")).lower().replace("-", "_")
        noise_name = str(_get(config, ("noise", "name"), "clean")).strip().lower()
        if data_name != str(dataset_adapter).lower().replace("-", "_"):
            continue
        if noise_name != wanted_noise:
            continue
        if noise_selection.transition_matrix is not None and _get(config, ("noise", "transition_matrix")) != [list(row) for row in noise_selection.transition_matrix]:
            continue
        if noise_selection.path is not None and str(_get(config, ("noise", "path"), "")) != noise_selection.path:
            continue
        configured_rate = _get(config, ("noise", "rate"))
        if noise_selection.rate is not None and configured_rate is not None:
            if float(configured_rate) != float(noise_selection.rate):
                continue
        if wanted_noise == "binary_asymmetric_rcn" and (
            noise_selection.rho_positive is None or noise_selection.rho_negative is None
            or float(_get(config, ("noise", "rho_positive"), -1)) != noise_selection.rho_positive
            or float(_get(config, ("noise", "rho_negative"), -1)) != noise_selection.rho_negative
        ):
            continue
        if noise_selection.seed is not None and int(config.get("seed", 1)) != int(noise_selection.seed):
            continue
        return item.recipe_id
    return None


__all__ = [
    "MethodTemplate",
    "SPLIT_COUNT_PATHS",
    "adapt_split_counts",
    "adapt_method_template",
    "configure_validation_split",
    "find_exact_reproduction",
    "method_template_for_paper",
    "reference_train_size",
]
