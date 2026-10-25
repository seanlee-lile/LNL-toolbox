from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from lnl_toolbox.catalog import (
    PaperSpec,
    RecipeSpec,
    load_recipe_config,
    recipe_by_id,
)
from lnl_toolbox.noise.quickstart_catalog import build_noise_config

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
def _recipe_config_revision(
    recipe_id: str, config_path: str, mtime_ns: int, size: int,
) -> dict[str, Any]:
    """Cache parsed YAML by source revision; callers receive isolated copies."""
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


def _paper_method_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(config))


def _unset(config: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = config
    for key in path[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(key)
    if isinstance(current, dict):
        current.pop(path[-1], None)


def _synthetic_transition_matrix(selection: QuickStartNoiseSelection, classes: int) -> list[list[float]] | None:
    if classes < 2:
        return None
    if selection.kind == "clean":
        return [[float(row == col) for col in range(classes)] for row in range(classes)]
    if selection.kind != "synthetic":
        return None
    if selection.key == "class_conditional" and selection.transition_matrix is not None:
        if len(selection.transition_matrix) != classes:
            raise ValueError("类别转移矩阵大小必须与当前数据集类别数相同")
        return [list(row) for row in selection.transition_matrix]
    if selection.key in {"symmetric", "pairflip"} and selection.rate is not None:
        rate = float(selection.rate)
        return [[1 - rate if row == col else (
            rate / (classes - 1) if selection.key == "symmetric" else float(col == (row + 1) % classes) * rate
        ) for col in range(classes)] for row in range(classes)]
    if selection.key == "binary_asymmetric_rcn" and classes == 2:
        if selection.rho_positive is not None and selection.rho_negative is not None:
            positive, negative = float(selection.rho_positive), float(selection.rho_negative)
            return [[1 - negative, negative], [positive, 1 - positive]]
    return None


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
    candidate["data"] = data
    original_noise = dict(candidate.get("noise", {}) or {})
    original_noise_name = str(original_noise.get("name", "clean")).strip().lower()
    selected_noise_name = str(noise_selection.key).strip().lower()
    noise_protocol_changed = (
        original_noise_name != selected_noise_name
        or any(value is not None and original_noise.get(key) != value for key, value in (
            ("rate", noise_selection.rate), ("rho_positive", noise_selection.rho_positive),
            ("rho_negative", noise_selection.rho_negative), ("path", noise_selection.path),
            ("transition_matrix", None if noise_selection.transition_matrix is None else [list(row) for row in noise_selection.transition_matrix]),
        ))
    )
    if noise_selection.kind in {"clean", "native"} or selected_noise_name in {"clean", "native"}:
        # No synthetic generator is configured for clean/native labels. In
        # particular, a literal name='clean' must not satisfy configured-noise inputs.
        candidate.pop("noise", None)
        if "validation_targets" in original_noise:
            candidate["noise"] = {"validation_targets": original_noise["validation_targets"]}
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
            # Retain valid sampler/label-key choices, never stale source paths or matrices.
            for key in ("rng", "clean_key", "noisy_key"):
                if key in original_noise:
                    noise[key] = original_noise[key]
        if "validation_targets" in original_noise:
            noise["validation_targets"] = original_noise["validation_targets"]
        candidate["noise"] = noise

    inputs = dict(method_inputs or {})
    # Use the existing runner declaration to set a known synthetic prior.  No
    # paper-specific method branch is needed here.
    from lnl_toolbox.training.runners import resolve_runner

    runner = resolve_runner(candidate)
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
        if selected_noise_name == "class_conditional" and noise_selection.kind == "synthetic":
            candidate["noise"]["transition_matrix"] = _synthetic_transition_matrix(noise_selection, num_classes)
    if requirements is not None and requirements.requires_method_noise_prior:
        prior = inputs.get("noise_rate_prior", 0.0 if noise_selection.kind == "clean" else noise_selection.rate)
        for path in requirements.method_noise_prior_paths:
            if prior is None:
                _unset(candidate, path)
            else:
                _set(candidate, path, float(prior))
        if prior is not None and requirements.method in {"coteaching", "cnlcu"}:
            # Existing validators require this endpoint to match the supplied prior.
            _set(candidate, (requirements.method, "remember_schedule", "end"), 1.0 - float(prior))
    if requirements is not None:
        transition = _synthetic_transition_matrix(noise_selection, num_classes)
        binary_rates = None
        if num_classes == 2 and noise_selection.kind == "synthetic":
            if selected_noise_name == "binary_asymmetric_rcn":
                binary_rates = (noise_selection.rho_positive, noise_selection.rho_negative)
            elif selected_noise_name in {"symmetric", "pairflip"}:
                binary_rates = (noise_selection.rate, noise_selection.rate)
        for requirement in requirements.required_config_inputs:
            if requirement.code == "requires_transition_matrix":
                for path in requirement.paths:
                    if transition is None:
                        _unset(candidate, path)
                    else:
                        _set(candidate, path, transition)
            elif requirement.code == "requires_binary_noise_prior":
                for path in requirement.paths:
                    if path[0] == "noise":
                        continue
                    if binary_rates is None or any(rate is None for rate in binary_rates):
                        _unset(candidate, path)
                    elif path[-1] in {"rho_positive", "rho_negative"}:
                        _set(candidate, path, float(binary_rates[path[-1] == "rho_negative"]))
    for path_text, value in inputs.items():
        if path_text == "noise_rate_prior":
            continue
        path = tuple(str(part) for part in str(path_text).split("."))
        if path:
            _set(candidate, path, value)
    return candidate


def find_exact_reproduction(
    paper: PaperSpec,
    *,
    dataset_adapter: str,
    noise_selection: QuickStartNoiseSelection,
) -> str | None:
    """Find a formal recipe whose dataset adapter and noise choice match."""

    wanted_noise = str(noise_selection.key).strip().lower()
    for item in paper.configs:
        if item.profile != "reproduction":
            continue
        config = _cached_recipe_config(item.recipe_id)
        data_name = str(_get(config, ("data", "name"), "")).lower().replace("-", "_")
        noise_name = str(_get(config, ("noise", "name"), "clean")).strip().lower()
        if data_name != str(dataset_adapter).lower().replace("-", "_"):
            continue
        if noise_name != wanted_noise:
            continue
        if wanted_noise == "class_conditional" and noise_selection.transition_matrix is None:
            continue
        if wanted_noise == "external_torch" and not noise_selection.path:
            continue
        if noise_selection.transition_matrix is not None and _get(config, ("noise", "transition_matrix")) != [list(row) for row in noise_selection.transition_matrix]:
            continue
        if noise_selection.path is not None and str(_get(config, ("noise", "path"), "")) != noise_selection.path:
            continue
        if wanted_noise == "binary_asymmetric_rcn" and (
            noise_selection.rho_positive is None or noise_selection.rho_negative is None
            or _get(config, ("noise", "rho_positive")) != noise_selection.rho_positive
            or _get(config, ("noise", "rho_negative")) != noise_selection.rho_negative
        ):
            continue
        configured_rate = _get(config, ("noise", "rate"))
        if noise_selection.rate is not None and configured_rate is not None:
            if float(configured_rate) != float(noise_selection.rate):
                continue
        return item.recipe_id
    return None


__all__ = [
    "MethodTemplate",
    "adapt_method_template",
    "find_exact_reproduction",
    "method_template_for_paper",
]
