from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from lnl_toolbox.catalog import (
    PaperSpec,
    RecipeSpec,
    default_paper_config,
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
    selected, recipe = default_paper_config(paper)
    return MethodTemplate(paper.id, selected.recipe_id, recipe, load_recipe_config(recipe))


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
    if noise_selection.kind in {"clean", "native"} or noise_selection.key in {"clean", "native"}:
        candidate.pop("noise", None)
    else:
        noise = build_noise_config(
            noise_selection.key,
            rate=noise_selection.rate,
            seed=noise_selection.seed,
        )
        if noise is None:
            raise ValueError(
                f"noise choice {noise_selection.key!r} needs additional method/data inputs"
            )
        candidate["noise"] = noise

    inputs = dict(method_inputs or {})
    # Use the existing runner declaration to set a known synthetic prior.  No
    # paper-specific method branch is needed here.
    from lnl_toolbox.training.runners import resolve_runner

    runner = resolve_runner(candidate)
    requirements = runner.requirements(candidate)
    if requirements is not None and requirements.requires_method_noise_prior:
        prior = inputs.get("noise_rate_prior", noise_selection.rate)
        if prior is not None:
            for path in requirements.method_noise_prior_paths:
                _set(candidate, path, float(prior))
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

    wanted_noise = "clean" if noise_selection.kind in {"clean", "native"} else noise_selection.key
    for item in paper.configs:
        if item.profile != "reproduction":
            continue
        recipe = recipe_by_id(item.recipe_id)
        config = load_recipe_config(recipe)
        data_name = str(_get(config, ("data", "name"), "")).lower().replace("-", "_")
        noise_name = str(_get(config, ("noise", "name"), "clean"))
        if data_name != str(dataset_adapter).lower().replace("-", "_"):
            continue
        if noise_name != wanted_noise:
            continue
        configured_rate = _get(config, ("noise", "rate"))
        if noise_selection.rate is not None and configured_rate is not None:
            if float(configured_rate) != float(noise_selection.rate):
                continue
        return recipe.id
    return None


__all__ = [
    "MethodTemplate",
    "adapt_method_template",
    "find_exact_reproduction",
    "method_template_for_paper",
]
