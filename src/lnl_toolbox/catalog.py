from __future__ import annotations

"""Discoverable experiment recipes and paper-to-implementation metadata."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from lnl_toolbox.cli import repository_root
from lnl_toolbox.training.runners import RunnerSpec, resolve_runner


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


@dataclass(frozen=True, slots=True)
class PaperConfig:
    recipe_id: str
    profile: str
    variant: str
    fidelity: str


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


def load_yaml(path: Path) -> dict[str, Any]:
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
    return dict(value)


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


def discover_recipes(root: Path | None = None) -> tuple[RecipeSpec, ...]:
    project = (root or repository_root()).resolve()
    paths = sorted((project / "configs" / "experiment").glob("*.yaml"))
    paths += sorted((project / "configs" / "reproduction").glob("*.yaml"))
    recipes: list[RecipeSpec] = []
    seen: set[str] = set()
    for path in paths:
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
            )
        )
    return tuple(recipes)


def recipe_by_id(recipe_id: str, root: Path | None = None) -> RecipeSpec:
    key = recipe_id.strip().lower().replace("_", "-")
    recipes = {item.id: item for item in discover_recipes(root)}
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
        (project / "src" / "lnl_toolbox" / "paper_catalog.json").read_text(encoding="utf-8")
    )
    recipes = {item.id: item for item in discover_recipes(project)}
    papers: list[PaperSpec] = []
    for item in raw:
        configs = tuple(PaperConfig(**value) for value in item["configs"])
        # The user-facing list intentionally excludes metadata-only papers.
        if not configs or any(config.recipe_id not in recipes for config in configs):
            continue
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
        f"unknown runnable paper {paper_id!r}; valid papers: " + ", ".join(sorted(papers))
    )


def select_paper_config(
    paper: PaperSpec,
    *,
    profile: str | None = None,
    variant: str | None = None,
    root: Path | None = None,
) -> tuple[PaperConfig, RecipeSpec]:
    candidates = list(paper.configs)
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
    return config_path.resolve().parent if config_path is not None else repository_root().resolve()


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
    return resolved


def validate_config(config: Mapping[str, Any], *, check_data: bool = False) -> RunnerSpec:
    runner = resolve_runner(config)
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
    if runner.name in {"supervised", "clean"}:
        model = config.get("model", {}) or {}
        if not isinstance(model, Mapping):
            raise ValueError("model configuration must be a mapping")
        model_name = str(model.get("name", "preact_resnet18")).lower()
        supported_models = {
            "tiny_cnn", "cifar_cnn8", "resnet14", "resnet32", "resnet18", "resnet34",
            "resnet50", "resnet101", "preact_resnet18"
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
    return runner


__all__ = [
    "PaperConfig",
    "PaperSpec",
    "RecipeSpec",
    "discover_recipes",
    "find_project_root",
    "load_papers",
    "load_yaml",
    "paper_by_id",
    "recipe_by_id",
    "resolve_config_paths",
    "select_paper_config",
    "validate_config",
]
