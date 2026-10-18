"""Read-only developer checks; never load datasets or external artifacts."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path, PurePosixPath
import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from lnl_toolbox.catalog import load_yaml
from lnl_toolbox.training.runners import resolve_runner


@dataclass(frozen=True)
class Issue:
    code: str
    location: str
    message: str


def _json(path: Path, issues: list[Issue]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        issues.append(Issue("catalog_unreadable", str(path), str(error)))
        return None


def _relative_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("expected a nonempty repository-relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ":" in value or ".." in relative.parts:
        raise ValueError("path must stay inside the repository")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path resolves outside the repository")
    if not path.is_file():
        raise ValueError("referenced file does not exist")
    return path


def check_repository(root: Path) -> list[Issue]:
    """Check registrations structurally, without config preflight/consumer IO."""
    root = root.resolve()
    issues: list[Issue] = []
    manifest = _json(root / "src/lnl_toolbox/cli/data/recipe_catalog.json", issues)
    papers = _json(root / "src/lnl_toolbox/paper_catalog.json", issues)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("recipes"), list):
        issues.append(Issue("recipe_catalog_schema", "recipe_catalog", "recipes must be a list"))
        return issues
    if not isinstance(papers, list):
        issues.append(Issue("paper_catalog_schema", "paper_catalog", "expected a list"))
        return issues
    paths = manifest["recipes"]
    known_paths = {value for value in paths if isinstance(value, str)}
    for value in manifest.get("conditional", []):
        if value not in known_paths:
            issues.append(Issue("unknown_catalog_reference", "conditional", str(value)))
    for entry in manifest.get("public", []):
        value = entry.get("path") if isinstance(entry, Mapping) else None
        if value not in known_paths:
            issues.append(Issue("unknown_catalog_reference", "public", str(value)))
    try:
        packaging = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        package_files = packaging.get("tool", {}).get("setuptools", {}).get("data-files", {})
        if not isinstance(package_files, Mapping):
            raise ValueError("setuptools data-files must be a mapping")
    except (OSError, ValueError) as error:
        issues.append(Issue("packaging_unreadable", "pyproject.toml", str(error)))
        package_files = {}
    seen_paths: set[str] = set()
    recipes: dict[str, tuple[str, dict]] = {}
    formal_paths: dict[str, str] = {}
    for relative in paths:
        try:
            path = _relative_file(root, relative)
        except ValueError as error:
            issues.append(Issue("recipe_path_invalid", str(relative), str(error)))
            continue
        if relative in seen_paths:
            issues.append(Issue("duplicate_recipe_path", relative, "listed more than once"))
            continue
        seen_paths.add(relative)
        try:
            config = load_yaml(path)
            if config.get("kind") != "experiment":
                continue  # producer/fragment configs are not standalone recipes
            recipe_id = path.stem.lower().replace("_", "-")
            if recipe_id in recipes:
                issues.append(Issue("duplicate_recipe_id", recipe_id, relative))
                continue
            recipes[recipe_id] = (relative, config)
            if "reproduction" in path.stem or path.parent.name == "reproduction":
                formal_paths[relative] = recipe_id
            runner = resolve_runner(config)
        except (KeyError, TypeError, ValueError, ImportError) as error:
            issues.append(Issue("recipe_resolution_failed", relative, str(error)))
            continue
        try:
            requirements = runner.requirements(config)
            # clean is a compatibility wrapper; its actual execution delegates
            # to supervised. Check that real provider rather than inventing one.
            if runner.name == "clean" and requirements is None:
                candidate = dict(config, execution={"runner": "supervised"})
                requirements = resolve_runner(candidate).requirements(candidate)
            if requirements is None:
                raise ValueError("runner has no effective MethodRequirements")
        except (KeyError, TypeError, ValueError, ImportError) as error:
            issues.append(Issue("requirements_unavailable", relative, str(error)))
    seen_papers: set[str] = set()
    for paper in papers:
        if not isinstance(paper, Mapping) or not isinstance(paper.get("id"), str):
            issues.append(Issue("paper_catalog_schema", "paper_catalog", "paper requires an id"))
            continue
        paper_id = paper["id"]
        if paper_id in seen_papers:
            issues.append(Issue("duplicate_paper_id", paper_id, "listed more than once"))
        seen_papers.add(paper_id)
        configs = paper.get("configs")
        if not isinstance(configs, list):
            issues.append(Issue("paper_catalog_schema", paper_id, "configs must be a list (empty is allowed)"))
            continue
        seen_configs: set[str] = set()
        for entry in configs:
            recipe_id = entry.get("recipe_id") if isinstance(entry, Mapping) else None
            if not isinstance(recipe_id, str) or recipe_id not in recipes:
                issues.append(Issue("paper_recipe_missing", paper_id, str(recipe_id)))
                continue
            if recipe_id in seen_configs:
                issues.append(Issue("duplicate_paper_config", paper_id, recipe_id))
            seen_configs.add(recipe_id)
            relative, config = recipes[recipe_id]
            formal = entry.get("profile") == "reproduction"
            formal = formal or "reproduction" in PurePosixPath(relative).stem or PurePosixPath(relative).parent.name == "reproduction"
            if formal:
                formal_paths[relative] = recipe_id
    for relative, recipe_id in formal_paths.items():
        destination = "share/lnl-toolbox/" + PurePosixPath(relative).parent.as_posix()
        patterns = package_files.get(destination, [])
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns):
            issues.append(Issue("formal_recipe_not_packaged", recipe_id, relative))
    return issues


def check_draft(directory: Path) -> list[Issue]:
    """Validate linked draft materials; outstanding TODOs never mean READY."""
    directory = directory.resolve()
    issues: list[Issue] = []
    draft = _json(directory / "draft.json", issues)
    if not isinstance(draft, Mapping) or draft.get("kind") != "paper_registration_draft" or draft.get("schema_version") != 1 or draft.get("registered") is not False:
        return issues + [Issue("draft_schema", "draft.json", "expected an unregistered version-1 draft")]
    paper_id = draft.get("paper_id", "")
    if not isinstance(paper_id, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", paper_id):
        issues.append(Issue("draft_schema", "paper_id", "expected lowercase kebab-case ID"))
    try:
        paper_path = _relative_file(directory, draft.get("paper_file"))
        manifest_path = _relative_file(directory, draft.get("recipe_catalog_file"))
        recipe_path = _relative_file(directory, draft.get("recipe_file"))
    except ValueError as error:
        return issues + [Issue("draft_path_invalid", "draft.json", str(error))]
    paper = _json(paper_path, issues)
    manifest = _json(manifest_path, issues)
    if not isinstance(paper, Mapping) or paper.get("id") != paper_id:
        issues.append(Issue("draft_paper_mismatch", str(paper_path), "paper id must match draft id"))
        return issues
    required_fields = {"acronym", "title", "venue", "year", "source_url", "summary", "mechanism", "lifecycle", "limitations", "configs", "concept_to_config", "implementation_paths"}
    if required_fields - paper.keys():
        issues.append(Issue("draft_paper_schema", str(paper_path), "missing paper metadata fields"))
    relative = draft["recipe_file"]
    if not isinstance(manifest, Mapping) or manifest.get("recipes") != [relative]:
        issues.append(Issue("draft_recipe_mismatch", str(manifest_path), "recipe fragment must reference the draft recipe"))
    try:
        config = load_yaml(recipe_path)
        recipe_id = recipe_path.stem.lower().replace("_", "-")
        for entry in paper.get("configs", []):
            if not isinstance(entry, Mapping) or entry.get("recipe_id") != recipe_id:
                issues.append(Issue("paper_recipe_missing", str(paper_path), "paper references another recipe"))
        if config.get("kind") != "experiment":
            issues.append(Issue("draft_incomplete", relative, "fragment needs an implemented experiment and requirements"))
        else:
            runner = resolve_runner(config)
            requirements = runner.requirements(config)
            if runner.name == "clean" and requirements is None:
                requirements = resolve_runner(dict(config, execution={"runner": "supervised"})).requirements(config)
            if requirements is None:
                issues.append(Issue("requirements_unavailable", relative, "no effective requirements"))
    except (KeyError, TypeError, ValueError, ImportError) as error:
        issues.append(Issue("recipe_resolution_failed", relative, str(error)))
    if draft.get("todo") or any(isinstance(v, str) and v.startswith("TODO") for v in paper.values()):
        issues.append(Issue("draft_incomplete", str(paper_path), "review outstanding TODOs; this is not a registered paper"))
    if paper.get("reproduction_status") != "not_run":
        issues.append(Issue("draft_reproduction_claim", str(paper_path), "draft must not claim numerical reproduction"))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--draft", type=Path, help="check isolated draft materials, not live registration")
    args = parser.parse_args(argv)
    issues = check_draft(args.draft) if args.draft else check_repository(args.root)
    for issue in issues:
        print(f"{issue.code}: {issue.location}: {issue.message}")
    label = "Draft materials (NOT REGISTERED)" if args.draft else "Registration consistency"
    print(f"{label}: {'INCOMPLETE/INVALID' if issues else 'PASS'} ({len(issues)} issues)")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
