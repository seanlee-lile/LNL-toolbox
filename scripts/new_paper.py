"""Create isolated developer drafts, never register or implement a paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import yaml

from lnl_toolbox.catalog import load_yaml
from lnl_toolbox.training.runners import resolve_runner


EXTENSION_TYPES = ("CONFIG_ONLY", "COMPONENT_EXTENSION", "DEDICATED_RUNNER", "PREREQUISITE_WORKFLOW")
_RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {f"{prefix}{n}" for prefix in ("com", "lpt") for n in range(1, 10)}


def create_draft(
    paper_id: str,
    output_dir: Path,
    *,
    extension_type: str = "CONFIG_ONLY",
    title: str | None = None,
    base_config: Path | None = None,
) -> Path:
    """Create one exclusive directory; all repository registration is manual."""
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", paper_id) or len(paper_id) > 64 or paper_id in _RESERVED_NAMES:
        raise ValueError("paper ID must be a lowercase kebab-case identifier (max 64), not a reserved filename")
    if extension_type not in EXTENSION_TYPES:
        raise ValueError("unknown extension type")
    output_dir = Path(output_dir).expanduser()
    if ".." in output_dir.parts:
        raise ValueError("output directory must not contain parent traversal")
    for ancestor in (output_dir, *output_dir.parents):
        if ancestor.is_symlink():
            raise ValueError("output directory must not traverse symlinks")
        if ancestor.exists() and not ancestor.is_dir():
            raise ValueError("output directory has a non-directory ancestor")
    destination = output_dir.resolve() / paper_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"draft already exists; refusing to overwrite: {destination}")

    config = {"schema_version": 1, "kind": "fragment"}
    if base_config is not None:
        config = load_yaml(Path(base_config))
        if config.get("kind") != "experiment":
            raise ValueError("base config must be a real experiment")
        runner = resolve_runner(config)
        requirements = runner.requirements(config)
        if runner.name == "clean" and requirements is None:
            requirements = resolve_runner(dict(config, execution={"runner": "supervised"})).requirements(config)
        if requirements is None:
            raise ValueError("base config has no effective requirements")

    relative = "configs/experiment/" + paper_id.replace("-", "_") + "_draft.yaml"
    recipe_id = Path(relative).stem.replace("_", "-")
    todo = [
        "Review paper metadata, mechanism, limitations and bibliography.",
        "Implement or explicitly select real components/runner; declare audited requirements.",
        "Review data, budgets, prerequisite consumer and resume contracts; do not guess them.",
        "Add deterministic tests and run registration consistency checks.",
        "Manually review catalog fragments and packaging before registration; generation does not register anything.",
    ]
    if base_config is None:
        todo.insert(1, "Replace fragment recipe with a real experiment; no algorithm/requirements are generated.")
    paper = {
        "id": paper_id, "acronym": paper_id, "title": title or "TODO: paper title",
        "venue": "TODO", "year": 0, "source_url": "TODO",
        "summary": "TODO: problem addressed", "mechanism": "TODO: actual implemented mechanism",
        "lifecycle": [], "limitations": ["DRAFT: not registered; numerical reproduction not verified."],
        "configs": ([{"recipe_id": recipe_id, "profile": "experiment", "variant": "draft", "configuration_fidelity": "engineering", "reproduction_status": "not_run", "availability": "unavailable"}] if base_config else []),
        "concept_to_config": [], "implementation_paths": [],
        "implementation_status": "draft", "reproduction_status": "not_run", "availability": "unavailable",
    }
    manifest = {
        "schema_version": 1, "kind": "paper_registration_draft", "paper_id": paper_id,
        "extension_type": extension_type, "registered": False,
        "paper_file": "paper.json", "recipe_catalog_file": "recipe_catalog.fragment.json",
        "recipe_file": relative, "todo": todo,
    }
    recipe_fragment = {"recipes": [relative], "public": [], "conditional": []}
    packaging = {"destination": "share/lnl-toolbox/configs/experiment", "files": [relative]}
    readme = (
        f"# {title or paper_id} — developer draft\n\n"
        "NOT REGISTERED. This directory is not read by the toolbox catalogs.\n"
        "No method implementation, requirements, readiness or reproduction claim is generated.\n\n"
        "An explicitly selected base config is copied unchanged, including its training budget and paths. "
        "The draft filename does not make it a smoke protocol. Do not run it without reviewing it.\n\n"
        "Validate with `python scripts/check_paper_registration.py --draft <this-directory>`. "
        "TODOs are reported as incomplete (exit 1), not registration success.\n\n"
        "## Acceptance checklist\n" + "".join(f"- [ ] {item}\n" for item in todo) +
        "\nAfter implementation/review, manually integrate the fragments, review pyproject data-files, "
        "and run the repository checker. Do not copy generated placeholders into a release.\n"
    )
    files = {
        "draft.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        "paper.json": json.dumps(paper, ensure_ascii=False, indent=2) + "\n",
        "recipe_catalog.fragment.json": json.dumps(recipe_fragment, indent=2) + "\n",
        "packaging.fragment.json": json.dumps(packaging, indent=2) + "\n",
        relative: "# DRAFT: review all TODOs before use; no automatic registration.\n" + yaml.safe_dump(config, sort_keys=False),
        "reproduction.md": readme,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # exclusive; never reuse an existing draft
    for name, contents in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(contents)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_id")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper-drafts"))
    parser.add_argument("--extension-type", choices=EXTENSION_TYPES, default="CONFIG_ONLY")
    parser.add_argument("--title")
    parser.add_argument("--base-config", type=Path, help="explicitly copy an existing experiment without changing its budget")
    args = parser.parse_args(argv)
    try:
        path = create_draft(args.paper_id, args.output_dir, extension_type=args.extension_type, title=args.title, base_config=args.base_config)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.exit(2, f"Draft creation failed: {error}\n")
    print(f"Draft created: {path}\nNOT REGISTERED — review TODOs and catalog/package fragments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
