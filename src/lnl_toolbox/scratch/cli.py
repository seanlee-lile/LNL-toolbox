"""Standalone command line interface for Scratch recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import execute_recipe, list_blocks, load_recipe, resolve_recipe, save_recipe, validate_recipe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lnl_toolbox.scratch.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-blocks", help="list registry blocks as JSON")
    validate = commands.add_parser("validate", help="validate a YAML recipe")
    validate.add_argument("recipe", type=Path)
    run = commands.add_parser("run", help="execute a YAML recipe")
    run.add_argument("recipe", type=Path)
    run.add_argument("--output", type=Path, default=None, help="artifact directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list-blocks":
        print(json.dumps([definition.describe() for definition in list_blocks()], ensure_ascii=False, indent=2))
        return 0
    try:
        recipe = load_recipe(args.recipe)
        validated = validate_recipe(recipe)
        if args.command == "validate":
            print(f"有效 recipe: {recipe.get('name', args.recipe.stem)} ({len(recipe['steps'])} 个顶层步骤)")
            return 0
        output = args.output or Path("artifacts") / "scratch" / str(recipe["name"])
        output.mkdir(parents=True, exist_ok=True)
        save_recipe(recipe, output / "recipe.yaml")
        save_recipe(resolve_recipe(validated), output / "resolved_recipe.yaml")
        context = execute_recipe(validated, {"artifact_dir": str(output)})
        metrics = context.get("metrics", [])
        stdout = json.dumps({"name": recipe["name"], "metrics": metrics}, ensure_ascii=False) + "\n"
        (output / "stdout.log").write_text(stdout, encoding="utf-8")
        (output / "metrics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metrics), encoding="utf-8"
        )
        print(json.dumps({"name": recipe["name"], "artifact_dir": str(output), "metrics": metrics}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"Scratch error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
