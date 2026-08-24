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
    run.add_argument("--max-epochs", type=int, default=None, help="runtime-only cap; does not modify the recipe")
    run.add_argument("--max-batches", type=int, default=None, help="runtime-only cap for each loader loop")
    run.add_argument("--skip-final-test", action="store_true", help="skip the final test evaluation for structural validation")
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
        limits = {
            "max_epochs": args.max_epochs,
            "max_batches": args.max_batches,
            "skip_final_test": bool(args.skip_final_test),
        }
        context = execute_recipe(validated, {"artifact_dir": str(output)}, runtime_limits=limits)
        metrics = context.get("metrics", [])
        stdout = json.dumps({"name": recipe["name"], "metrics": metrics}, ensure_ascii=False) + "\n"
        (output / "stdout.log").write_text(stdout, encoding="utf-8")
        (output / "metrics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metrics), encoding="utf-8"
        )
        print(json.dumps({"name": recipe["name"], "artifact_dir": str(output), "metrics": metrics, "runtime_limits": limits}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"Scratch error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
