from __future__ import annotations

"""Run all discoverable smoke recipes and build a local toolbox summary."""

import argparse
import json
from pathlib import Path
import traceback

from lnl_toolbox.catalog import (
    discover_recipes,
    load_recipe_config,
    resolve_config_paths,
    validate_config,
)
from lnl_toolbox.training.reporting import write_run_report, write_toolbox_report
from lnl_toolbox.training.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LNL smoke matrix")
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke-matrix"))
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--include-conditional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    recipes = [
        recipe
        for recipe in discover_recipes(root, include_conditional=args.include_conditional)
        if recipe.profile == "smoke"
        and (not args.methods or recipe.method in set(args.methods) or recipe.runner in set(args.methods))
    ]
    results: list[dict[str, object]] = []
    for recipe in recipes:
        record: dict[str, object] = {"recipe": recipe.id, "runner": recipe.runner, "status": "planned"}
        config = None
        try:
            config = resolve_config_paths(load_recipe_config(recipe), root)
            validate_config(config)
            if args.dry_run:
                record["status"] = "validated"
            else:
                run_dir = run_experiment(config, output / recipe.id, None)
                write_run_report(
                    run_dir,
                    config=config,
                    runner=recipe.runner,
                    method=recipe.method,
                    smoke_status="completed",
                )
                record["status"] = "completed"
                record["run_dir"] = str(run_dir)
        except Exception as exc:  # smoke matrix must record every failure
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc(limit=8)
            write_run_report(
                output / recipe.id,
                config=config,
                runner=recipe.runner,
                method=recipe.method,
                status="failed",
                smoke_status="failed",
                error=str(exc),
            )
            if args.stop_on_error:
                results.append(record)
                break
        results.append(record)
    summary = output / "smoke_matrix.json"
    summary.write_text(json.dumps({"recipes": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_toolbox_report(output, output / "report")
    failed = sum(item["status"] == "failed" for item in results)
    print(json.dumps({"recipes": len(results), "failed": failed, "summary": str(summary)}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
