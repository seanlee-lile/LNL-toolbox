from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lnl_toolbox.cli import (
    PromptCancelled,
    PromptSession,
    command_arguments,
    prompt_training_selection,
)
from lnl_toolbox.catalog import find_project_root, load_yaml, resolve_config_paths
from lnl_toolbox.training.clean_baseline import run_clean_experiment, run_seed_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible clean-label CIFAR baseline")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, help="Override the total epoch target")
    parser.add_argument("--seeds", type=int, nargs="+", help="Run a sequential multi-seed suite")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return resolve_config_paths(load_yaml(resolved), find_project_root(resolved))


def _run_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    config = _load_config(args.config)
    if args.epochs is not None:
        config["trainer"]["epochs"] = args.epochs
    if args.seeds:
        if args.resume:
            parser.error("--resume cannot be combined with --seeds")
        run_seed_suite(config, args.seeds, args.output_dir or Path(config.get("output_root", "artifacts/runs")) / "seed-suite")
    else:
        run_clean_experiment(config, args.output_dir, args.resume)


def main(
    argv: Sequence[str] | None = None,
    session: PromptSession | None = None,
) -> int:
    arguments = command_arguments(argv)
    active_session = session or PromptSession()
    try:
        if not arguments:
            selection = prompt_training_selection(active_session, clean=True)
            if selection is None:
                return 0
            if selection.seeds:
                output_dir = selection.output_dir or (
                    Path(selection.config.get("output_root", "artifacts/runs")) / "seed-suite"
                )
                run_seed_suite(selection.config, selection.seeds, output_dir)
            else:
                run_clean_experiment(selection.config, selection.output_dir, selection.resume)
        else:
            parser = build_parser()
            _run_arguments(parser.parse_args(arguments), parser)
    except PromptCancelled:
        active_session.write("\n已取消。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
