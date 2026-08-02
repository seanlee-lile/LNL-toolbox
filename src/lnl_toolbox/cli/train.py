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
from lnl_toolbox.training.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an LNL Toolbox classification experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, help="Override the total epoch target")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return resolve_config_paths(load_yaml(resolved), find_project_root(resolved))


def _run_arguments(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.epochs is not None:
        config["trainer"]["epochs"] = args.epochs
    run_experiment(config, args.output_dir, args.resume)


def main(
    argv: Sequence[str] | None = None,
    session: PromptSession | None = None,
) -> int:
    arguments = command_arguments(argv)
    active_session = session or PromptSession()
    try:
        if not arguments:
            selection = prompt_training_selection(active_session, clean=False)
            if selection is None:
                return 0
            run_experiment(selection.config, selection.output_dir, selection.resume)
        else:
            _run_arguments(build_parser().parse_args(arguments))
    except PromptCancelled:
        active_session.write("\n已取消。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
