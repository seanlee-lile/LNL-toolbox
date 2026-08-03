from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from lnl_toolbox.cli import (
    PromptCancelled,
    PromptSession,
    command_arguments,
    prompt_training_selection,
)
from lnl_toolbox.training.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an LNL Toolbox classification experiment",
        epilog=(
            "Public method values include: t_revision, coteaching, cnlcu, "
            "dual_t, importance_reweighting, and pcse. Omitting method uses "
            "the standard supervised workflow."
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--epochs",
        type=int,
        help=(
            "Override the total epoch target. For method: t_revision this "
            "sets t_revision.revision.epochs; earlier stages are unchanged."
        ),
    )
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return config


def _run_arguments(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        method = config.get("method")
        method_name = (
            str(method.get("name", "")).strip().lower()
            if isinstance(method, dict)
            else str(method or "").strip().lower()
        )
        if method_name == "t_revision":
            t_revision = config.get("t_revision")
            if not isinstance(t_revision, dict):
                raise ValueError("method: t_revision requires a t_revision mapping")
            revision = t_revision.get("revision")
            if not isinstance(revision, dict):
                raise ValueError(
                    "method: t_revision requires t_revision.revision mapping"
                )
            revision["epochs"] = args.epochs
        else:
            trainer = config.setdefault("trainer", {})
            if not isinstance(trainer, dict):
                raise TypeError("trainer configuration must be a mapping")
            trainer["epochs"] = args.epochs
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
