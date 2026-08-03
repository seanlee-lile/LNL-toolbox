from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from lnl_toolbox.training.instance_transition_experiment import run_instance_transition_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a staged instance-transition experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must contain a YAML mapping")
    if arguments.epochs is not None:
        config["trainer"]["epochs"] = int(arguments.epochs)
    run_instance_transition_experiment(config, arguments.output_dir, arguments.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
