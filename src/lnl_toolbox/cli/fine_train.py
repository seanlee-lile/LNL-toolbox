from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lnl_toolbox.training.fine_experiment import run_fine_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an SED+FINE noisy-label experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("FINE configuration must contain a YAML mapping")
    run_fine_experiment(config, args.output_dir, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
