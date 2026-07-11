from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lnl_toolbox.training.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LNL Toolbox classification experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, help="Override the total epoch target")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["trainer"]["epochs"] = args.epochs
    run_experiment(config, args.output_dir, args.resume)


if __name__ == "__main__":
    main()
