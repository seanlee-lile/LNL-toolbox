from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lnl_toolbox.training.clean_baseline import run_clean_experiment, run_seed_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible clean-label CIFAR baseline")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, help="Override the total epoch target")
    parser.add_argument("--seeds", type=int, nargs="+", help="Run a sequential multi-seed suite")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["trainer"]["epochs"] = args.epochs
    if args.seeds:
        if args.resume:
            parser.error("--resume cannot be combined with --seeds")
        run_seed_suite(config, args.seeds, args.output_dir or Path(config.get("output_root", "artifacts/runs")) / "seed-suite")
    else:
        run_clean_experiment(config, args.output_dir, args.resume)


if __name__ == "__main__":
    main()
