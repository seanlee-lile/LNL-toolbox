from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from lnl_toolbox.training.mentor_learning import train_mentor_artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a reusable Mentor artifact")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Mentor training configuration must be a mapping")
    train_mentor_artifact(config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
