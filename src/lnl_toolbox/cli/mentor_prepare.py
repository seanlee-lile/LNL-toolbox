from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from lnl_toolbox.training.mentor_learning import prepare_trusted_mentor_features


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare isolated MentorNet trusted data")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Mentor preparation configuration must be a mapping")
    prepare_trusted_mentor_features(config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
