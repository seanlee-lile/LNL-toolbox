from __future__ import annotations

"""Run the standalone Dual-T evidence chain without registering a method."""

import argparse
from pathlib import Path
import sys

import yaml

from lnl_toolbox.training.dual_t_evidence_experiment import (
    run_dual_t_evidence_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("experiment configuration must be a mapping")
    destination = run_dual_t_evidence_experiment(
        values,
        arguments.output_dir,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
