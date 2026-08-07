from __future__ import annotations

"""Clean-only compatibility wrappers around the unified supervised runner."""

from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from lnl_toolbox.training.experiment import (
    build_model,
    build_optimizer,
    build_scheduler,
    run_supervised_experiment,
)


build_clean_model = build_model
build_clean_optimizer = build_optimizer
build_clean_scheduler = build_scheduler


def run_clean_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    if config.get("noise"):
        raise ValueError(
            "Clean baseline does not accept noise configuration; use lnl-train for noisy-label runs"
        )
    return run_supervised_experiment(config, output_dir, resume)


def run_seed_suite(config: dict[str, Any], seeds: list[int], output_dir: str | Path) -> Path:
    if config.get("noise"):
        raise ValueError("Multi-seed clean baseline does not accept noise configuration")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        current = deepcopy(config)
        current["seed"] = int(seed)
        run_dir = run_clean_experiment(current, root / f"seed-{seed}")
        result = json.loads(
            (run_dir / "final_metrics.json").read_text(encoding="utf-8")
        )
        results.append({"seed": seed, **result})
    accuracies = np.asarray(
        [row["test_accuracy"] for row in results], dtype=np.float64
    )
    summary = {
        "seeds": seeds,
        "runs": results,
        "test_accuracy_mean": float(accuracies.mean()),
        "test_accuracy_std": float(accuracies.std(ddof=1))
        if len(accuracies) > 1
        else 0.0,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "best_epoch",
                "best_validation_accuracy",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    return root
