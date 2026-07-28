from __future__ import annotations

"""Dependency-light comparison of reproduction and paper learning curves."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def load_metrics_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping) and value.get("event") == "epoch":
            rows.append(dict(value))
    return rows


def _series(values: Iterable[Any], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"curve {name!r} must contain finite values")
    return result


def _paper_series(paper_curve: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]], metric: str) -> list[float]:
    if isinstance(paper_curve, Mapping):
        if metric not in paper_curve:
            raise KeyError(f"paper curve is missing metric {metric!r}")
        return _series(paper_curve[metric], metric)
    return _series((row[metric] for row in paper_curve), metric)


def compare_curves(
    reproduced: Sequence[Mapping[str, Any]],
    paper_curve: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
    *,
    metric: str = "validation_accuracy",
) -> dict[str, Any]:
    """Align by one-based epoch and report overlap errors without interpolation."""

    if not reproduced:
        raise ValueError("reproduced curve must not be empty")
    ours = _series((row[metric] for row in reproduced), metric)
    paper = _paper_series(paper_curve, metric)
    count = min(len(ours), len(paper))
    differences = [ours[index] - paper[index] for index in range(count)]
    absolute = [abs(value) for value in differences]
    return {
        "metric": metric,
        "reproduced_epochs": len(ours),
        "paper_epochs": len(paper),
        "overlap_epochs": count,
        "final_reproduced": ours[-1],
        "final_paper": paper[-1],
        "best_reproduced": max(ours),
        "best_paper": max(paper),
        "mean_absolute_error": sum(absolute) / count,
        "max_absolute_error": max(absolute),
        "differences": differences,
    }


def _svg_line(values: Sequence[float], *, color: str, width: float = 760.0) -> str:
    minimum = min(values)
    maximum = max(values)
    span = max(maximum - minimum, 1e-12)
    denominator = max(len(values) - 1, 1)
    points = " ".join(
        f"{80 + width * index / denominator:.2f},{300 - 220 * (value - minimum) / span:.2f}"
        for index, value in enumerate(values)
    )
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>'


def write_curve_comparison(
    reproduced: Sequence[Mapping[str, Any]],
    paper_curve: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
    output_dir: str | Path,
    *,
    metric: str = "validation_accuracy",
) -> dict[str, Path]:
    """Write overlay SVG, per-epoch difference CSV, and JSON summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ours = _series((row[metric] for row in reproduced), metric)
    paper = _paper_series(paper_curve, metric)
    summary = compare_curves(reproduced, paper_curve, metric=metric)
    overlay = output / "curve_comparison.svg"
    overlay.write_text(
        "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360">',
            '<rect width="900" height="360" fill="white"/>',
            f'<text x="450" y="30" text-anchor="middle">{metric} comparison</text>',
            _svg_line(ours, color="#2563eb"),
            _svg_line(paper, color="#dc2626"),
            '<text x="80" y="345" fill="#2563eb">reproduction</text>',
            '<text x="210" y="345" fill="#dc2626">paper</text>',
            '</svg>',
        ]),
        encoding="utf-8",
    )
    difference = output / "curve_difference.csv"
    count = min(len(ours), len(paper))
    with difference.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "reproduced", "paper", "difference"])
        for index in range(count):
            writer.writerow([index + 1, ours[index], paper[index], ours[index] - paper[index]])
    summary_path = output / "curve_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"overlay": overlay, "difference": difference, "summary": summary_path}


__all__ = ["compare_curves", "load_metrics_jsonl", "write_curve_comparison"]
