from __future__ import annotations

"""Dependency-light comparison of experiment and reference curves."""

import csv
from html import escape
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_metrics_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid metrics JSON at line {line_number}"
            ) from exc
        if isinstance(value, Mapping) and value.get("event") == "epoch":
            rows.append(dict(value))
    return rows


def _indexed_series(
    curve: (
        Sequence[Mapping[str, Any]]
        | Mapping[str, Sequence[Any]]
    ),
    metric: str,
    *,
    owner: str,
) -> list[tuple[int, float]]:
    if isinstance(curve, Mapping):
        if "epoch" not in curve:
            raise KeyError(f"{owner} curve is missing epoch")
        if metric not in curve:
            raise KeyError(
                f"{owner} curve is missing metric {metric!r}"
            )
        epochs = list(curve["epoch"])
        values = list(curve[metric])
        if len(epochs) != len(values):
            raise ValueError(
                f"{owner} epoch and metric series must have equal length"
            )
        rows = zip(epochs, values)
    else:
        rows = []
        for position, row in enumerate(curve):
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"{owner} curve row {position} must be a mapping"
                )
            if "epoch" not in row:
                raise KeyError(
                    f"{owner} curve row {position} is missing epoch"
                )
            if metric not in row:
                raise KeyError(
                    f"{owner} curve row {position} is missing "
                    f"metric {metric!r}"
                )
            rows.append((row["epoch"], row[metric]))

    indexed: list[tuple[int, float]] = []
    seen: set[int] = set()
    for epoch_value, metric_value in rows:
        if isinstance(epoch_value, bool) or not isinstance(
            epoch_value,
            Integral,
        ):
            raise ValueError(
                f"{owner} epochs must be finite integers"
            )
        epoch = int(epoch_value)
        if epoch in seen:
            raise ValueError(
                f"{owner} curve contains duplicate epoch {epoch}"
            )
        seen.add(epoch)
        try:
            value = float(metric_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{owner} curve metric {metric!r} must be numeric"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"{owner} curve metric {metric!r} must be finite"
            )
        indexed.append((epoch, value))
    if not indexed:
        raise ValueError(f"{owner} curve must not be empty")
    return sorted(indexed)


def compare_curves(
    reproduced: Sequence[Mapping[str, Any]],
    paper_curve: (
        Sequence[Mapping[str, Any]]
        | Mapping[str, Sequence[Any]]
    ),
    *,
    metric: str = "validation_accuracy",
) -> dict[str, Any]:
    """Align by one-based epoch and compare overlap without interpolation."""

    ours = dict(_indexed_series(
        reproduced,
        metric,
        owner="reproduced",
    ))
    paper = dict(_indexed_series(
        paper_curve,
        metric,
        owner="reference",
    ))
    epochs = sorted(set(ours).intersection(paper))
    if not epochs:
        raise ValueError(
            "reproduced and reference curves have no overlapping epochs"
        )
    differences = [
        ours[epoch] - paper[epoch] for epoch in epochs
    ]
    absolute = [abs(value) for value in differences]
    final_epoch = epochs[-1]
    return {
        "metric": metric,
        "reproduced_epochs": len(ours),
        "paper_epochs": len(paper),
        "overlap_epochs": len(epochs),
        "epochs": epochs,
        "final_epoch": final_epoch,
        "final_reproduced": ours[final_epoch],
        "final_paper": paper[final_epoch],
        "best_reproduced": max(ours[epoch] for epoch in epochs),
        "best_paper": max(paper[epoch] for epoch in epochs),
        "mean_absolute_error": sum(absolute) / len(epochs),
        "max_absolute_error": max(absolute),
        "differences": differences,
    }


def _svg_line(
    values: Sequence[tuple[int, float]],
    *,
    color: str,
    series: str,
    epoch_min: int,
    epoch_max: int,
    value_min: float,
    value_max: float,
    width: float = 760.0,
) -> str:
    epoch_span = epoch_max - epoch_min
    value_span = value_max - value_min
    points = " ".join(
        f"{450.0 if epoch_span == 0 else 80 + width * (epoch - epoch_min) / epoch_span:.2f},"
        f"{190.0 if value_span == 0.0 else 300 - 220 * (value - value_min) / value_span:.2f}"
        for epoch, value in values
    )
    return (
        f'<polyline data-series="{escape(series)}" fill="none" '
        f'stroke="{color}" stroke-width="2.5" points="{points}"/>'
    )


def write_curve_comparison(
    reproduced: Sequence[Mapping[str, Any]],
    paper_curve: (
        Sequence[Mapping[str, Any]]
        | Mapping[str, Sequence[Any]]
    ),
    output_dir: str | Path,
    *,
    metric: str = "validation_accuracy",
) -> dict[str, Path]:
    """Write an overlay SVG, differences CSV, and JSON summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ours = _indexed_series(
        reproduced,
        metric,
        owner="reproduced",
    )
    paper = _indexed_series(
        paper_curve,
        metric,
        owner="reference",
    )
    summary = compare_curves(
        reproduced,
        paper_curve,
        metric=metric,
    )
    all_points = ours + paper
    epoch_min = min(epoch for epoch, _ in all_points)
    epoch_max = max(epoch for epoch, _ in all_points)
    value_min = min(value for _, value in all_points)
    value_max = max(value for _, value in all_points)
    overlay = output / "curve_comparison.svg"
    overlay.write_text(
        "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'width="900" height="360">',
            '<rect width="900" height="360" fill="white"/>',
            f'<text x="450" y="30" text-anchor="middle">'
            f"{escape(metric)} comparison</text>",
            _svg_line(
                ours,
                color="#2563eb",
                series="reproduction",
                epoch_min=epoch_min,
                epoch_max=epoch_max,
                value_min=value_min,
                value_max=value_max,
            ),
            _svg_line(
                paper,
                color="#dc2626",
                series="reference",
                epoch_min=epoch_min,
                epoch_max=epoch_max,
                value_min=value_min,
                value_max=value_max,
            ),
            '<text x="80" y="345" fill="#2563eb">'
            "reproduction</text>",
            '<text x="210" y="345" fill="#dc2626">'
            "reference</text>",
            "</svg>",
        ]),
        encoding="utf-8",
    )
    difference = output / "curve_difference.csv"
    ours_by_epoch = dict(ours)
    paper_by_epoch = dict(paper)
    with difference.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch",
            "reproduced",
            "reference",
            "difference",
        ])
        for epoch in summary["epochs"]:
            writer.writerow([
                epoch,
                ours_by_epoch[epoch],
                paper_by_epoch[epoch],
                ours_by_epoch[epoch] - paper_by_epoch[epoch],
            ])
    summary_path = output / "curve_comparison.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "overlay": overlay,
        "difference": difference,
        "summary": summary_path,
    }


__all__ = [
    "compare_curves",
    "load_metrics_jsonl",
    "write_curve_comparison",
]
