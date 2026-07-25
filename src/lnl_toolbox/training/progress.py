from __future__ import annotations

"""Dependency-free terminal progress and SVG curves for long experiments."""

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import sys
from typing import TextIO


class TerminalTrainingProgress:
    """Render bounded batch progress without changing experiment semantics."""

    def __init__(
        self,
        *,
        epoch: int,
        total_epochs: int,
        total_batches: int,
        update_interval: int = 20,
        enabled: bool = True,
        stream: TextIO | None = None,
        force: bool = False,
        width: int = 24,
    ) -> None:
        if epoch <= 0 or total_epochs <= 0 or epoch > total_epochs:
            raise ValueError("epoch must be within [1, total_epochs]")
        if total_batches <= 0:
            raise ValueError("total_batches must be positive")
        if update_interval <= 0 or width <= 0:
            raise ValueError("update_interval and width must be positive")
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.update_interval = update_interval
        self.enabled = bool(enabled)
        self.stream = stream or sys.stderr
        self.force = bool(force)
        self.width = width

    def update(self, batch: int, *, loss: float, accuracy: float) -> None:
        if not 1 <= batch <= self.total_batches:
            raise ValueError("batch must be within [1, total_batches]")
        if not math.isfinite(loss) or not math.isfinite(accuracy):
            raise ValueError("progress loss and accuracy must be finite")
        interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        if not self.enabled or (not interactive and not self.force):
            return
        if batch != self.total_batches and batch % self.update_interval:
            return
        completed = int(self.width * batch / self.total_batches)
        bar = "#" * completed + "-" * (self.width - completed)
        message = (
            f"Epoch {self.epoch:03d}/{self.total_epochs:03d} "
            f"[{bar}] {batch:04d}/{self.total_batches:04d} "
            f"loss={loss:.4f} acc={accuracy * 100.0:5.1f}%"
        )
        ending = "\n" if batch == self.total_batches else "\r"
        self.stream.write(message + ending)
        self.stream.flush()


def _finite_series(rows: Sequence[Mapping[str, object]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = float(row[name])
        if not math.isfinite(value):
            raise ValueError(f"Curve metric {name!r} must be finite")
        values.append(value)
    return values


def _points(
    values: Sequence[float],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    minimum: float,
    maximum: float,
) -> str:
    span = maximum - minimum
    if span <= 0.0:
        span = 1.0
    denominator = max(1, len(values) - 1)
    return " ".join(
        f"{left + width * index / denominator:.2f},"
        f"{top + height * (maximum - value) / span:.2f}"
        for index, value in enumerate(values)
    )


def write_training_curves_svg(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
) -> Path:
    """Write loss, accuracy and learning-rate curves from epoch metric rows."""

    if not rows:
        raise ValueError("At least one epoch row is required")
    required = (
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
        "learning_rate",
    )
    series = {name: _finite_series(rows, name) for name in required}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    canvas_width = 960
    canvas_height = 720
    left = 80.0
    plot_width = 820.0
    panel_height = 150.0
    panel_tops = (80.0, 300.0, 520.0)
    colors = {"train": "#2563eb", "validation": "#dc2626", "lr": "#059669"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="720" '
        'viewBox="0 0 960 720">',
        '<rect width="960" height="720" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}'
        '.axis{stroke:#64748b;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}'
        '.line{fill:none;stroke-width:2.5;stroke-linejoin:round}</style>',
        '<text x="480" y="35" text-anchor="middle" font-size="22">'
        "Training progress</text>",
    ]

    panels = (
        ("Loss", series["train_loss"], series["validation_loss"]),
        ("Accuracy", series["train_accuracy"], series["validation_accuracy"]),
        ("Learning rate", series["learning_rate"], None),
    )
    for panel_index, (title, first, second) in enumerate(panels):
        top = panel_tops[panel_index]
        combined = first if second is None else first + second
        minimum = min(combined)
        maximum = max(combined)
        if title == "Accuracy":
            minimum, maximum = 0.0, 1.0
        padding = (maximum - minimum) * 0.08
        if padding == 0.0:
            padding = max(abs(maximum) * 0.08, 1e-6)
        plot_min = minimum if title == "Accuracy" else minimum - padding
        plot_max = maximum if title == "Accuracy" else maximum + padding
        parts.extend((
            f'<text x="20" y="{top + 18:.0f}" font-size="15">{title}</text>',
            f'<line class="axis" x1="{left}" y1="{top + panel_height}" '
            f'x2="{left + plot_width}" y2="{top + panel_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{top + panel_height}"/>',
            f'<line class="grid" x1="{left}" y1="{top + panel_height / 2}" '
            f'x2="{left + plot_width}" y2="{top + panel_height / 2}"/>',
            f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" '
            f'font-size="11">{plot_max:.4g}</text>',
            f'<text x="{left - 8}" y="{top + panel_height + 4}" '
            f'text-anchor="end" font-size="11">{plot_min:.4g}</text>',
            f'<polyline class="line" stroke="{colors["train"] if second is not None else colors["lr"]}" '
            f'points="{_points(first, left=left, top=top, width=plot_width, height=panel_height, minimum=plot_min, maximum=plot_max)}"/>',
        ))
        if second is not None:
            parts.append(
                f'<polyline class="line" stroke="{colors["validation"]}" '
                f'points="{_points(second, left=left, top=top, width=plot_width, height=panel_height, minimum=plot_min, maximum=plot_max)}"/>'
            )

    parts.extend((
        '<line x1="690" y1="35" x2="720" y2="35" '
        f'stroke="{colors["train"]}" stroke-width="3"/>',
        '<text x="728" y="39" font-size="12">train</text>',
        '<line x1="785" y1="35" x2="815" y2="35" '
        f'stroke="{colors["validation"]}" stroke-width="3"/>',
        '<text x="823" y="39" font-size="12">validation</text>',
        f'<text x="900" y="705" text-anchor="end" font-size="11">'
        f"epochs: {len(rows)}</text>",
        "</svg>",
    ))
    target.write_text("\n".join(parts), encoding="utf-8")
    return target
