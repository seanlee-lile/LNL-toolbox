from __future__ import annotations

"""Reusable preprocessing for heterogeneous binary benchmark files.

The preprocessor deliberately keeps fitting separate from transformation so a
paper split can fit statistics on the training fold only and reuse the exact
state for validation/test data.
"""

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_MISSING = {"", "?", "na", "nan", "null", "none"}
_LIBSVM_SUFFIXES = {".svm", ".libsvm"}


def _is_missing(value: str) -> bool:
    return str(value).strip().lower() in _MISSING


def _as_float(value: str) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _stable_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values})


@dataclass(frozen=True)
class BinaryPreprocessingConfig:
    """Configuration for :class:`BinaryPreprocessor`.

    ``standardize`` is disabled by default so preprocessing never silently
    changes a paper's feature convention.
    """

    file_format: str = "auto"
    delimiter: str | None = None
    target_column: int | str = -1
    has_header: bool = False
    missing_policy: str = "median_mode"
    categorical_policy: str = "one_hot"
    standardize: bool = False
    label_values: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        file_format = str(self.file_format).strip().lower()
        if file_format not in {"auto", "delimited", "csv", "whitespace", "libsvm"}:
            raise ValueError("file_format must be auto, delimited, csv, whitespace, or libsvm")
        missing_policy = str(self.missing_policy).strip().lower()
        if missing_policy not in {"error", "median_mode"}:
            raise ValueError("missing_policy must be error or median_mode")
        categorical_policy = str(self.categorical_policy).strip().lower()
        if categorical_policy not in {"error", "one_hot"}:
            raise ValueError("categorical_policy must be error or one_hot")
        if self.label_values is not None and len(self.label_values) != 2:
            raise ValueError("label_values must contain exactly two labels")
        object.__setattr__(self, "file_format", file_format)
        object.__setattr__(self, "missing_policy", missing_policy)
        object.__setattr__(self, "categorical_policy", categorical_policy)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "BinaryPreprocessingConfig":
        values = dict(config or {})
        if "format" in values and "file_format" not in values:
            values["file_format"] = values.pop("format")
        labels = values.get("label_values")
        if labels is not None:
            values["label_values"] = tuple(str(value) for value in labels)
        return cls(**values)


@dataclass
class BinaryPreprocessor:
    """Fit deterministic feature/label transformations on a training split."""

    config: BinaryPreprocessingConfig = field(default_factory=BinaryPreprocessingConfig)
    feature_names: list[str] = field(default_factory=list, init=False)
    label_mapping: dict[str, int] = field(default_factory=dict, init=False)
    column_specs: list[dict[str, Any]] = field(default_factory=list, init=False)
    fitted: bool = field(default=False, init=False)
    source_fingerprint: str = field(default="", init=False)

    def _resolve_format(self, path: Path, lines: Sequence[str]) -> str:
        configured = self.config.file_format
        if configured != "auto":
            return "delimited" if configured == "csv" else configured
        if path.suffix.lower() in _LIBSVM_SUFFIXES:
            return "libsvm"
        for line in lines:
            if ":" in line and len(line.split()) > 1:
                return "libsvm"
        if self.config.delimiter is None and any(re.search(r"\s+", line.strip()) for line in lines):
            return "whitespace"
        return "delimited"

    def _read_rows(self, path: str | Path) -> tuple[list[list[str]], str]:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line and not line.startswith("#")]
        if not lines:
            raise ValueError("binary benchmark file contains no data rows")
        file_format = self._resolve_format(source, lines)
        if file_format == "libsvm":
            return self._read_libsvm(lines), file_format
        if file_format == "whitespace":
            return [line.split() for line in lines], file_format
        delimiter = self.config.delimiter
        if delimiter is None:
            sample = "\n".join(lines[: min(8, len(lines))])
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
        return [row for row in csv.reader(lines, delimiter=delimiter)], "delimited"

    @staticmethod
    def _read_libsvm(lines: Sequence[str]) -> list[list[str]]:
        parsed: list[tuple[str, dict[int, str]]] = []
        width = 0
        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            values: dict[int, str] = {}
            for token in tokens[1:]:
                if ":" not in token:
                    raise ValueError(f"invalid LibSVM token: {token!r}")
                index_text, value = token.split(":", 1)
                index = int(index_text)
                if index <= 0:
                    raise ValueError("LibSVM feature indices must be positive")
                values[index - 1] = value
                width = max(width, index)
            parsed.append((tokens[0], values))
        return [[label] + [values.get(index, "0") for index in range(width)] for label, values in parsed]

    @staticmethod
    def _drop_header(rows: list[list[str]], has_header: bool) -> tuple[list[list[str]], list[str]]:
        if not rows:
            raise ValueError("binary benchmark file contains no rows")
        if has_header:
            header = [str(value).strip() or f"feature_{index}" for index, value in enumerate(rows[0])]
            rows = rows[1:]
        else:
            header = [f"column_{index}" for index in range(len(rows[0]))]
        if not rows:
            raise ValueError("binary benchmark file contains no data rows after header")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("all binary benchmark rows must have the same number of columns")
        return rows, header

    def _resolve_target_column(self, header: Sequence[str]) -> int:
        target = self.config.target_column
        if isinstance(target, str):
            if target not in header:
                raise KeyError(f"target column {target!r} is not present")
            return header.index(target)
        column = int(target)
        if column < 0:
            column += len(header)
        if not 0 <= column < len(header):
            raise IndexError("target_column is out of range")
        return column

    def _fit_labels(self, values: Sequence[str]) -> None:
        observed = _stable_unique(value.strip() for value in values)
        configured = None if self.config.label_values is None else list(self.config.label_values)
        labels = configured if configured is not None else observed
        if len(labels) != 2 or set(labels) != set(observed):
            raise ValueError("binary targets must contain exactly two distinct labels")
        self.label_mapping = {str(value): index for index, value in enumerate(labels)}

    def _fit_columns(self, rows: Sequence[Sequence[str]], feature_columns: Sequence[int], header: Sequence[str]) -> None:
        self.column_specs = []
        self.feature_names = []
        for column in feature_columns:
            values = [str(row[column]).strip() for row in rows]
            numeric = all(_as_float(value) is not None for value in values if not _is_missing(value))
            name = str(header[column])
            if not numeric and self.config.categorical_policy == "error":
                raise ValueError(f"categorical feature {name!r} is not allowed")
            if numeric:
                parsed = [_as_float(value) for value in values]
                observed = np.asarray([value for value in parsed if value is not None], dtype=np.float64)
                if observed.size == 0:
                    if self.config.missing_policy == "error":
                        raise ValueError(f"numeric feature {name!r} is entirely missing")
                    fill = 0.0
                else:
                    fill = float(np.median(observed))
                mean = float(np.mean(np.asarray([value if value is not None else fill for value in parsed], dtype=np.float64)))
                scale = float(np.std(np.asarray([value if value is not None else fill for value in parsed], dtype=np.float64)))
                self.column_specs.append({"index": column, "name": name, "kind": "numeric", "fill": fill, "mean": mean, "scale": scale})
                self.feature_names.append(name)
            else:
                observed = [value for value in values if not _is_missing(value)]
                if not observed:
                    if self.config.missing_policy == "error":
                        raise ValueError(f"categorical feature {name!r} is entirely missing")
                    categories = ["__missing__"]
                    fill = "__missing__"
                else:
                    categories = _stable_unique(observed)
                    counts = {category: observed.count(category) for category in categories}
                    fill = min(categories, key=lambda category: (-counts[category], category))
                self.column_specs.append({"index": column, "name": name, "kind": "categorical", "fill": fill, "categories": categories})
                self.feature_names.extend(f"{name}={category}" for category in categories)

    def _transform_rows(self, rows: Sequence[Sequence[str]], target_column: int) -> tuple[np.ndarray, np.ndarray]:
        features: list[list[float]] = []
        targets: list[int] = []
        for row in rows:
            target = str(row[target_column]).strip()
            if target not in self.label_mapping:
                raise ValueError(f"unknown binary target {target!r}")
            targets.append(self.label_mapping[target])
            encoded: list[float] = []
            for spec in self.column_specs:
                value = str(row[int(spec["index"])]).strip()
                if _is_missing(value):
                    value = str(spec["fill"])
                if spec["kind"] == "numeric":
                    number = float(value)
                    if self.config.standardize:
                        scale = float(spec["scale"])
                        number = (number - float(spec["mean"])) / (scale if scale > 0.0 else 1.0)
                    encoded.append(number)
                else:
                    categories = list(spec["categories"])
                    if value not in categories:
                        raise ValueError(f"unknown category {value!r} in feature {spec['name']!r}")
                    encoded.extend(1.0 if value == category else 0.0 for category in categories)
            features.append(encoded)
        result = np.asarray(features, dtype=np.float32)
        labels = np.asarray(targets, dtype=np.int64)
        if not np.isfinite(result).all():
            raise ValueError("preprocessed binary features must be finite")
        return result, labels

    def fit(self, path: str | Path) -> "BinaryPreprocessor":
        rows, file_format = self._read_rows(path)
        rows, header = self._drop_header(rows, self.config.has_header)
        target_column = 0 if file_format == "libsvm" and self.config.target_column == -1 else self._resolve_target_column(header)
        self._fit_labels([row[target_column] for row in rows])
        self._fit_columns(rows, [index for index in range(len(header)) if index != target_column], header)
        self.fitted = True
        self.source_fingerprint = self._fingerprint(rows, header)
        return self

    def transform(self, path: str | Path, *, dataset: str | None = None, split: str = "train"):
        if not self.fitted:
            raise RuntimeError("BinaryPreprocessor must be fitted before transform")
        rows, file_format = self._read_rows(path)
        rows, header = self._drop_header(rows, self.config.has_header)
        target_column = 0 if file_format == "libsvm" and self.config.target_column == -1 else self._resolve_target_column(header)
        features, targets = self._transform_rows(rows, target_column)
        from .binary_benchmarks import BinaryBenchmark
        return BinaryBenchmark(features, targets, dataset or Path(path).stem, split)

    def fit_transform(self, path: str | Path, *, dataset: str | None = None, split: str = "train"):
        self.fit(path)
        return self.transform(path, dataset=dataset, split=split)

    @staticmethod
    def _fingerprint(rows: Sequence[Sequence[str]], header: Sequence[str]) -> str:
        payload = json.dumps({"header": list(header), "rows": [list(row) for row in rows]}, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted BinaryPreprocessor")
        return {
            "version": 1,
            "config": asdict(self.config),
            "feature_names": list(self.feature_names),
            "label_mapping": dict(self.label_mapping),
            "column_specs": list(self.column_specs),
            "source_fingerprint": self.source_fingerprint,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BinaryPreprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported BinaryPreprocessor state version")
        config = BinaryPreprocessingConfig.from_mapping(payload.get("config"))
        result = cls(config)
        result.feature_names = list(payload["feature_names"])
        result.label_mapping = {str(key): int(value) for key, value in payload["label_mapping"].items()}
        result.column_specs = list(payload["column_specs"])
        result.source_fingerprint = str(payload.get("source_fingerprint", ""))
        result.fitted = True
        return result


__all__ = ["BinaryPreprocessingConfig", "BinaryPreprocessor"]
