from __future__ import annotations

"""Reusable preprocessing for heterogeneous binary benchmark files.

The preprocessor deliberately keeps fitting separate from transformation so a
paper split can fit statistics on the training fold only and reuse the exact
state for validation/test data.
"""

import copy
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
        if file_format not in {
            "auto",
            "delimited",
            "csv",
            "whitespace",
            "libsvm",
        }:
            raise ValueError(
                "file_format must be auto, delimited, csv, whitespace, "
                "or libsvm"
            )
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
    def from_mapping(
        cls,
        config: Mapping[str, Any] | None,
    ) -> "BinaryPreprocessingConfig":
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

    config: BinaryPreprocessingConfig = field(
        default_factory=BinaryPreprocessingConfig
    )
    feature_names: list[str] = field(default_factory=list, init=False)
    label_mapping: dict[str, int] = field(default_factory=dict, init=False)
    column_specs: list[dict[str, Any]] = field(default_factory=list, init=False)
    fitted: bool = field(default=False, init=False)
    source_fingerprint: str = field(default="", init=False)
    parser_format: str = field(default="", init=False)
    input_width: int = field(default=0, init=False)
    input_header: list[str] = field(default_factory=list, init=False)
    target_column_index: int = field(default=-1, init=False)
    feature_columns: list[int] = field(default_factory=list, init=False)

    def _resolve_format(self, path: Path, lines: Sequence[str]) -> str:
        configured = self.config.file_format
        if configured != "auto":
            return "delimited" if configured == "csv" else configured
        if path.suffix.lower() in _LIBSVM_SUFFIXES:
            return "libsvm"
        for line in lines:
            if ":" in line and len(line.split()) > 1:
                return "libsvm"
        if self.config.delimiter is None and any(
            re.search(r"\s+", line.strip()) for line in lines
        ):
            return "whitespace"
        return "delimited"

    def _read_rows(
        self,
        path: str | Path,
        *,
        expected_libsvm_width: int | None = None,
    ) -> tuple[list[list[str]], str]:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        lines = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
        ]
        lines = [line for line in lines if line and not line.startswith("#")]
        if not lines:
            raise ValueError("binary benchmark file contains no data rows")
        file_format = self._resolve_format(source, lines)
        if file_format == "libsvm":
            return self._read_libsvm(
                lines,
                expected_width=expected_libsvm_width,
            ), file_format
        if file_format == "whitespace":
            return [line.split() for line in lines], file_format
        delimiter = self.config.delimiter
        if delimiter is None:
            sample = "\n".join(lines[: min(8, len(lines))])
            try:
                delimiter = csv.Sniffer().sniff(
                    sample,
                    delimiters=",;\t|",
                ).delimiter
            except csv.Error:
                delimiter = ","
        return [
            row for row in csv.reader(lines, delimiter=delimiter)
        ], "delimited"

    @staticmethod
    def _read_libsvm(
        lines: Sequence[str],
        *,
        expected_width: int | None = None,
    ) -> list[list[str]]:
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
                    raise ValueError(
                        "LibSVM feature indices must be positive"
                    )
                if index - 1 in values:
                    raise ValueError(
                        "LibSVM rows must not contain duplicate feature indices"
                    )
                values[index - 1] = value
                width = max(width, index)
            parsed.append((tokens[0], values))
        if expected_width is not None and width != expected_width:
            raise ValueError(
                "LibSVM observed feature width does not match the fitted "
                f"schema: expected {expected_width}, got {width}"
            )
        return [
            [label] + [
                values.get(index, "0") for index in range(width)
            ]
            for label, values in parsed
        ]

    @staticmethod
    def _drop_header(
        rows: list[list[str]],
        has_header: bool,
    ) -> tuple[list[list[str]], list[str]]:
        if not rows:
            raise ValueError("binary benchmark file contains no rows")
        if has_header:
            header = [
                str(value).strip() or f"feature_{index}"
                for index, value in enumerate(rows[0])
            ]
            rows = rows[1:]
        else:
            header = [
                f"column_{index}" for index in range(len(rows[0]))
            ]
        if not rows:
            raise ValueError(
                "binary benchmark file contains no data rows after header"
            )
        width = len(rows[0])
        if len(header) != width:
            raise ValueError("header width must match the data width")
        if len(set(header)) != len(header):
            raise ValueError("binary benchmark column names must be unique")
        if any(len(row) != width for row in rows):
            raise ValueError(
                "all binary benchmark rows must have the same number of columns"
            )
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
        configured = (
            None
            if self.config.label_values is None
            else list(self.config.label_values)
        )
        labels = configured if configured is not None else observed
        if len(labels) != 2 or set(labels) != set(observed):
            raise ValueError(
                "binary targets must contain exactly two distinct labels"
            )
        self.label_mapping = {
            str(value): index for index, value in enumerate(labels)
        }

    def read_targets(self, path: str | Path) -> np.ndarray:
        """Read only target values for split construction, without fitting features."""

        rows, file_format = self._read_rows(path)
        rows, header = self._drop_header(rows, self.config.has_header)
        target_column = (
            0
            if file_format == "libsvm" and self.config.target_column == -1
            else self._resolve_target_column(header)
        )
        observed = [str(row[target_column]).strip() for row in rows]
        labels = (
            _stable_unique(observed)
            if self.config.label_values is None
            else list(self.config.label_values)
        )
        if len(labels) != 2 or set(labels) != set(observed):
            raise ValueError("binary targets must contain exactly the configured labels")
        mapping = {str(label): index for index, label in enumerate(labels)}
        return np.asarray([mapping[value] for value in observed], dtype=np.int64)

    def _fit_columns(
        self,
        rows: Sequence[Sequence[str]],
        feature_columns: Sequence[int],
        header: Sequence[str],
    ) -> None:
        self.column_specs = []
        self.feature_names = []
        for column in feature_columns:
            values = [str(row[column]).strip() for row in rows]
            if (
                self.config.missing_policy == "error"
                and any(_is_missing(value) for value in values)
            ):
                raise ValueError(
                    f"missing value in feature {str(header[column])!r}"
                )
            numeric = all(
                _as_float(value) is not None
                for value in values
                if not _is_missing(value)
            )
            name = str(header[column])
            if not numeric and self.config.categorical_policy == "error":
                raise ValueError(
                    f"categorical feature {name!r} is not allowed"
                )
            if numeric:
                parsed = [_as_float(value) for value in values]
                observed = np.asarray(
                    [value for value in parsed if value is not None],
                    dtype=np.float64,
                )
                fill = 0.0 if observed.size == 0 else float(
                    np.median(observed)
                )
                filled = np.asarray(
                    [
                        value if value is not None else fill
                        for value in parsed
                    ],
                    dtype=np.float64,
                )
                scale = float(np.std(filled))
                self.column_specs.append({
                    "index": column,
                    "name": name,
                    "kind": "numeric",
                    "fill": fill,
                    "mean": float(np.mean(filled)),
                    "scale": scale if scale > 0.0 else 1.0,
                })
                self.feature_names.append(name)
            else:
                observed = [
                    value for value in values if not _is_missing(value)
                ]
                if not observed:
                    categories = ["__missing__"]
                    fill = "__missing__"
                else:
                    categories = _stable_unique(observed)
                    counts = {
                        category: observed.count(category)
                        for category in categories
                    }
                    fill = min(
                        categories,
                        key=lambda category: (
                            -counts[category],
                            category,
                        ),
                    )
                self.column_specs.append({
                    "index": column,
                    "name": name,
                    "kind": "categorical",
                    "fill": fill,
                    "categories": categories,
                })
                self.feature_names.extend(
                    f"{name}={category}" for category in categories
                )

    def _transform_rows(
        self,
        rows: Sequence[Sequence[str]],
        target_column: int,
    ) -> tuple[np.ndarray, np.ndarray]:
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
                    if self.config.missing_policy == "error":
                        raise ValueError(
                            f"missing value in feature {spec['name']!r}"
                        )
                    value = str(spec["fill"])
                if spec["kind"] == "numeric":
                    try:
                        number = float(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"non-numeric value {value!r} in numeric "
                            f"feature {spec['name']!r}"
                        ) from exc
                    if self.config.standardize:
                        scale = float(spec["scale"])
                        number = (
                            number - float(spec["mean"])
                        ) / (scale if scale > 0.0 else 1.0)
                    encoded.append(number)
                else:
                    categories = list(spec["categories"])
                    if value not in categories:
                        raise ValueError(
                            f"unknown category {value!r} in feature "
                            f"{spec['name']!r}"
                        )
                    encoded.extend(
                        1.0 if value == category else 0.0
                        for category in categories
                    )
            features.append(encoded)
        result = np.asarray(features, dtype=np.float32)
        labels = np.asarray(targets, dtype=np.int64)
        if not np.isfinite(result).all():
            raise ValueError(
                "preprocessed binary features must be finite"
            )
        return result, labels

    @staticmethod
    def _select_rows(
        rows: Sequence[Sequence[str]],
        row_indices: Sequence[int] | np.ndarray | None,
        *,
        owner: str,
    ) -> list[list[str]]:
        materialized = [list(row) for row in rows]
        if row_indices is None:
            return materialized
        indices = np.asarray(row_indices)
        if (
            indices.ndim != 1
            or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
        ):
            raise ValueError(f"{owner} row indices must be non-empty integers")
        indices = indices.astype(np.int64, copy=False)
        if (
            indices.min() < 0
            or indices.max() >= len(materialized)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError(
                f"{owner} row indices must be unique and within the source file"
            )
        return [materialized[int(index)] for index in indices]

    def fit(
        self,
        path: str | Path,
        *,
        row_indices: Sequence[int] | np.ndarray | None = None,
    ) -> "BinaryPreprocessor":
        rows, file_format = self._read_rows(path)
        rows, header = self._drop_header(
            rows,
            self.config.has_header,
        )
        source_rows = rows
        rows = self._select_rows(rows, row_indices, owner="fit")
        target_column = (
            0
            if file_format == "libsvm"
            and self.config.target_column == -1
            else self._resolve_target_column(header)
        )
        self._fit_labels([row[target_column] for row in rows])
        feature_columns = [
            index
            for index in range(len(header))
            if index != target_column
        ]
        self._fit_columns(
            rows,
            feature_columns,
            header,
        )
        self.parser_format = file_format
        self.input_width = len(header)
        self.input_header = list(header)
        self.target_column_index = target_column
        self.feature_columns = feature_columns
        self.fitted = True
        self.source_fingerprint = self._fingerprint(source_rows, header)
        return self

    def transform(
        self,
        path: str | Path,
        *,
        dataset: str | None = None,
        split: str = "train",
        row_indices: Sequence[int] | np.ndarray | None = None,
    ):
        if not self.fitted:
            raise RuntimeError(
                "BinaryPreprocessor must be fitted before transform"
            )
        expected_libsvm_width = (
            self.input_width - 1
            if self.parser_format == "libsvm"
            else None
        )
        rows, file_format = self._read_rows(
            path,
            expected_libsvm_width=expected_libsvm_width,
        )
        rows, header = self._drop_header(
            rows,
            self.config.has_header,
        )
        if file_format != self.parser_format:
            raise ValueError(
                "input parser format does not match the fitted schema"
            )
        if len(header) != self.input_width:
            raise ValueError(
                "input feature width does not match the fitted schema"
            )
        if list(header) != self.input_header:
            raise ValueError(
                "input columns and order do not match the fitted schema"
            )
        target_column = (
            0
            if file_format == "libsvm"
            and self.config.target_column == -1
            else self._resolve_target_column(header)
        )
        if target_column != self.target_column_index:
            raise ValueError(
                "target column does not match the fitted schema"
            )
        rows = self._select_rows(rows, row_indices, owner=split)
        features, targets = self._transform_rows(rows, target_column)
        from .binary_benchmarks import BinaryBenchmark

        return BinaryBenchmark(
            features,
            targets,
            dataset or Path(path).stem,
            split,
            None if row_indices is None else np.asarray(row_indices, dtype=np.int64),
        )

    def fit_transform(
        self,
        path: str | Path,
        *,
        dataset: str | None = None,
        split: str = "train",
    ):
        self.fit(path)
        return self.transform(path, dataset=dataset, split=split)

    @staticmethod
    def _fingerprint(
        rows: Sequence[Sequence[str]],
        header: Sequence[str],
    ) -> str:
        payload = json.dumps(
            {
                "header": list(header),
                "rows": [list(row) for row in rows],
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError(
                "cannot serialize an unfitted BinaryPreprocessor"
            )
        return copy.deepcopy({
            "version": 1,
            "config": asdict(self.config),
            "feature_names": list(self.feature_names),
            "label_mapping": dict(self.label_mapping),
            "column_specs": self.column_specs,
            "source_fingerprint": self.source_fingerprint,
            "parser_format": self.parser_format,
            "input_width": self.input_width,
            "input_header": list(self.input_header),
            "target_column_index": self.target_column_index,
            "feature_columns": list(self.feature_columns),
        })

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.state_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load_state_dict(
        self,
        payload: Mapping[str, Any],
    ) -> "BinaryPreprocessor":
        if not isinstance(payload, Mapping):
            raise TypeError("preprocessing state must be a mapping")
        if payload.get("version") != 1:
            raise ValueError(
                "unsupported BinaryPreprocessor state version"
            )
        required = {
            "config",
            "feature_names",
            "label_mapping",
            "column_specs",
            "source_fingerprint",
            "parser_format",
            "input_width",
            "input_header",
            "target_column_index",
            "feature_columns",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(
                "preprocessing state is missing fields: "
                + ", ".join(missing)
            )

        config_payload = payload["config"]
        if not isinstance(config_payload, Mapping):
            raise TypeError("preprocessing config state must be a mapping")
        config = BinaryPreprocessingConfig.from_mapping(config_payload)

        parser_format = payload["parser_format"]
        if parser_format not in {"delimited", "whitespace", "libsvm"}:
            raise ValueError("invalid fitted parser format")
        input_width = payload["input_width"]
        if (
            isinstance(input_width, bool)
            or not isinstance(input_width, int)
            or input_width < 2
        ):
            raise ValueError("input_width must be an integer of at least 2")

        input_header = payload["input_header"]
        if (
            not isinstance(input_header, list)
            or len(input_header) != input_width
            or any(
                not isinstance(name, str) or not name
                for name in input_header
            )
            or len(set(input_header)) != len(input_header)
        ):
            raise ValueError(
                "input_header must contain unique non-empty column names"
            )

        target_column = payload["target_column_index"]
        if (
            isinstance(target_column, bool)
            or not isinstance(target_column, int)
            or not 0 <= target_column < input_width
        ):
            raise ValueError("invalid target_column_index")
        expected_features = [
            index for index in range(input_width)
            if index != target_column
        ]
        feature_columns = payload["feature_columns"]
        if feature_columns != expected_features:
            raise ValueError(
                "feature_columns do not match the fitted input schema"
            )

        label_mapping = payload["label_mapping"]
        if (
            not isinstance(label_mapping, Mapping)
            or len(label_mapping) != 2
            or any(
                not isinstance(label, str) or not label
                for label in label_mapping
            )
            or set(label_mapping.values()) != {0, 1}
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in label_mapping.values()
            )
        ):
            raise ValueError(
                "label_mapping must map exactly two labels to 0 and 1"
            )

        raw_specs = payload["column_specs"]
        if (
            not isinstance(raw_specs, list)
            or len(raw_specs) != len(expected_features)
        ):
            raise ValueError(
                "column_specs must align with all fitted feature columns"
            )
        column_specs: list[dict[str, Any]] = []
        derived_names: list[str] = []
        for raw_spec, column in zip(raw_specs, expected_features):
            if not isinstance(raw_spec, Mapping):
                raise TypeError("each column spec must be a mapping")
            name = raw_spec.get("name")
            if (
                raw_spec.get("index") != column
                or name != input_header[column]
            ):
                raise ValueError(
                    "column spec does not match the fitted input schema"
                )
            kind = raw_spec.get("kind")
            if kind == "numeric":
                values: dict[str, float] = {}
                for field_name in ("fill", "mean", "scale"):
                    value = raw_spec.get(field_name)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not np.isfinite(value)
                    ):
                        raise ValueError(
                            f"numeric column {field_name} must be finite"
                        )
                    values[field_name] = float(value)
                if values["scale"] <= 0.0:
                    raise ValueError(
                        "numeric column scale must be greater than zero"
                    )
                column_specs.append({
                    "index": column,
                    "name": name,
                    "kind": "numeric",
                    **values,
                })
                derived_names.append(name)
            elif kind == "categorical":
                categories = raw_spec.get("categories")
                fill = raw_spec.get("fill")
                if (
                    not isinstance(categories, list)
                    or not categories
                    or any(
                        not isinstance(value, str) or not value
                        for value in categories
                    )
                    or len(set(categories)) != len(categories)
                    or fill not in categories
                ):
                    raise ValueError(
                        "categorical column state is invalid"
                    )
                column_specs.append({
                    "index": column,
                    "name": name,
                    "kind": "categorical",
                    "fill": fill,
                    "categories": list(categories),
                })
                derived_names.extend(
                    f"{name}={category}" for category in categories
                )
            else:
                raise ValueError(
                    "column spec kind must be numeric or categorical"
                )

        feature_names = payload["feature_names"]
        if feature_names != derived_names:
            raise ValueError(
                "feature_names do not match the fitted column specs"
            )
        fingerprint = payload["source_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        ):
            raise ValueError("source_fingerprint must be a SHA-256 hex digest")

        self.config = config
        self.feature_names = list(derived_names)
        self.label_mapping = dict(label_mapping)
        self.column_specs = copy.deepcopy(column_specs)
        self.source_fingerprint = fingerprint
        self.parser_format = parser_format
        self.input_width = input_width
        self.input_header = list(input_header)
        self.target_column_index = target_column
        self.feature_columns = list(feature_columns)
        self.fitted = True
        return self

    @classmethod
    def load(cls, path: str | Path) -> "BinaryPreprocessor":
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        return cls().load_state_dict(payload)


__all__ = ["BinaryPreprocessingConfig", "BinaryPreprocessor"]
