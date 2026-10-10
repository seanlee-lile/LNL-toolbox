from __future__ import annotations

"""User datasets: dataset.yaml plus a UTF-8 samples.csv manifest."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .contracts import DataSpec, RawDatasetSplit, UnsupportedDatasetSplitError
from .sources import UciBinaryAdapter


def standard_metadata(root: Path) -> dict:
    value = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("task") != "classification":
        raise ValueError("dataset.yaml must declare task: classification")
    if value.get("modality") not in {"image", "tabular"}:
        raise ValueError("standard datasets support image or tabular; text training is not supported")
    classes = value.get("classes")
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("classes must list at least two {id, name} entries")
    if any(not isinstance(c, dict) or type(c.get("id")) is not int or not str(c.get("name", "")).strip() for c in classes):
        raise ValueError("each class requires an integer id and a name")
    if [c["id"] for c in classes] != list(range(len(classes))):
        raise ValueError("class ids must be consecutive, starting at 0")
    if value.get("label_status", "unknown") not in {"clean", "noisy", "unknown"}:
        raise ValueError("label_status must be clean, noisy or unknown")
    rate = value.get("noise_rate")
    if rate is not None and (isinstance(rate, bool) or not np.isfinite(float(rate)) or not 0 <= float(rate) <= 1):
        raise ValueError("noise_rate must be null or a number in [0, 1]")
    if value.get("label_status") == "clean" and rate not in {None, 0}:
        raise ValueError("clean datasets cannot declare a positive noise_rate")
    if rate is not None and value.get("label_status", "unknown") == "unknown":
        raise ValueError("declare label_status before supplying noise_rate")
    return value


class StandardDatasetAdapter:
    name = "standard"
    aliases = ("manifest_dataset",)

    def validate(self, spec: DataSpec) -> None:
        if spec.root is None or not spec.root.is_dir():
            raise ValueError("standard dataset requires a directory in data.root")
        standard_metadata(spec.root)
        if not (spec.root / "samples.csv").is_file():
            raise FileNotFoundError("standard dataset requires samples.csv")

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        if split not in {"train", "test", "validation"}:
            raise UnsupportedDatasetSplitError(f"unsupported standard split: {split}")
        self.validate(spec)
        root = spec.root.resolve()
        metadata = standard_metadata(root)
        with (root / "samples.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            if not {"sample_id", "label"}.issubset(columns) or len(columns) != len(set(columns)):
                raise ValueError("samples.csv requires unique columns including sample_id and label")
            rows = list(reader)
        if not rows:
            raise ValueError("samples.csv must contain samples")
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError("every CSV row must have the same number of fields as the header")
        ids = [r["sample_id"].strip() for r in rows]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError("sample_id must be non-empty and unique")
        classes = len(metadata["classes"])
        try:
            labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
            clean_values = [r.get("clean_label", "").strip() for r in rows]
            if any(clean_values) and not all(clean_values):
                raise ValueError("clean_label must be provided for every sample or omitted")
            clean = np.asarray([int(v) for v in clean_values], dtype=np.int64) if all(clean_values) else None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid labels in samples.csv: {exc}") from exc
        for values in (labels, clean):
            if values is not None and ((values < 0).any() or (values >= classes).any()):
                raise ValueError("sample labels are outside dataset.yaml classes")
        if metadata.get("label_status") == "clean":
            if clean is not None and not np.array_equal(clean, labels):
                raise ValueError("clean_label conflicts with label_status: clean")
            clean = labels.copy()
        if metadata["modality"] == "image":
            if "path" not in columns:
                raise ValueError("image samples.csv requires path")
            inputs = []
            shape = None
            for row in rows:
                relative = Path(row["path"])
                path = (root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(root):
                    raise ValueError("image paths must stay inside the dataset directory")
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    current = (image.height, image.width, len(image.getbands()))
                if shape is not None and current != shape:
                    raise ValueError("images must have a consistent size and channel count")
                shape = current
                inputs.append(path)
        else:
            features = [c for c in columns if c.startswith("feature_")]
            if not features:
                raise ValueError("tabular samples.csv requires numeric feature_* columns")
            try:
                inputs = np.asarray([[float(r[c]) for c in features] for r in rows], dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("features must be numeric; encode categorical values and resolve missing values first") from exc
            if not np.isfinite(inputs).all():
                raise ValueError("features must be finite, without missing values")
        native = [r.get("split", "").strip() for r in rows]
        if any(native):
            if not all(v in {"train", "validation", "test"} for v in native) or "train" not in native:
                raise ValueError("split must be train, validation or test on every row, with a training set")
            if spec.options.get("split"):
                raise ValueError("remove CSV split assignments before requesting a new split")
            groups = {}
            for row, assignment in zip(rows, native):
                group = row.get("group_id", "").strip()
                if group and group in groups and groups[group] != assignment:
                    raise ValueError("a group_id must not cross dataset splits")
                if group:
                    groups[group] = assignment
            indices = np.flatnonzero(np.asarray(native) == split)
        elif spec.options.get("split"):
            if "group_id" in columns and any(r.get("group_id") for r in rows):
                raise ValueError("automatic group-aware splitting is not supported; supply explicit split assignments")
            indices = UciBinaryAdapter._split_rows(labels, dict(spec.options), seed)[split]
        else:
            # Registration does not invent a training/test partition.
            indices = np.arange(len(rows)) if split == "train" else np.empty(0, dtype=np.int64)
        selected = inputs[indices] if isinstance(inputs, np.ndarray) else [inputs[i] for i in indices]
        return RawDatasetSplit(selected, labels[indices], indices, self.name, split, classes,
                               clean_targets=None if clean is None else clean[indices],
                               class_names=tuple(c["name"] for c in metadata["classes"]),
                               source="user_manifest")
