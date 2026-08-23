from __future__ import annotations

"""Built-in adapters for existing CIFAR, synthetic, and UCI sources."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .binary_synthetic import (
    generate_synthetic_binary_2d,
    generate_synthetic_binary_high_dim,
)
from .binary_benchmarks import stratified_binary_splits
from .cifar import load_cifar10, load_cifar100
from .contracts import DataSpec, RawDatasetSplit
from .multiclass_synthetic import generate_synthetic_multiclass
from .preprocessing import BinaryPreprocessingConfig, BinaryPreprocessor
from .registry import DatasetRegistry


class CifarAdapter:
    def __init__(self, name: str, classes: int) -> None:
        self.name = name
        self.aliases = (name.replace("cifar", "cifar_"),)
        self.classes = classes

    def validate(self, spec: DataSpec) -> None:
        # The strict pickle reader performs file-level validation.  Keeping
        # this adapter check side-effect free also permits injected in-memory
        # sources in tests and downstream integrations.
        del spec

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        del seed
        if split not in {"train", "test"}:
            raise ValueError("CIFAR source split must be train or test")
        corpus = (
            load_cifar10(spec.root, split)
            if self.name == "cifar10"
            else load_cifar100(spec.root, split)
        )
        indices = np.arange(len(corpus), dtype=np.int64)
        return RawDatasetSplit(
            corpus.images,
            corpus.labels,
            indices,
            self.name,
            split,
            self.classes,
            clean_targets=corpus.labels,
            class_names=corpus.class_names,
            source="official_pickle",
        )


class CifarBinaryViewAdapter:
    name = "cifar10_airplane_automobile"
    aliases = ("cifar10_binary", "cifar_10_airplane_automobile")

    def validate(self, spec: DataSpec) -> None:
        CifarAdapter("cifar10", 10).validate(spec)

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        if "folds" in spec.options:
            if split not in {"train", "test"}:
                raise ValueError("folded CIFAR binary view exposes train and test")
            source_train = CifarAdapter("cifar10", 10).load(spec, "train", seed=seed)
            source_test = CifarAdapter("cifar10", 10).load(spec, "test", seed=seed)
            train_positions = np.flatnonzero(np.isin(source_train.observed_targets, (0, 1)))
            test_positions = np.flatnonzero(np.isin(source_test.observed_targets, (0, 1)))
            inputs = np.concatenate((source_train.inputs[train_positions], source_test.inputs[test_positions]))
            targets = np.concatenate((source_train.observed_targets[train_positions], source_test.observed_targets[test_positions]))
            indices = np.concatenate((source_train.global_indices[train_positions], len(source_train) + source_test.global_indices[test_positions]))
            maximum = spec.options.get("max_samples")
            if maximum is not None and int(maximum) < targets.size:
                maximum = int(maximum)
                random = np.random.default_rng(seed)
                selected = []
                for label in (0, 1):
                    candidates = np.flatnonzero(targets == label)
                    random.shuffle(candidates)
                    selected.append(candidates[: maximum // 2])
                limited = np.sort(np.concatenate(selected))
                inputs, targets, indices = inputs[limited], targets[limited], indices[limited]
            folds = stratified_binary_splits(targets, folds=int(spec.options["folds"]), seed=seed)
            fold_index = int(spec.options.get("fold_index", 0))
            if not 0 <= fold_index < len(folds):
                raise ValueError("data.fold_index is outside the configured fold range")
            selected = folds[fold_index][0 if split == "train" else 1]
            return RawDatasetSplit(
                inputs[selected], targets[selected], indices[selected], self.name, split, 2,
                clean_targets=targets[selected], class_names=("airplane", "automobile"),
                source=f"official_pickle:fold={fold_index}/{len(folds)}",
            )
        raw = CifarAdapter("cifar10", 10).load(spec, split, seed=seed)
        positions = np.flatnonzero(np.isin(raw.observed_targets, (0, 1)))
        return RawDatasetSplit(
            raw.inputs[positions],
            raw.observed_targets[positions],
            raw.global_indices[positions],
            self.name,
            split,
            2,
            clean_targets=raw.clean_targets[positions],
            class_names=("airplane", "automobile"),
            source=raw.source,
        )


class SyntheticAdapter:
    aliases: tuple[str, ...] = ()

    def __init__(self, name: str) -> None:
        self.name = name

    def validate(self, spec: DataSpec) -> None:
        options = spec.options
        if self.name == "synthetic_multiclass":
            for key in ("num_classes", "dimension"):
                if key not in options:
                    raise ValueError(f"{self.name} requires data.{key}")
        elif self.name == "synthetic_binary_high_dim" and "dimension" not in options:
            raise ValueError(f"{self.name} requires data.dimension")

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        size_key = f"{split}_size"
        default = {"train": 200, "validation": 100, "test": 100}.get(split)
        if default is None:
            raise ValueError("synthetic split must be train, validation, or test")
        size = int(spec.options.get(size_key, default))
        offsets = {
            "train": 0,
            "validation": int(spec.options.get("train_size", 200)),
            "test": int(spec.options.get("train_size", 200))
            + int(spec.options.get("validation_size", 100)),
        }
        split_seed = seed + {"train": 1, "validation": 2, "test": 3}[split]
        if self.name == "synthetic_multiclass":
            data = generate_synthetic_multiclass(
                size,
                int(spec.options["dimension"]),
                int(spec.options["num_classes"]),
                split_seed,
                start_index=offsets[split],
                split=split,
            )
            classes = data.num_classes
        elif self.name == "synthetic_binary_high_dim":
            data = generate_synthetic_binary_high_dim(
                size,
                int(spec.options["dimension"]),
                split_seed,
                start_index=offsets[split],
                split=split,
            )
            classes = 2
        else:
            data = generate_synthetic_binary_2d(
                size,
                split_seed,
                start_index=offsets[split],
                split=split,
            )
            classes = 2
        return RawDatasetSplit(
            data.features,
            data.labels,
            data.global_indices,
            self.name,
            split,
            classes,
            clean_targets=data.labels,
            source="deterministic_generator",
        )


class UciBinaryAdapter:
    name = "uci_binary"
    aliases = ("uci_statlog_heart", "statlog_heart")

    def validate(self, spec: DataSpec) -> None:
        path = spec.path
        if path is None:
            raise ValueError("UCI datasets require data.path")
        if not path.is_file():
            raise FileNotFoundError(f"UCI dataset file does not exist: {path}")
        expected_sha = spec.options.get("sha256")
        if expected_sha is not None:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual.lower() != str(expected_sha).lower():
                raise ValueError("UCI source SHA-256 does not match data.sha256")

    @staticmethod
    def _split_rows(targets: np.ndarray, options: dict, seed: int) -> dict[str, np.ndarray]:
        split = dict(options.get("split", {}) or {})
        split_seed = int(split.get("seed", seed))
        validation_fraction = float(split.get("validation_fraction", 0.0))
        test_fraction = float(split.get("test_fraction", 0.0))
        if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
            raise ValueError("UCI validation/test fractions must be non-negative and sum to less than one")
        random = np.random.default_rng(split_seed)
        grouped: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
        for label in np.unique(targets):
            rows = np.flatnonzero(targets == label)
            rows = rows[random.permutation(rows.size)]
            validation_count = int(round(rows.size * validation_fraction))
            test_count = int(round(rows.size * test_fraction))
            if validation_fraction and validation_count == 0:
                validation_count = 1
            if test_fraction and test_count == 0:
                test_count = 1
            if validation_count + test_count >= rows.size:
                raise ValueError("UCI split fractions leave no training samples")
            grouped["validation"].append(rows[:validation_count])
            grouped["test"].append(rows[validation_count:validation_count + test_count])
            grouped["train"].append(rows[validation_count + test_count:])
        return {
            name: np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
            for name, parts in grouped.items()
        }

    def load(self, spec: DataSpec, split: str, *, seed: int) -> RawDatasetSplit:
        if split not in {"train", "validation", "test"}:
            raise ValueError("UCI split must be train, validation, or test")
        self.validate(spec)
        preprocessing = dict(spec.options.get("preprocessing", {}) or {})
        config = BinaryPreprocessingConfig(
            file_format=str(preprocessing.get("format", "delimited")),
            delimiter=preprocessing.get("delimiter"),
            target_column=preprocessing.get("target_column", -1),
            has_header=bool(preprocessing.get("has_header", False)),
            missing_policy=str(preprocessing.get("missing_policy", "error")),
            categorical_policy=str(preprocessing.get("categorical_policy", "one_hot")),
            standardize=bool(preprocessing.get("standardize", False)),
            label_values=None
            if preprocessing.get("label_values") is None
            else tuple(map(str, preprocessing["label_values"])),
        )
        probe = BinaryPreprocessor(config)
        targets = probe.read_targets(spec.path)
        expected_samples = spec.options.get("expected_samples")
        if expected_samples is not None and targets.size != int(expected_samples):
            raise ValueError("UCI source sample count does not match data.expected_samples")
        rows = self._split_rows(targets, dict(spec.options), seed)
        processor = BinaryPreprocessor(config).fit(spec.path, row_indices=rows["train"])
        benchmark = processor.transform(
            spec.path,
            dataset=spec.name,
            split=split,
            row_indices=rows[split],
        )
        return RawDatasetSplit(
            benchmark.features,
            benchmark.targets,
            benchmark.global_indices,
            spec.name,
            split,
            2,
            clean_targets=benchmark.targets,
            source=f"{Path(spec.path).resolve()}#{processor.source_fingerprint}",
        )

    def identity_artifacts(self, spec: DataSpec, *, seed: int) -> dict[str, dict]:
        """Return deterministic preprocessing and split identities for a run."""

        preprocessing = dict(spec.options.get("preprocessing", {}) or {})
        config = BinaryPreprocessingConfig(
            file_format=str(preprocessing.get("format", "delimited")),
            delimiter=preprocessing.get("delimiter"),
            target_column=preprocessing.get("target_column", -1),
            has_header=bool(preprocessing.get("has_header", False)),
            missing_policy=str(preprocessing.get("missing_policy", "error")),
            categorical_policy=str(preprocessing.get("categorical_policy", "one_hot")),
            standardize=bool(preprocessing.get("standardize", False)),
            label_values=None
            if preprocessing.get("label_values") is None
            else tuple(map(str, preprocessing["label_values"])),
        )
        probe = BinaryPreprocessor(config)
        targets = probe.read_targets(spec.path)
        rows = self._split_rows(targets, dict(spec.options), seed)
        processor = BinaryPreprocessor(config).fit(spec.path, row_indices=rows["train"])
        preprocessing_state = json.loads(json.dumps(processor.state_dict()))
        preprocessing_state["state_hash"] = hashlib.sha256(json.dumps(
            preprocessing_state, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        split_config = dict(spec.options.get("split", {}) or {})
        split_state = {
            "seed": int(split_config.get("seed", seed)),
            "validation_fraction": float(split_config.get("validation_fraction", 0.0)),
            "test_fraction": float(split_config.get("test_fraction", 0.0)),
            "train_indices": rows["train"].tolist(),
            "validation_indices": rows["validation"].tolist(),
            "test_indices": rows["test"].tolist(),
        }
        split_state["split_hash"] = hashlib.sha256(json.dumps(
            split_state, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        return {
            "preprocessing_state.json": preprocessing_state,
            "split_manifest.json": split_state,
        }


def add_existing_sources(registry: DatasetRegistry) -> None:
    registry.add(CifarAdapter("cifar10", 10))
    registry.add(CifarAdapter("cifar100", 100))
    registry.add(CifarBinaryViewAdapter())
    registry.add(SyntheticAdapter("synthetic_binary_2d"))
    registry.add(SyntheticAdapter("synthetic_binary_high_dim"))
    registry.add(SyntheticAdapter("synthetic_multiclass"))
    registry.add(UciBinaryAdapter())


__all__ = ["add_existing_sources"]
