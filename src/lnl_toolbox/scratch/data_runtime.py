"""Scratch-native data contracts and materialisation pipeline.

This module deliberately depends only on Scratch code and optional runtime
packages.  The legacy data service is an oracle for tests, never a runtime
dependency of Scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROLE_NAMES = {
    "train", "train_eval", "noisy_validation", "clean_validation",
    "trusted_validation", "test",
}


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Scratch data runtime requires PyTorch") from exc
    return torch


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _dataset_catalog_path() -> Path:
    """Resolve the shared local dataset catalog without importing legacy code."""

    configured = os.environ.get("LNL_DATA_CATALOG")
    if configured:
        return Path(configured).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return (Path(local) / "lnl-toolbox" / "datasets.json").resolve()
    return (Path.home() / ".lnl-toolbox" / "datasets.json").resolve()


def registered_dataset_catalog() -> dict[str, dict[str, Any]]:
    """Read the shared registration catalog as plain, runtime-safe metadata.

    Registration is stored as JSON by the data-management page. Reading that
    file here keeps Scratch independent from the legacy DataService while
    allowing a selected alias to resolve to its registered adapter/path.
    """

    path = _dataset_catalog_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    records = raw.get("datasets") if isinstance(raw, Mapping) else None
    if not isinstance(records, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for alias, value in records.items():
        if not isinstance(value, Mapping):
            continue
        key = str(alias).strip().lower().replace(" ", "-")
        if not key:
            continue
        data = value.get("data", {})
        data = dict(data) if isinstance(data, Mapping) else {}
        adapter = str(value.get("adapter") or data.get("name") or "").strip().lower().replace("-", "_")
        if not adapter:
            continue
        profile = value.get("profile")
        evidence = value.get("evidence")
        if isinstance(profile, Mapping):
            if profile.get("num_classes") is not None and _is_missing(data.get("num_classes")):
                data["num_classes"] = profile["num_classes"]
            if profile.get("input_shape") is not None and _is_missing(data.get("input_shape")):
                data["input_shape"] = profile["input_shape"]
        if isinstance(evidence, Mapping) and evidence.get("classes") is not None and _is_missing(data.get("num_classes")):
            data["num_classes"] = evidence["classes"]
        result[key] = {
            "alias": key,
            "adapter": adapter,
            "data": data,
            "state": str(value.get("state", "registered")),
            "error": value.get("error"),
            "profile": profile if isinstance(profile, Mapping) else None,
            "evidence": evidence if isinstance(evidence, Mapping) else None,
        }
    return result


def registered_dataset_config(alias: object) -> dict[str, Any] | None:
    """Return one registered alias and its adapter-backed source config."""

    key = str(alias).strip().lower().replace(" ", "-")
    return registered_dataset_catalog().get(key)


def _as_input(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("input", "inputs", "image", "images", "x"):
            if key in value:
                return value[key]
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def _item_targets(value: Any) -> tuple[Any, Any | None, int | None]:
    if isinstance(value, Mapping):
        observed = None
        for key in ("observed_target", "target", "targets", "label", "labels", "y"):
            if key in value:
                observed = value[key]
                break
        clean = next((value[key] for key in ("clean_target", "clean_targets") if key in value), None)
        index = next((value[key] for key in ("index", "indices") if key in value), None)
        if observed is None:
            raise ValueError("Scratch source sample is missing observed_target")
        return observed, clean, None if index is None else int(index)
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        raise ValueError("Scratch source samples must contain input and observed_target")
    return value[1], value[3] if len(value) > 3 else None, int(value[2]) if len(value) > 2 else None


@dataclass(frozen=True)
class ScratchSample:
    input: Any
    index: int
    observed_target: int
    clean_target: int | None = None

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("Scratch sample index must be non-negative")
        if int(self.observed_target) < 0:
            raise ValueError("Scratch observed_target must be non-negative")
        if self.clean_target is not None and int(self.clean_target) < 0:
            raise ValueError("Scratch clean_target must be non-negative")


@dataclass(frozen=True)
class ScratchSplit:
    dataset: str
    split: str
    samples: tuple[ScratchSample, ...]
    num_classes: int
    version: str = "1"

    def __post_init__(self) -> None:
        indices = [int(sample.index) for sample in self.samples]
        if len(indices) != len(set(indices)):
            raise ValueError(f"{self.split} sample indices must be unique")
        if int(self.num_classes) < 2:
            raise ValueError("Scratch datasets require at least two classes")
        for sample in self.samples:
            if sample.observed_target >= self.num_classes:
                raise ValueError("observed_target is outside num_classes")
            if sample.clean_target is not None and sample.clean_target >= self.num_classes:
                raise ValueError("clean_target is outside num_classes")

    def subset(self, positions: Sequence[int], split: str | None = None) -> "ScratchSplit":
        return ScratchSplit(self.dataset, split or self.split,
                            tuple(self.samples[int(position)] for position in positions),
                            self.num_classes, self.version)

    @property
    def has_clean_targets(self) -> bool:
        return bool(self.samples) and all(sample.clean_target is not None for sample in self.samples)

    @property
    def namespace(self) -> tuple[str, str, str]:
        """Stable split identity; sample indices are only unique inside it."""
        return (self.dataset, self.split, self.version)


class ScratchRoleDataset:
    """A role view over already materialised Scratch samples."""

    def __init__(self, split: ScratchSplit, *, transform: Callable[[Any], Any] | None = None,
                 views: Mapping[str, Callable[[Any], Any] | None] | None = None) -> None:
        self.split = split
        self.samples = split.samples
        self.transform = transform
        self.views = dict(views or {})
        self.indices = [sample.index for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self.samples[int(item)]
        value = _decode_input(sample.input)
        input_value = value if self.transform is None else self.transform(value)
        result = {
            "inputs": input_value,
            "targets": int(sample.observed_target),
            "indices": int(sample.index),
            "clean_targets": None if sample.clean_target is None else int(sample.clean_target),
        }
        if self.views:
            view_values = {}
            for name, transform in self.views.items():
                view_input = _decode_input(sample.input)
                view_values[name] = view_input if transform is None else transform(view_input)
            result["views"] = view_values
            if "strong" in view_values:
                result["strong_input"] = view_values["strong"]
        return result


def _decode_input(value: Any) -> Any:
    import numpy as np
    import torch
    if isinstance(value, (str, Path)):
        from PIL import Image
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, np.ndarray):
        if value.ndim in (2, 3) and value.dtype == np.uint8:
            from PIL import Image
            return Image.fromarray(value)
        return torch.as_tensor(value)
    return value


def _to_tensor_input(value: Any) -> Any:
    """Pickle-safe tensor conversion used by worker DataLoaders."""
    torch = _torch()
    if isinstance(value, torch.Tensor):
        result = value
        if result.ndim == 3 and result.shape[-1] in {1, 3} and result.shape[0] not in {1, 3}:
            result = result.permute(2, 0, 1)
        if result.dtype == torch.uint8:
            result = result.float().div(255.0)
        return result.float()
    from torchvision import transforms
    return transforms.ToTensor()(value)


def _to_uint8_image(value: Any) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return value


class ScratchBatch(dict):
    """Canonical batch mapping with read-only aliases for old consumers.

    The stored/output fields remain exactly ``inputs``, ``targets``,
    ``indices`` and ``clean_targets`` (plus optional views). ``get`` and
    subscription aliases let existing Scratch algorithm blocks read the same
    tensors while they are migrated to the canonical names.
    """

    _ALIASES = {"input": "inputs", "images": "inputs", "target": "targets",
                "labels": "targets", "index": "indices"}

    def get(self, key: Any, default: Any = None) -> Any:
        return super().get(self._ALIASES.get(key, key), default)

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(self._ALIASES.get(key, key))


def collate_scratch_batch(rows: Sequence[Mapping[str, Any]]) -> ScratchBatch:
    torch = _torch()
    if not rows:
        raise ValueError("cannot collate an empty Scratch batch")
    inputs = torch.utils.data.default_collate([row["inputs"] for row in rows])
    targets = torch.as_tensor([int(row["targets"]) for row in rows], dtype=torch.long)
    indices = torch.as_tensor([int(row["indices"]) for row in rows], dtype=torch.long)
    clean_values = [row.get("clean_targets") for row in rows]
    clean = None if any(value is None for value in clean_values) else torch.as_tensor(clean_values, dtype=torch.long)
    result = ScratchBatch(inputs=inputs, targets=targets, indices=indices, clean_targets=clean)
    if all("views" in row for row in rows):
        names = set(rows[0]["views"])
        result["views"] = {
            name: torch.utils.data.default_collate([row["views"][name] for row in rows])
            for name in names
        }
        if "strong" in result["views"]:
            result["strong_input"] = result["views"]["strong"]
    return result


class ScratchNoiseManifest:
    def __init__(self, split: str, indices: Any, clean_targets: Any, observed_targets: Any,
                 noise_type: str = "none", seed: int = 0, rate: float = 0.0,
                 transition_matrix: Any | None = None, dataset: str = "",
                 per_sample_transition: Any | None = None, metadata: Mapping[str, Any] | None = None) -> None:
        self.dataset = str(dataset)
        self.split = str(split)
        self.global_indices = indices
        self.clean_targets = clean_targets
        self.noisy_targets = observed_targets
        self.noise_type = str(noise_type)
        self.seed = int(seed)
        self.requested_rate = float(rate)
        self.transition_matrix = transition_matrix
        self.per_sample_transition = per_sample_transition
        self.metadata = dict(metadata or {})
        payload = json.dumps({"dataset": self.dataset, "split": self.split, "indices": _tolist(indices),
                              "clean": _tolist(clean_targets), "observed": _tolist(observed_targets),
                              "transition": _tolist(per_sample_transition), "metadata": self.metadata},
                             sort_keys=True).encode()
        self.mapping_hash = hashlib.sha256(payload).hexdigest()


def _tolist(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _source_split(source: Any, split: str) -> Any:
    if isinstance(source, Mapping):
        if split in source:
            return source[split]
        raise ValueError(f"Scratch source does not provide `{split}` split")
    loader = getattr(source, "load", None)
    if callable(loader):
        return loader(split)
    value = getattr(source, split, None)
    if value is not None:
        return value
    raise ValueError(f"Scratch source does not provide `{split}` split")


def _normalise_split(value: Any, *, dataset: str, split: str, classes: int,
                     clean_default: bool = True) -> ScratchSplit:
    if isinstance(value, ScratchSplit):
        return value
    samples: list[ScratchSample] = []
    for position in range(len(value)):
        raw = value[position]
        observed, clean, index = _item_targets(raw)
        if clean is None and clean_default:
            clean = observed
        samples.append(ScratchSample(_as_input(raw), position if index is None else index,
                                     int(observed), None if clean is None else int(clean)))
    return ScratchSplit(dataset, split, tuple(samples), int(classes))


def _restrict_binary(split: ScratchSplit, requested: Any) -> ScratchSplit:
    if not isinstance(requested, Sequence) or len(requested) != 2 or isinstance(requested, (str, bytes)):
        return split
    names = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
             "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
    class_ids = [names.get(str(value).lower(), int(value) if str(value).isdigit() else None) for value in requested]
    if any(value is None for value in class_ids):
        raise ValueError("binary_classes must contain known CIFAR class names or integer ids")
    mapped = []
    for sample in split.samples:
        if sample.observed_target not in class_ids:
            continue
        clean = None if sample.clean_target is None else sample.clean_target
        if clean is not None and clean not in class_ids:
            continue
        mapped.append(ScratchSample(sample.input, sample.index, class_ids.index(sample.observed_target),
                                    None if clean is None else class_ids.index(clean)))
    return ScratchSplit(split.dataset, split.split, tuple(mapped), 2, split.version)


def _synthetic_source(data: Mapping[str, Any], seed: int) -> dict[str, ScratchSplit]:
    torch = _torch()
    name = str(data.get("name", "synthetic"))
    classes = int(data.get("classes", 2))
    image_shape = tuple(int(value) for value in data.get("image_shape", ()))
    features = int(data.get("features", 4)) if not image_shape else int(torch.tensor(image_shape).prod().item())
    train_size = int(data.get("train_size", data.get("samples", 64)))
    test_size = int(data.get("test_size", max(8, train_size // 4)))
    generator = torch.Generator().manual_seed(int(data.get("data_seed", seed)))
    weights = torch.randn(features, classes, generator=generator)
    def make(size: int, split: str, offset: int) -> ScratchSplit:
        inputs = (torch.rand(size, *image_shape, generator=generator)
                  if image_shape else torch.randn(size, features, generator=generator))
        labels = (inputs.reshape(size, -1) @ weights).argmax(1).long()
        return ScratchSplit(name, split, tuple(ScratchSample(inputs[i], i, int(labels[i]), int(labels[i])) for i in range(size)), classes)
    return {"train": make(train_size, "train", 0), "test": make(test_size, "test", train_size)}


def _registered_uci_source(data: Mapping[str, Any], name: str, seed: int) -> tuple[ScratchSplit, ScratchSplit | None, ScratchSplit]:
    """Load a registered UCI-style binary file using only Scratch primitives."""

    import numpy as np

    path = data.get("path")
    if not path:
        raise ValueError("registered uci_binary requires a file path")
    preprocessing = data.get("preprocessing", {})
    preprocessing = dict(preprocessing) if isinstance(preprocessing, Mapping) else {}
    file_format = str(preprocessing.get("format", "delimited")).lower()
    delimiter = None if file_format in {"whitespace", "space", "text"} else str(preprocessing.get("delimiter", data.get("delimiter", ",")))
    raw = np.genfromtxt(
        str(path), delimiter=delimiter, skip_header=1 if bool(preprocessing.get("has_header", False)) else 0,
    )
    if raw.ndim != 2 or raw.shape[0] < 3 or raw.shape[1] < 2 or not np.isfinite(raw).all():
        raise ValueError("registered uci_binary must contain a finite 2-D feature/label table")
    target_column = int(preprocessing.get("target_column", data.get("target_column", -1)))
    target_column = target_column if target_column >= 0 else raw.shape[1] + target_column
    if not 0 <= target_column < raw.shape[1]:
        raise ValueError("uci_binary target_column is outside the table")
    labels = raw[:, target_column].astype(np.int64)
    unique = np.unique(labels)
    if unique.size != 2:
        raise ValueError("registered uci_binary must contain exactly two target classes")
    labels = np.searchsorted(unique, labels).astype(np.int64)
    features = np.delete(raw, target_column, axis=1).astype(np.float32)
    split_cfg = data.get("split", {})
    split_cfg = dict(split_cfg) if isinstance(split_cfg, Mapping) else {}
    validation_fraction = float(split_cfg.get("validation_fraction", 0.0))
    test_fraction = float(split_cfg.get("test_fraction", 0.2))
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("uci_binary split fractions must be non-negative and sum to less than one")
    rng = np.random.default_rng(int(split_cfg.get("seed", seed)))
    groups: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        rng.shuffle(positions)
        validation_count = int(round(len(positions) * validation_fraction))
        test_count = int(round(len(positions) * test_fraction))
        if validation_fraction and validation_count == 0:
            validation_count = 1
        if test_fraction and test_count == 0:
            test_count = 1
        if validation_count + test_count >= len(positions):
            raise ValueError("uci_binary split fractions leave no training samples")
        groups["validation"].append(positions[:validation_count])
        groups["test"].append(positions[validation_count:validation_count + test_count])
        groups["train"].append(positions[validation_count + test_count:])
    indices_by_split = {
        split: np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
        for split, parts in groups.items()
    }
    if bool(preprocessing.get("standardize", False)):
        train_values = features[indices_by_split["train"]]
        mean, std = train_values.mean(axis=0), train_values.std(axis=0)
        std[std == 0] = 1.0
        features = (features - mean) / std
    def make(split: str) -> ScratchSplit:
        positions = indices_by_split[split]
        return ScratchSplit(name, split, tuple(
            ScratchSample(_torch().as_tensor(features[position], dtype=_torch().float32), int(position), int(labels[position]), int(labels[position]))
            for position in positions
        ), 2)
    return make("train"), make("validation"), make("test")


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _registered_cifar_source(data: Mapping[str, Any], name: str, adapter_name: str) -> tuple[ScratchSplit, ScratchSplit | None, ScratchSplit] | None:
    """Read extracted CIFAR files directly from a registered local root.

    The data-registration page accepts both the torchvision download layout
    and the official extracted layout (``data_batch_1``/``test_batch``).  The
    latter is deliberately handled here instead of relying on torchvision's
    directory convention so a registered path is usable as-is.
    """

    import pickle
    import numpy as np

    root_value = data.get("root") or data.get("path")
    if not root_value:
        return None
    root = Path(str(root_value)).expanduser()
    base_name = "cifar100" if adapter_name in {"cifar100", "cifar100n"} else "cifar10"
    candidates = [root]
    if base_name == "cifar10":
        candidates.append(root / "cifar-10-batches-py")
    else:
        candidates.append(root / "cifar-100-python")
    layout = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if layout is None:
        return None

    def read(path: Path) -> Mapping[Any, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"missing CIFAR file: {path}")
        with path.open("rb") as handle:
            value = pickle.load(handle, encoding="bytes")
        if not isinstance(value, Mapping):
            raise ValueError(f"CIFAR file must contain a mapping: {path}")
        return value

    def value(record: Mapping[Any, Any], key: str) -> Any:
        if key in record:
            return record[key]
        encoded = key.encode()
        if encoded in record:
            return record[encoded]
        raise KeyError(f"CIFAR file is missing {key!r}")

    def split_arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
        if base_name == "cifar10":
            files = [layout / f"data_batch_{index}" for index in range(1, 6)] if split == "train" else [layout / "test_batch"]
            records = [read(path) for path in files]
            images = np.concatenate([np.asarray(value(record, "data"), dtype=np.uint8) for record in records])
            labels = np.concatenate([np.asarray(value(record, "labels"), dtype=np.int64) for record in records])
        else:
            record = read(layout / split)
            images = np.asarray(value(record, "data"), dtype=np.uint8)
            labels = np.asarray(value(record, "fine_labels"), dtype=np.int64)
        if images.ndim != 2 or images.shape[1] != 3072 or labels.shape != (images.shape[0],):
            raise ValueError(f"invalid {base_name} {split} payload under {layout}")
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
        return images, labels

    train_inputs, train_labels = split_arrays("train")
    test_inputs, test_labels = split_arrays("test")
    classes = 100 if base_name == "cifar100" else 10
    def make(split: str, inputs: np.ndarray, labels: np.ndarray, clean: Any | None = None) -> ScratchSplit:
        clean_values = labels if clean is None else clean
        samples = tuple(ScratchSample(inputs[index], index, int(labels[index]), int(clean_values[index]))
                        for index in range(labels.size))
        return ScratchSplit(name, split, samples, classes)

    train = make("train", train_inputs, train_labels)
    test = make("test", test_inputs, test_labels)
    if adapter_name in {"cifar10n", "cifar100n"}:
        noise_path = data.get("noise_path") or data.get("labels_path")
        if not noise_path:
            raise ValueError(f"registered {adapter_name} requires noise_path/labels_path")
        labels_payload = _load_external_payload(Path(str(noise_path)))
        variant = str(data.get("noise_variant", "aggre_label" if adapter_name == "cifar10n" else "noisy_label"))
        if "clean_label" not in labels_payload or variant not in labels_payload:
            raise KeyError(f"{adapter_name} labels must contain 'clean_label' and {variant!r}")
        noisy = _torch().as_tensor(labels_payload[variant], dtype=_torch().long)
        clean = _torch().as_tensor(labels_payload["clean_label"], dtype=_torch().long)
        if noisy.numel() != len(train.samples) or clean.numel() != len(train.samples):
            raise ValueError(f"{adapter_name} labels must align with the full training split")
        train = ScratchSplit(name, "train", tuple(
            ScratchSample(sample.input, sample.index, int(noisy[index]), int(clean[index]))
            for index, sample in enumerate(train.samples)
        ), classes)
    return train, None, test


def _registered_mnist_source(data: Mapping[str, Any], name: str, adapter_name: str) -> tuple[ScratchSplit, ScratchSplit | None, ScratchSplit] | None:
    """Read official MNIST/Fashion-MNIST IDX files from a registered root."""

    import gzip
    import struct
    import numpy as np

    root_value = data.get("root") or data.get("path")
    if not root_value:
        return None
    root = Path(str(root_value)).expanduser()
    directory_name = "FashionMNIST" if adapter_name == "fashion_mnist" else "MNIST"
    directories = (root, root / "raw", root / directory_name / "raw")

    def find_pair(split: str) -> tuple[Path, Path] | None:
        prefix = "train" if split == "train" else "t10k"
        names = (f"{prefix}-images-idx3-ubyte", f"{prefix}-labels-idx1-ubyte")
        for directory in directories:
            for suffix in ("", ".gz"):
                pair = (directory / f"{names[0]}{suffix}", directory / f"{names[1]}{suffix}")
                if all(path.is_file() for path in pair):
                    return pair
        return None

    def read_bytes(path: Path) -> bytes:
        return gzip.open(path, "rb").read() if path.suffix.lower() == ".gz" else path.read_bytes()

    def read_split(split: str) -> tuple[np.ndarray, np.ndarray] | None:
        pair = find_pair(split)
        if pair is None:
            return None
        image_payload, label_payload = read_bytes(pair[0]), read_bytes(pair[1])
        if len(image_payload) < 16 or len(label_payload) < 8:
            raise ValueError(f"{adapter_name} IDX files are truncated under {root}")
        image_magic, count, rows, columns = struct.unpack(">IIII", image_payload[:16])
        label_magic, label_count = struct.unpack(">II", label_payload[:8])
        if image_magic != 2051 or label_magic != 2049 or count != label_count or rows != 28 or columns != 28:
            raise ValueError(f"{adapter_name} IDX dimensions or magic numbers are invalid")
        if len(image_payload) != 16 + count * rows * columns or len(label_payload) != 8 + count:
            raise ValueError(f"{adapter_name} IDX payload lengths are invalid")
        images = np.frombuffer(image_payload, dtype=np.uint8, offset=16).reshape(count, rows, columns).copy()
        labels = np.frombuffer(label_payload, dtype=np.uint8, offset=8).astype(np.int64, copy=True)
        if labels.size and labels.max() >= 10:
            raise ValueError(f"{adapter_name} labels are outside the class range")
        return images, labels

    train_values, test_values = read_split("train"), read_split("test")
    if train_values is None or test_values is None:
        return None
    def make(split: str, values: tuple[np.ndarray, np.ndarray]) -> ScratchSplit:
        images, labels = values
        return ScratchSplit(name, split, tuple(
            ScratchSample(images[index], index, int(labels[index]), int(labels[index]))
            for index in range(labels.size)
        ), 10)
    return make("train", train_values), None, make("test", test_values)


def _registered_clothing_source(data: Mapping[str, Any], name: str) -> tuple[ScratchSplit, ScratchSplit | None, ScratchSplit]:
    """Load the key-list/label-map layout used by registered Clothing1M data."""

    root_value = data.get("root") or data.get("path")
    if not root_value:
        raise ValueError("registered clothing1m requires a dataset root")
    root = Path(str(root_value)).expanduser()
    manifests = {
        "train": data.get("train_manifest", "noisy_train_key_list.txt"),
        "validation": data.get("validation_manifest", "clean_val_key_list.txt"),
        "test": data.get("test_manifest", "clean_test_key_list.txt"),
    }
    labels = {
        "train": data.get("train_labels", data.get("noisy_labels", "noisy_label_kv.txt")),
        "validation": data.get("validation_labels", data.get("clean_labels", "clean_label_kv.txt")),
        "test": data.get("test_labels", data.get("clean_labels", "clean_label_kv.txt")),
    }

    def resolve(value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else root / path

    def read_split(split: str) -> ScratchSplit:
        manifest_path, label_path = resolve(manifests[split]), resolve(labels[split])
        if not manifest_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(
                f"clothing1m {split} requires key list and label map under {root}"
            )
        keys = [line.strip().replace("\\", "/").removeprefix("./")
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError(f"clothing1m {split} key list must be non-empty and unique")
        mapping: dict[str, int] = {}
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                key, label = value.rsplit(maxsplit=1)
                key = key.replace("\\", "/").removeprefix("./")
                label = int(label)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid clothing1m label row {label_path}:{line_number}") from exc
            if not 0 <= label < 14:
                raise ValueError(f"clothing1m label outside [0, 14) at {label_path}:{line_number}")
            if key in mapping:
                raise ValueError(f"duplicate clothing1m image label at {label_path}:{line_number}")
            mapping[key] = label
        samples: list[ScratchSample] = []
        for index, key in enumerate(keys):
            if key not in mapping:
                raise KeyError(f"clothing1m label map does not cover image {key}")
            image = Path(key)
            image = image if image.is_absolute() else root / image
            if not image.is_file():
                raise FileNotFoundError(f"clothing1m image does not exist: {image}")
            observed = int(mapping[key])
            clean = observed if split != "train" else None
            samples.append(ScratchSample(image, index, observed, clean))
        return ScratchSplit(name, split, tuple(samples), 14)

    return read_split("train"), read_split("validation"), read_split("test")


def _registered_animal_source(data: Mapping[str, Any], name: str) -> tuple[ScratchSplit, ScratchSplit | None, ScratchSplit]:
    """Load the official binary or image-folder layout used by Animal-10N."""

    import re
    import numpy as np

    root_value = data.get("root") or data.get("path")
    if not root_value:
        raise ValueError("registered animal10n requires a dataset root")
    root = Path(str(root_value)).expanduser()
    class_names = ("cat", "lynx", "wolf", "coyote", "cheetah", "jaguar",
                   "chimpanzee", "orangutan", "hamster", "guinea pig")
    record_size = 4 + 4 + 3 * 64 * 64

    def binary_files(split: str) -> list[Path]:
        if split == "test":
            path = root / "test_batch.bin"
            return [path] if path.is_file() else []
        return sorted(root.glob("data_batch_*.bin"),
                      key=lambda value: int(re.search(r"(\d+)$", value.stem).group(1)))

    def folder(split: str) -> Path | None:
        names = ("training", "train") if split == "train" else ("testing", "test")
        return next((root / value for value in names if (root / value).is_dir()), None)

    def read_binary(split: str) -> tuple[Any, Any]:
        files = binary_files(split)
        if not files:
            raise FileNotFoundError(f"animal10n {split} binary files are missing under {root}")
        images, labels = [], []
        for path in files:
            payload = np.fromfile(path, dtype=np.uint8)
            if payload.size == 0 or payload.size % record_size:
                raise ValueError(f"animal10n binary file has invalid size: {path}")
            records = payload.reshape(-1, record_size)
            packed = np.ascontiguousarray(records[:, 4:8])
            candidates = (packed.view("<u4").reshape(-1), packed.view(">u4").reshape(-1))
            label = next((candidate.astype(np.int64) for candidate in candidates
                          if candidate.size and candidate.min() >= 0 and candidate.max() < 10), None)
            if label is None:
                raise ValueError(f"animal10n binary labels are invalid: {path}")
            labels.append(label)
            images.append(records[:, 8:].reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1).copy())
        return np.concatenate(images), np.concatenate(labels)

    def directory_label(value: str) -> int | None:
        normalized = re.sub(r"[\s_-]+", " ", value.strip().lower())
        match = re.fullmatch(r"(?:class\s*)?(\d+)", normalized)
        if match and 0 <= int(match.group(1)) < 10:
            return int(match.group(1))
        return {value: index for index, value in enumerate(class_names)}.get(normalized)

    def read_images(split: str) -> tuple[Any, Any]:
        directory = folder(split)
        if directory is None:
            raise FileNotFoundError(f"animal10n {split} image directory is missing under {root}")
        class_dirs = [value for value in directory.iterdir() if value.is_dir()]
        paths: list[Path] = []
        labels: list[int] = []
        if class_dirs:
            mapped = [(directory_label(value.name), value) for value in class_dirs]
            if len(mapped) != 10 or any(label is None for label, _ in mapped):
                raise ValueError(f"animal10n requires ten recognized class directories under {directory}")
            for label, class_dir in sorted(mapped):
                for image in sorted(class_dir.rglob("*")):
                    if image.is_file() and image.suffix.lower() in _IMAGE_SUFFIXES:
                        paths.append(image)
                        labels.append(int(label))
        else:
            for image in sorted(directory.iterdir()):
                if image.is_file() and image.suffix.lower() in _IMAGE_SUFFIXES:
                    if not image.name or not image.name[0].isdigit() or int(image.name[0]) >= 10:
                        raise ValueError(f"animal10n flat filename must begin with class 0-9: {image}")
                    paths.append(image)
                    labels.append(int(image.name[0]))
        if not paths:
            raise ValueError(f"animal10n {split} split contains no images")
        return tuple(paths), np.asarray(labels, dtype=np.int64)

    use_binary = bool(binary_files("train") and binary_files("test"))

    def make(split: str) -> ScratchSplit:
        inputs, labels = read_binary(split) if use_binary else read_images(split)
        clean_default = split != "train"
        samples = tuple(ScratchSample(inputs[index], index, int(label), int(label) if clean_default else None)
                        for index, label in enumerate(labels))
        return ScratchSplit(name, split, samples, 10)

    return make("train"), None, make("test")


def _load_source(plan: Mapping[str, Any], ctx: Mapping[str, Any]) -> tuple[ScratchSplit, ScratchSplit | None, ScratchSplit | None]:
    data = dict(plan.get("data", {}))
    name = str(data.get("name", "synthetic")).lower()
    adapter_name = str(data.get("adapter") or name).lower().replace("-", "_")
    classes = int(data.get("num_classes") or plan.get("semantics", {}).get("num_classes") or {
        "cifar10": 10, "cifar100": 100, "cifar10n": 10, "cifar100n": 100,
        "cifar10_airplane_automobile": 10, "cifar10_binary": 10,
        "cifar_10_airplane_automobile": 10, "mnist": 10, "fashion_mnist": 10,
    }.get(adapter_name, 2))
    if data.get("binary_classes") and adapter_name in {"cifar10", "cifar100"}:
        classes = 100 if adapter_name == "cifar100" else 10
    source = data.get("source")
    # A context-provided catalog may contain the same JSON record shape as the
    # shared catalog. Treat it as adapter/path configuration, not as a split
    # mapping, so mounted WebUI runs follow the same path as standalone runs.
    if isinstance(source, Mapping) and "train" not in source and "test" not in source:
        source_data = source.get("data", source)
        if isinstance(source_data, Mapping) and (source_data.get("adapter") or source_data.get("name")):
            data.update({key: value for key, value in source_data.items() if not _is_missing(value)})
            adapter_name = str(data.get("adapter") or data.get("name") or name).lower().replace("-", "_")
            classes = int(data.get("num_classes") or plan.get("semantics", {}).get("num_classes") or {
                "cifar10": 10, "cifar100": 100, "cifar10n": 10, "cifar100n": 100,
                "cifar10_airplane_automobile": 10, "cifar10_binary": 10,
                "cifar_10_airplane_automobile": 10, "mnist": 10, "fashion_mnist": 10,
            }.get(adapter_name, 2))
            source = None
    if source is not None:
        clean_default = bool(source.get("clean_targets_available", True)) if isinstance(source, Mapping) else bool(getattr(source, "clean_targets_available", True))
        train = _normalise_split(_source_split(source, "train"), dataset=name, split="train", classes=classes, clean_default=clean_default)
        test = _normalise_split(_source_split(source, "test"), dataset=name, split="test", classes=classes, clean_default=clean_default)
        validation = None
        try:
            validation = _normalise_split(_source_split(source, "validation"), dataset=name, split="validation", classes=classes, clean_default=clean_default)
        except ValueError:
            pass
        requested = data.get("binary_classes")
        return _restrict_binary(train, requested), (_restrict_binary(validation, requested) if validation is not None else None), _restrict_binary(test, requested)
    if adapter_name == "uci_binary":
        return _registered_uci_source(data, name, int(plan.get("seed", ctx.get("seed", 1))))
    if adapter_name in {"clothing1m", "clothing_1m"}:
        return _registered_clothing_source(data, name)
    if adapter_name in {"animal10n", "animal_10n"}:
        return _registered_animal_source(data, name)
    if adapter_name in {"synthetic", "synthetic_classification", "synthetic_binary_2d",
                        "synthetic_binary_high_dim", "synthetic_multiclass"}:
        if adapter_name == "synthetic_multiclass":
            data = {**data, "classes": data.get("classes", data.get("num_classes", 3)),
                    "features": data.get("features", data.get("dimension", 8))}
        elif adapter_name == "synthetic_binary_high_dim":
            data = {**data, "classes": 2,
                    "features": data.get("features", data.get("dimension", 16))}
        values = _synthetic_source(data, int(plan.get("seed", ctx.get("seed", 1))))
        requested = data.get("binary_classes")
        return _restrict_binary(values["train"], requested), None, _restrict_binary(values["test"], requested)
    if adapter_name in {"cifar10_airplane_automobile", "cifar10_binary", "cifar_10_airplane_automobile",
                        "cifar10", "cifar100", "cifar10n", "cifar100n", "mnist", "fashion_mnist"}:
        if adapter_name in {"cifar10_airplane_automobile", "cifar10_binary", "cifar_10_airplane_automobile",
                            "cifar10", "cifar100", "cifar10n", "cifar100n"}:
            direct = _registered_cifar_source(data, name, adapter_name)
            if direct is not None:
                train, validation, test = direct
                requested = data.get("binary_classes")
                if adapter_name in {"cifar10_airplane_automobile", "cifar10_binary", "cifar_10_airplane_automobile"}:
                    requested = requested or ("airplane", "automobile")
                return _restrict_binary(train, requested), validation, _restrict_binary(test, requested)
        if adapter_name in {"mnist", "fashion_mnist"}:
            direct = _registered_mnist_source(data, name, adapter_name)
            if direct is not None:
                train, validation, test = direct
                return train, validation, test
        try:
            from torchvision import datasets, transforms
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Scratch vision data requires torchvision") from exc
        root = str(data.get("root") or data.get("path") or "data")
        base_name = {"cifar10n": "cifar10", "cifar100n": "cifar100",
                     "cifar10_airplane_automobile": "cifar10",
                     "cifar10_binary": "cifar10",
                     "cifar_10_airplane_automobile": "cifar10"}.get(adapter_name, adapter_name)
        cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100,
               "mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}[base_name]
        kwargs = {"root": root, "download": bool(data.get("download", False)), "transform": transforms.ToTensor()}
        train = _normalise_split(cls(train=True, **kwargs), dataset=name, split="train", classes=classes, clean_default=True)
        test = _normalise_split(cls(train=False, **kwargs), dataset=name, split="test", classes=classes, clean_default=True)
        if adapter_name in {"cifar10n", "cifar100n"}:
            noise_path = data.get("noise_path") or data.get("labels_path")
            if not noise_path:
                raise ValueError(f"registered {adapter_name} requires noise_path/labels_path")
            labels_payload = _load_external_payload(Path(str(noise_path)))
            variant = str(data.get("noise_variant", "aggre_label" if adapter_name == "cifar10n" else "noisy_label"))
            clean_key = "clean_label"
            if clean_key not in labels_payload or variant not in labels_payload:
                raise KeyError(f"{adapter_name} labels must contain {clean_key!r} and {variant!r}")
            noisy = _torch().as_tensor(labels_payload[variant], dtype=_torch().long)
            clean = _torch().as_tensor(labels_payload[clean_key], dtype=_torch().long)
            if noisy.numel() != len(train.samples) or clean.numel() != len(train.samples):
                raise ValueError(f"{adapter_name} labels must align with the full training split")
            train = ScratchSplit(train.dataset, train.split, tuple(
                ScratchSample(sample.input, sample.index, int(noisy[position]), int(clean[position]))
                for position, sample in enumerate(train.samples)
            ), train.num_classes, train.version)
        requested = data.get("binary_classes")
        if adapter_name in {"cifar10_airplane_automobile", "cifar10_binary", "cifar_10_airplane_automobile"}:
            requested = requested or ("airplane", "automobile")
        return _restrict_binary(train, requested), None, _restrict_binary(test, requested)
    raise ValueError(
        f"Scratch cannot materialize dataset `{name}` with adapter `{adapter_name}`; "
        "registered datasets must use a supported Scratch adapter or provide explicit train/test source slots"
    )


def _split_train(train: ScratchSplit, plan: Mapping[str, Any]) -> tuple[ScratchSplit, ScratchSplit]:
    config = dict(plan.get("split", {}))
    size = int(config.get("validation_size", 0))
    if size <= 0:
        return train, ScratchSplit(train.dataset, "validation", tuple(), train.num_classes, train.version)
    if size >= len(train.samples):
        raise ValueError("validation_size must be smaller than the training source")
    import numpy as np
    labels = np.asarray([sample.clean_target if sample.clean_target is not None else sample.observed_target for sample in train.samples])
    rng = np.random.default_rng(int(config.get("seed", plan.get("seed", 1))))
    strategy = str(config.get("strategy", "random")).lower()
    if strategy in {"stratified", "official"}:
        val_positions = []
        for label in np.unique(labels):
            candidates = np.flatnonzero(labels == label)
            rng.shuffle(candidates)
            quota = int(size * len(candidates) / len(labels))
            val_positions.extend(candidates[:quota].tolist())
        remaining = size - len(val_positions)
        rest = np.setdiff1d(np.arange(len(labels)), np.asarray(val_positions, dtype=np.int64))
        rng.shuffle(rest)
        val_positions.extend(rest[:remaining].tolist())
    else:
        val_positions = rng.choice(len(train.samples), size=size, replace=False).tolist()
    val_set = set(int(value) for value in val_positions)
    train_positions = [i for i in range(len(train.samples)) if i not in val_set]
    return train.subset(train_positions), train.subset(val_positions, "validation")


def _fixture_source_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded synthetic source used by fixture execution.

    Fixture substitution belongs to source loading, not to a later materializer.
    The requested dataset name is retained for protocol inspection while the
    actual source is deliberately small and Scratch-native.
    """
    fixture_plan = dict(plan)
    fixture_data = dict(plan.get("data", {}))
    original_name = str(fixture_data.get("name", "")).lower()
    # Binary CIFAR recipes request named classes (airplane/automobile).  The
    # bounded fixture must still contain both requested classes; sampling a
    # sparse ten-class fixture can otherwise produce an empty train_eval
    # loader and make downstream snapshot blocks fail before any batch exists.
    fixture_adapter = str(fixture_data.get("adapter", "")).lower().replace("-", "_")
    fixture_classes = 2 if (
        original_name in {"synthetic_binary_2d", "synthetic_binary_high_dim"}
        or fixture_adapter in {"cifar10_airplane_automobile", "cifar10_binary", "cifar_10_airplane_automobile"}
        or bool(fixture_data.get("binary_classes"))
    ) else int(plan.get("semantics", {}).get("num_classes", 10))
    fixture_data.update({
        "name": "synthetic",
        "train_size": 8,
        "test_size": 4,
        "features": 2 if original_name == "synthetic_binary_2d" else 4,
        "classes": fixture_classes,
    })
    # A registered alias carries its real adapter in the plan.  Fixture runs
    # intentionally replace that source with the bounded Scratch generator;
    # leaving the adapter behind would route the synthetic fixture back into a
    # real CIFAR/MNIST loader and fail before the recipe can be exercised.
    fixture_data.pop("adapter", None)
    # Keep at least one learning sample when a formal recipe requests a
    # trusted subset larger than the bounded fixture.  This only affects
    # ``runtime_limits.fixture`` substitution; formal data sizes remain
    # untouched.
    if "num_clean" in fixture_data or "trusted_size" in fixture_data:
        key = "num_clean" if "num_clean" in fixture_data else "trusted_size"
        fixture_data[key] = min(int(fixture_data.get(key, 0)), 7)
    if original_name in {"cifar10", "cifar100"}:
        fixture_data["image_shape"] = (3, 32, 32)
    elif original_name in {"mnist", "fashion_mnist"}:
        fixture_data["image_shape"] = (1, 28, 28)
    fixture_plan["data"] = fixture_data
    fixture_split = dict(plan.get("split", {}))
    requested_validation = int(fixture_split.get("validation_size", 0))
    if requested_validation > 0:
        fixture_split["validation_size"] = min(requested_validation, 2)
    fixture_plan["split"] = fixture_split
    fixture_loader = dict(plan.get("loader", {}))
    fixture_loader["num_workers"] = 0
    fixture_plan["loader"] = fixture_loader
    noise = dict(plan.get("noise", {}))
    if str(noise.get("name", "none")).lower() in {"external", "external_torch"}:
        noise = {"name": "symmetric", "rate": float(noise.get("rate", 0.0)),
                 "seed": int(noise.get("seed", 1))}
    fixture_plan["noise"] = noise
    return fixture_plan


def load_sources(plan: Mapping[str, Any], ctx: Mapping[str, Any]) -> tuple[dict[str, Any], ScratchSplit, ScratchSplit | None, ScratchSplit]:
    """Load train/test/optional validation sources for one Load Dataset block."""
    runtime_limits = ctx.get("_runtime_limits", {}) if isinstance(ctx, Mapping) else {}
    effective = _fixture_source_plan(plan) if (
        isinstance(runtime_limits, Mapping)
        and bool(runtime_limits.get("fixture"))
        and plan.get("data", {}).get("source") is None
    ) else dict(plan)
    train, validation, test = _load_source(effective, ctx)
    return effective, train, validation, test


def split_source(train: ScratchSplit, validation_source: ScratchSplit | None,
                 plan: Mapping[str, Any]) -> tuple[ScratchSplit, ScratchSplit]:
    """Create the requested train/validation split without test fallback."""
    config = dict(plan.get("split", {}))
    strategy = str(config.get("strategy", "random")).lower()
    if strategy == "official":
        if validation_source is None:
            if int(config.get("validation_size", 0)) == 0:
                return train, ScratchSplit(train.dataset, "validation", tuple(), train.num_classes, train.version)
            raise ValueError("official split requires a source validation split")
        return train, validation_source
    train_split, validation = _split_train(train, plan)
    return train_split, validation


def _resolve_transition_matrix(value: Any, *, classes: int, rate: float) -> Any:
    if isinstance(value, str):
        if value != "formal_loss_correction" or classes != 10:
            raise ValueError(f"unsupported symbolic transition matrix `{value}`")
        # Formal Loss Correction CIFAR-10 matrix from the reproduction config.
        return [
            [1,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0.4,0,0.6,0,0,0,0,0,0,0],
            [0,0,0,0.6,0,0.4,0,0,0,0],
            [0,0,0,0,0.6,0,0,0.4,0,0],
            [0,0,0,0.4,0,0.6,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,1,0],
            [0,0.4,0,0,0,0,0,0,0,0.6],
        ]
    if value is None:
        return [[1.0 if i == j else 0.0 for j in range(classes)] for i in range(classes)]
    return value


def _resolve_external_path(config: Mapping[str, Any]) -> Path:
    explicit = [config.get(key) for key in ("path", "artifact_path", "external_path") if config.get(key)]
    env_value = os.environ.get(str(config["source_env"])) if config.get("source_env") else None
    if len({str(value) for value in explicit}) > 1 or (env_value and explicit and str(env_value) not in {str(value) for value in explicit}):
        raise ValueError("external noise sources conflict; specify only one matching path")
    value = explicit[0] if explicit else env_value
    if not value:
        raise ValueError("external noise requires an artifact path or source_env")
    path = Path(str(value))
    if path.is_dir():
        candidates = [path / "noise_manifest.npz", path / "noise.pt", path / "artifact.pt"]
        path = next((candidate for candidate in candidates if candidate.exists()), path)
    if not path.exists():
        raise FileNotFoundError(f"external noise artifact does not exist: {path}")
    return path


def _load_external_payload(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".npz":
        import numpy as np
        archive = np.load(path, allow_pickle=False)
        payload: dict[str, Any] = {key: archive[key] for key in archive.files}
        if "metadata_json" in payload:
            raw = payload["metadata_json"]
            payload["metadata"] = json.loads(str(raw.item() if hasattr(raw, "item") else raw))
        return payload
    payload = _torch().load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("external noise artifact must contain a mapping")
    return payload


def _instance_class_scores(config: Mapping[str, Any], samples: Sequence[ScratchSample], classes: int) -> Any:
    """Return explicitly supplied per-sample class scores, aligned by index."""
    torch = _torch()
    value = config.get("class_scores", config.get("instance_scores"))
    if value is None:
        raise ValueError("instance_dependent noise requires aligned class_scores")
    if isinstance(value, Mapping):
        try:
            value = [value[int(sample.index)] for sample in samples]
        except KeyError as exc:
            raise ValueError("instance class_scores are missing a Scratch sample index") from exc
    scores = torch.as_tensor(value, dtype=torch.float64)
    if scores.ndim != 2 or tuple(scores.shape) != (len(samples), classes):
        raise ValueError(f"class_scores must have shape [{len(samples)}, {classes}]")
    if not torch.isfinite(scores).all():
        raise ValueError("class_scores must be finite")
    return scores


def _sample_truncated_normal(size: int, mean: float, std: float, generator: Any) -> Any:
    torch = _torch()
    if std <= 0:
        raise ValueError("pdl rate_std must be positive")
    result = torch.empty(size, dtype=torch.float64)
    filled = 0
    while filled < size:
        draw = torch.normal(float(mean), float(std), (max(size - filled, 1) * 2,), generator=generator,
                             dtype=torch.float64)
        accepted = draw[(draw >= 0.0) & (draw <= 1.0)]
        take = min(int(accepted.numel()), size - filled)
        if take:
            result[filled:filled + take] = accepted[:take]
            filled += take
    return result


def _sample_instance_dependent(samples: Sequence[ScratchSample], clean: Any,
                               classes: int, rate: float, seed: int,
                               config: Mapping[str, Any]) -> tuple[Any, Any]:
    torch = _torch()
    scores = _instance_class_scores(config, samples, classes)
    shifted = scores - scores.max(dim=1, keepdim=True).values
    probabilities = torch.exp(shifted)
    probabilities[torch.arange(len(samples)), clean] = 0.0
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    ambiguity = 1.0 - probabilities.max(dim=1).values
    ambiguity = ambiguity / ambiguity.mean().clamp_min(1e-12)
    flip_probability = (float(rate) * ambiguity).clamp(0.0, 1.0)
    generator = torch.Generator().manual_seed(int(seed))
    flip = torch.rand(len(samples), generator=generator) < flip_probability
    noisy = clean.clone()
    for position in torch.nonzero(flip, as_tuple=False).flatten().tolist():
        noisy[position] = torch.multinomial(probabilities[position].float(), 1, generator=generator)
    transition = probabilities * flip_probability[:, None]
    transition[torch.arange(len(samples)), clean] = 1.0 - flip_probability
    return noisy, transition


def _sample_pdl(samples: Sequence[ScratchSample], clean: Any, classes: int,
                rate: float, seed: int, config: Mapping[str, Any]) -> tuple[Any, Any]:
    """Scratch-native implementation of PDL Algorithm 2 over raw sample inputs."""
    torch = _torch()
    features = []
    supplied_features = config.get("features")
    if supplied_features is not None:
        matrix = torch.as_tensor(supplied_features, dtype=torch.float64)
        if matrix.ndim != 2 or tuple(matrix.shape[:1]) != (len(samples),):
            raise ValueError(f"pdl features must have shape [{len(samples)}, D]")
        features = [row.reshape(-1) for row in matrix]
    iter_samples = samples if supplied_features is None else ()
    for sample in iter_samples:
        value = _decode_input(sample.input)
        if not isinstance(value, torch.Tensor):
            import numpy as np
            value = torch.as_tensor(np.asarray(value))
        features.append(value.detach().to(dtype=torch.float64).reshape(-1))
    if not features:
        return clean.clone(), torch.empty((0, classes, classes), dtype=torch.float64)
    width = int(features[0].numel())
    if any(int(value.numel()) != width for value in features):
        raise ValueError("pdl inputs must have a consistent flattened feature shape")
    matrix = torch.stack(features)
    if not torch.isfinite(matrix).all():
        raise ValueError("pdl inputs must be finite")
    generator = torch.Generator().manual_seed(int(seed))
    rate_std = float(config.get("rate_std", 0.1))
    flip_rate = _sample_truncated_normal(len(samples), float(rate), rate_std, generator)
    weights = torch.randn((classes, width, classes), generator=generator, dtype=torch.float64)
    transitions = torch.zeros((len(samples), classes), dtype=torch.float64)
    for position, clean_class in enumerate(clean.tolist()):
        scores = matrix[position].reshape(1, -1).mm(weights[int(clean_class)]).squeeze(0)
        scores[int(clean_class)] = -float("inf")
        row = float(flip_rate[position]) * torch.softmax(scores, dim=0)
        row[int(clean_class)] += 1.0 - float(flip_rate[position])
        transitions[position] = row
    noisy = torch.stack([
        torch.multinomial(transitions[position].float(), 1, generator=generator).to(dtype=torch.long).squeeze(0)
        for position in range(len(samples))
    ])
    return noisy, transitions


def _noise_targets(samples: Sequence[ScratchSample], config: Mapping[str, Any], classes: int) -> tuple[list[int], ScratchNoiseManifest, dict[int, int]]:
    torch = _torch()
    name = str(config.get("name", "none")).lower()
    rate = float(config.get("rate", 0.0))
    seed = int(config.get("seed", 1))
    complete_clean = all(sample.clean_target is not None for sample in samples)
    clean = (torch.as_tensor([int(sample.clean_target) for sample in samples], dtype=torch.long)
             if complete_clean else None)
    observed = torch.as_tensor([sample.observed_target for sample in samples], dtype=torch.long)
    if name not in {"none", "clean"} and any(sample.clean_target is None for sample in samples) and name not in {"external", "external_torch"}:
        raise ValueError("Scratch noise requires complete clean_target supervision")
    external_clean = None
    if name in {"none", "clean"} or (rate <= 0 and name not in {"external", "external_torch"}):
        noisy = observed
    elif name == "symmetric":
        generator = torch.Generator().manual_seed(seed)
        mask = torch.rand(len(samples), generator=generator) < rate
        offsets = torch.randint(1, classes, (len(samples),), generator=generator)
        noisy = torch.where(mask, (clean + offsets) % classes, clean)
    elif name == "pairflip":
        generator = torch.Generator().manual_seed(seed)
        mask = torch.rand(len(samples), generator=generator) < rate
        noisy = torch.where(mask, (clean + 1) % classes, clean)
    elif name in {"binary_asymmetric_rcn", "asymmetric_rcn"}:
        generator = torch.Generator().manual_seed(seed)
        pos = float(config.get("rho_positive", rate)); neg = float(config.get("rho_negative", rate))
        probs = torch.where(clean == 1, torch.full_like(clean, pos, dtype=torch.float32), torch.full_like(clean, neg, dtype=torch.float32))
        noisy = torch.where(torch.rand(len(samples), generator=generator) < probs, 1 - clean, clean)
    elif name == "class_conditional":
        matrix = _resolve_transition_matrix(config.get("transition_matrix"), classes=classes, rate=rate)
        generator = torch.Generator().manual_seed(seed)
        rows = torch.as_tensor(matrix, dtype=torch.float32)
        if rows.shape != (classes, classes) or not torch.allclose(rows.sum(dim=1), torch.ones(classes)):
            raise ValueError("transition_matrix must be square and every row must sum to one")
        sampled = [int(torch.multinomial(rows[int(label)], 1, generator=generator)) for label in clean]
        noisy = torch.as_tensor(sampled, dtype=torch.long)
    elif name == "instance_dependent":
        noisy, per_sample_transition = _sample_instance_dependent(samples, clean, classes, rate, seed, config)
    elif name == "pdl":
        noisy, per_sample_transition = _sample_pdl(samples, clean, classes, rate, seed, config)
    elif name in {"external", "external_torch"}:
        path = _resolve_external_path(config)
        payload = _load_external_payload(path)
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            declared_dataset = metadata.get("dataset")
            declared_split = metadata.get("split")
            expected_dataset = str(config.get("_dataset", ""))
            if declared_dataset and expected_dataset and str(declared_dataset) not in {expected_dataset, "cifar10", "cifar100", "mnist", "fashion_mnist"}:
                raise ValueError("external noise dataset metadata does not match Scratch source")
            if declared_split and str(declared_split) not in {"train", "training", "train_split"}:
                raise ValueError("external noise artifact must describe the train split")
        values = payload.get(str(config.get("noisy_key", "noise_label_train")), payload.get("noisy_targets"))
        external_clean = payload.get(str(config.get("clean_key", "clean_label_train")), payload.get("clean_targets"))
        if values is None or external_clean is None:
            raise ValueError("external noise artifact must contain clean and noisy targets")
        external_clean = torch.as_tensor(external_clean, dtype=torch.long).reshape(-1)
        external_noisy = torch.as_tensor(values, dtype=torch.long).reshape(-1)
        sample_indices = torch.as_tensor([sample.index for sample in samples], dtype=torch.long)
        external_indices = payload.get("indices", payload.get("global_indices", payload.get("sample_indices")))
        if external_indices is not None:
            ext_index_tensor = torch.as_tensor(external_indices, dtype=torch.long).reshape(-1)
            if len(set(ext_index_tensor.tolist())) != len(ext_index_tensor):
                raise ValueError("external noise artifact indices must be unique")
            position = {int(index): offset for offset, index in enumerate(ext_index_tensor.tolist())}
            try:
                offsets = [position[int(index)] for index in sample_indices.tolist()]
            except KeyError as exc:
                raise ValueError("external noise artifact is missing a Scratch sample index") from exc
            external_clean = external_clean[offsets]
            external_noisy = external_noisy[offsets]
        elif len(external_clean) != len(samples) or len(external_noisy) != len(samples):
            # Artifacts produced for the unsplit training source may omit an
            # explicit index vector; in that case the stable source index is
            # the only safe alignment key.
            if len(external_clean) <= int(sample_indices.max().item()) or len(external_noisy) <= int(sample_indices.max().item()):
                raise ValueError("external noise targets do not align with the Scratch train split")
            external_clean = external_clean[sample_indices]
            external_noisy = external_noisy[sample_indices]
        if clean is not None and not torch.equal(external_clean, clean):
            raise ValueError("external clean labels do not match Scratch source")
        if torch.any(external_clean < 0) or torch.any(external_clean >= classes) or torch.any(external_noisy < 0) or torch.any(external_noisy >= classes):
            raise ValueError("external noise labels are outside the declared class range")
        clean = external_clean
        noisy = external_noisy
    else:
        raise ValueError(f"unsupported Scratch noise type `{name}`")
    transition = None
    per_sample_transition = locals().get("per_sample_transition")
    if name == "class_conditional":
        transition = _resolve_transition_matrix(config.get("transition_matrix"), classes=classes, rate=rate)
    elif name in {"binary_asymmetric_rcn", "asymmetric_rcn"}:
        positive = float(config.get("rho_positive", rate))
        negative = float(config.get("rho_negative", rate))
        transition = [[1.0 - negative, negative], [positive, 1.0 - positive]]
    elif name in {"external", "external_torch"} and "payload" in locals() and payload.get("transition_matrix") is not None:
        transition = payload.get("transition_matrix")
    manifest = ScratchNoiseManifest(
        "train", torch.as_tensor([sample.index for sample in samples]), clean, noisy,
        name, seed, rate, transition, str(config.get("_dataset", "")),
        per_sample_transition=per_sample_transition,
        metadata={"generator": "scratch_instance_dependent" if name == "instance_dependent" else "scratch_pdl"}
        if name in {"instance_dependent", "pdl"} else {},
    )
    clean_by_index = {} if clean is None else {
        int(sample.index): int(value) for sample, value in zip(samples, clean)
    }
    return [int(value) for value in noisy], manifest, clean_by_index


def _transform_pair(preprocessing: Mapping[str, Any], train: ScratchSplit,
                    views: Sequence[str],
                    preprocessing_transform: Callable[[Any], Any] | None = None) -> tuple[Callable[[Any], Any] | None, dict[str, Callable[[Any], Any] | None]]:
    preprocessing = dict(preprocessing)
    name = str(preprocessing.get("name", "standard")).lower()
    augment = bool(preprocessing.get("augment", False))
    views = [str(value) for value in views]
    try:
        from torchvision import transforms
    except ImportError:
        return None, {view: None for view in views}
    sample_value = train.samples[0].input if train.samples else None
    shape = getattr(sample_value, "shape", ())
    image_like = len(shape) in {2, 3} and (len(shape) == 2 or int(shape[0]) in {1, 3} or int(shape[-1]) in {1, 3})
    if not image_like:
        return None, {view: None for view in views}

    height = int(shape[-2]) if len(shape) >= 2 else 32
    width = int(shape[-1]) if len(shape) >= 2 else 32
    crop_size = max(1, min(32, height, width))
    padding = min(4, max(0, crop_size // 4))
    ops: list[Any] = [transforms.Lambda(_to_tensor_input)]
    if augment and name not in {"tensor_only", "binary_raw"}:
        ops = [transforms.RandomCrop(crop_size, padding=padding), transforms.RandomHorizontalFlip(), transforms.Lambda(_to_tensor_input)]
    if name == "standard" and image_like:
        ops.append(transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)))
    elif name == "gce2018":
        pass
    elif name in {"tensor_only", "binary_raw", "official_cifar10"}:
        pass
    elif name:
        raise ValueError(f"unsupported Scratch preprocessing `{name}`")
    weak = preprocessing_transform if preprocessing_transform is not None else transforms.Compose(ops)
    result = {"weak": weak if "weak" in views else None}
    if "strong" in views:
        result["strong"] = transforms.Compose((transforms.RandomCrop(crop_size, padding=padding), transforms.RandomHorizontalFlip(), transforms.Lambda(_to_uint8_image), transforms.RandAugment(), transforms.Lambda(_to_tensor_input)))
    return weak, result


def apply_noise_to_split(split: ScratchSplit, config: Mapping[str, Any]) -> tuple[ScratchSplit, ScratchNoiseManifest, dict[int, int]]:
    """Apply one noise policy to a concrete split."""
    noise_config = dict(config)
    noise_config["_dataset"] = split.dataset
    noisy_values, manifest, clean_by_index = _noise_targets(split.samples, noise_config, split.num_classes)
    noisy_by_index = {sample.index: value for sample, value in zip(split.samples, noisy_values)}
    noisy = ScratchSplit(
        split.dataset,
        split.split,
        tuple(ScratchSample(sample.input, sample.index, noisy_by_index[sample.index],
                            clean_by_index.get(sample.index, sample.clean_target))
              for sample in split.samples),
        split.num_classes,
        split.version,
    )
    return noisy, manifest, clean_by_index


def build_transforms(preprocessing: Mapping[str, Any], source: ScratchSplit,
                     views: Sequence[str] = ("weak",),
                     preprocessing_transform: Callable[[Any], Any] | None = None) -> tuple[Callable[[Any], Any] | None, dict[str, Callable[[Any], Any] | None]]:
    """Build concrete input transforms for one preprocessing/views step."""
    return _transform_pair(preprocessing, source, views, preprocessing_transform)


def build_role_datasets(
    *,
    train_split: ScratchSplit,
    noisy_train: ScratchSplit,
    validation_split: ScratchSplit,
    test_split: ScratchSplit,
    roles: Sequence[str],
    data_config: Mapping[str, Any],
    preprocessing_transform: Callable[[Any], Any] | None,
    view_transforms: Mapping[str, Callable[[Any], Any] | None],
) -> dict[str, ScratchRoleDataset]:
    """Select concrete role datasets from already materialised splits."""
    weak = preprocessing_transform
    views = dict(view_transforms)
    trusted: ScratchSplit | None = None
    learning_train = noisy_train
    if "trusted_validation" in roles:
        count = int(data_config.get("num_clean", data_config.get("trusted_size", 0)))
        if count <= 0:
            raise ValueError("trusted_validation requires a positive num_clean/trusted_size")
        generator = _torch().Generator().manual_seed(int(data_config.get("trusted_seed", 1)))
        order = _torch().randperm(len(train_split.samples), generator=generator).tolist()
        trusted_positions = order[:min(count, len(order))]
        trusted = train_split.subset(trusted_positions, "trusted_validation")
        if not trusted.has_clean_targets:
            raise ValueError("trusted_validation requires complete clean_target supervision")
        trusted_indices = {sample.index for sample in trusted.samples}
        learning_train = ScratchSplit(
            noisy_train.dataset,
            noisy_train.split,
            tuple(sample for sample in noisy_train.samples if sample.index not in trusted_indices),
            noisy_train.num_classes,
            noisy_train.version,
        )
    datasets: dict[str, ScratchRoleDataset] = {}
    for role in roles:
        if role in {"clean_validation", "noisy_validation"} and not validation_split.samples:
            raise ValueError(f"{role} requires a source validation split or positive validation_size")
        if role == "clean_validation" and not validation_split.has_clean_targets:
            raise ValueError("clean_validation requires complete clean_target supervision")
        if role == "trusted_validation" and trusted is not None:
            datasets[role] = ScratchRoleDataset(trusted, transform=weak, views=views)
        elif role == "train":
            datasets[role] = ScratchRoleDataset(learning_train, transform=weak, views=views)
        elif role == "train_eval":
            datasets[role] = ScratchRoleDataset(learning_train, transform=weak, views=views)
        elif role in {"clean_validation", "noisy_validation"}:
            datasets[role] = ScratchRoleDataset(validation_split, transform=weak, views=views)
        elif role == "test":
            datasets[role] = ScratchRoleDataset(test_split, transform=weak, views=views)
        else:
            raise ValueError(f"unknown Scratch data role `{role}`")
    return datasets


def assemble_prepared(datasets: Mapping[str, ScratchRoleDataset], *, num_classes: int,
                      loader_config: Mapping[str, Any], plan: Mapping[str, Any],
                      manifest: ScratchNoiseManifest | None) -> "ScratchPrepared":
    """Assemble completed Block outputs; performs no data materialization."""
    return ScratchPrepared(datasets, num_classes=num_classes, loader_config=loader_config,
                           plan=plan, manifest=manifest)


class ScratchPrepared:
    def __init__(self, datasets: Mapping[str, ScratchRoleDataset], *, num_classes: int,
                 loader_config: Mapping[str, Any], plan: Mapping[str, Any], manifest: ScratchNoiseManifest | None) -> None:
        self.datasets = dict(datasets); self.num_classes = int(num_classes)
        self.dataset = str(plan.get("data", {}).get("name", "synthetic")); self.loader_config = dict(loader_config)
        self.plan = dict(plan); self.manifest = manifest; self.manifest_path = None
        self.train_indices = list(getattr(self.datasets.get("train"), "indices", []))
        self.validation_indices = list(getattr(self.datasets.get("noisy_validation", self.datasets.get("clean_validation")), "indices", []))

    def dataset_for(self, role: Any) -> ScratchRoleDataset:
        name = getattr(role, "value", role)
        if str(name) not in self.datasets:
            raise KeyError(f"role `{name}` is not configured")
        return self.datasets[str(name)]

    def loader(self, role: Any = "train", *, batch_size: int | None = None, shuffle: bool | None = None,
               drop_last: bool | None = None, **_: Any) -> Any:
        torch = _torch(); name = str(getattr(role, "value", role)); cfg = self.loader_config; dataset = self.dataset_for(name)
        use_shuffle = bool((name == "train") if shuffle is None else shuffle) and len(dataset) > 0
        return torch.utils.data.DataLoader(dataset, batch_size=int(batch_size or cfg.get("batch_size", 128)),
            shuffle=use_shuffle,
            drop_last=bool(cfg.get("drop_last", False) if drop_last is None else drop_last),
            num_workers=int(cfg.get("num_workers", 0)), pin_memory=bool(cfg.get("pin_memory", False)), collate_fn=collate_scratch_batch)


__all__ = [
    "ScratchBatch", "ScratchNoiseManifest", "ScratchPrepared", "ScratchRoleDataset",
    "ScratchSample", "ScratchSplit", "apply_noise_to_split", "assemble_prepared",
    "build_role_datasets", "build_transforms", "collate_scratch_batch", "load_sources",
    "split_source", "registered_dataset_catalog", "registered_dataset_config",
]
