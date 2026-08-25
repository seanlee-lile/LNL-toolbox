"""Scratch-native data contracts and materialisation pipeline.

This module deliberately depends only on Scratch code and optional runtime
packages.  The legacy data service is an oracle for tests, never a runtime
dependency of Scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
                 noise_type: str = "none", seed: int = 0, rate: float = 0.0) -> None:
        self.split = str(split)
        self.global_indices = indices
        self.clean_targets = clean_targets
        self.noisy_targets = observed_targets
        self.noise_type = str(noise_type)
        self.seed = int(seed)
        self.requested_rate = float(rate)
        payload = json.dumps({"split": self.split, "indices": _tolist(indices),
                              "clean": _tolist(clean_targets), "observed": _tolist(observed_targets)},
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


def _normalise_split(value: Any, *, dataset: str, split: str, classes: int) -> ScratchSplit:
    if isinstance(value, ScratchSplit):
        return value
    samples: list[ScratchSample] = []
    for position in range(len(value)):
        raw = value[position]
        observed, clean, index = _item_targets(raw)
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


def _load_source(plan: Mapping[str, Any], ctx: Mapping[str, Any]) -> tuple[ScratchSplit, ScratchSplit | None, ScratchSplit | None]:
    data = dict(plan.get("data", {}))
    name = str(data.get("name", "synthetic")).lower()
    classes = int(data.get("num_classes") or plan.get("semantics", {}).get("num_classes") or {"cifar10": 10, "cifar100": 100, "mnist": 10, "fashion_mnist": 10}.get(name, 2))
    if data.get("binary_classes") and name in {"cifar10", "cifar100"}:
        classes = 100 if name == "cifar100" else 10
    source = data.get("source")
    if source is not None:
        train = _normalise_split(_source_split(source, "train"), dataset=name, split="train", classes=classes)
        test = _normalise_split(_source_split(source, "test"), dataset=name, split="test", classes=classes)
        validation = None
        try:
            validation = _normalise_split(_source_split(source, "validation"), dataset=name, split="validation", classes=classes)
        except ValueError:
            pass
        requested = data.get("binary_classes")
        return _restrict_binary(train, requested), (_restrict_binary(validation, requested) if validation is not None else None), _restrict_binary(test, requested)
    if name in {"synthetic", "synthetic_classification", "synthetic_binary_2d"}:
        values = _synthetic_source(data, int(plan.get("seed", ctx.get("seed", 1))))
        requested = data.get("binary_classes")
        return _restrict_binary(values["train"], requested), None, _restrict_binary(values["test"], requested)
    if name in {"cifar10", "cifar100", "mnist", "fashion_mnist"}:
        try:
            from torchvision import datasets, transforms
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Scratch vision data requires torchvision") from exc
        root = str(data.get("root") or data.get("path") or "data")
        cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100,
               "mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}[name]
        kwargs = {"root": root, "download": bool(data.get("download", False)), "transform": transforms.ToTensor()}
        train = _normalise_split(cls(train=True, **kwargs), dataset=name, split="train", classes=classes)
        test = _normalise_split(cls(train=False, **kwargs), dataset=name, split="test", classes=classes)
        requested = data.get("binary_classes")
        return _restrict_binary(train, requested), None, _restrict_binary(test, requested)
    raise ValueError(f"Scratch cannot materialize dataset `{name}` without a train/test source")


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
        matrix = config.get("transition_matrix")
        if not isinstance(matrix, Sequence):
            matrix = [[1.0 if i == j else 0.0 for j in range(classes)] for i in range(classes)]
            for i in range(classes):
                matrix[i][i] = 1.0 - rate
                matrix[i][(i + 1) % classes] = rate
        generator = torch.Generator().manual_seed(seed)
        rows = torch.as_tensor(matrix, dtype=torch.float32)
        sampled = [int(torch.multinomial(rows[int(label)], 1, generator=generator)) for label in clean]
        noisy = torch.as_tensor(sampled, dtype=torch.long)
    elif name in {"instance_dependent", "pdl"}:
        generator = torch.Generator().manual_seed(seed)
        scores = torch.rand(len(samples), generator=generator)
        noisy = torch.where(scores < rate, (clean + 1 + torch.arange(len(samples)) % max(classes - 1, 1)) % classes, clean)
    elif name in {"external", "external_torch"}:
        path = config.get("path") or config.get("external_path") or config.get("artifact_path")
        if not path:
            raise ValueError("external noise requires an artifact path")
        payload = torch.load(Path(str(path)), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError("external noise artifact must contain a mapping")
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
        clean = external_clean
        noisy = external_noisy
    else:
        raise ValueError(f"unsupported Scratch noise type `{name}`")
    manifest = ScratchNoiseManifest("train", torch.as_tensor([sample.index for sample in samples]), clean, noisy, name, seed, rate)
    clean_by_index = {} if clean is None else {
        int(sample.index): int(value) for sample, value in zip(samples, clean)
    }
    return [int(value) for value in noisy], manifest, clean_by_index


def _transform_pair(plan: Mapping[str, Any], train: ScratchSplit) -> tuple[Callable[[Any], Any] | None, dict[str, Callable[[Any], Any] | None]]:
    data = dict(plan.get("data", {})); preprocessing = dict(plan.get("preprocessing", {}))
    name = str(preprocessing.get("name", "standard")).lower()
    augment = bool(preprocessing.get("augment", False))
    views = [str(value) for value in plan.get("views", ["weak"])]
    try:
        from torchvision import transforms
    except ImportError:
        return None, {view: None for view in views}
    sample_value = train.samples[0].input if train.samples else None
    shape = getattr(sample_value, "shape", ())
    image_like = len(shape) in {2, 3} and (len(shape) == 2 or int(shape[0]) in {1, 3} or int(shape[-1]) in {1, 3})
    if not image_like:
        return None, {view: None for view in views}

    ops: list[Any] = [transforms.Lambda(_to_tensor_input)]
    if augment and name not in {"tensor_only", "binary_raw"}:
        ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Lambda(_to_tensor_input)]
    if name == "standard" and image_like:
        ops.append(transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)))
    elif name == "gce2018":
        pass
    elif name in {"tensor_only", "binary_raw", "official_cifar10"}:
        pass
    elif name:
        raise ValueError(f"unsupported Scratch preprocessing `{name}`")
    weak = transforms.Compose(ops)
    result = {"weak": weak if "weak" in views else None}
    if "strong" in views:
        result["strong"] = transforms.Compose((transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Lambda(_to_uint8_image), transforms.RandAugment(), transforms.Lambda(_to_tensor_input)))
    return weak, result


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
        torch = _torch(); name = str(getattr(role, "value", role)); cfg = self.loader_config
        return torch.utils.data.DataLoader(self.dataset_for(name), batch_size=int(batch_size or cfg.get("batch_size", 128)),
            shuffle=bool((name == "train") if shuffle is None else shuffle),
            drop_last=bool(cfg.get("drop_last", False) if drop_last is None else drop_last),
            num_workers=int(cfg.get("num_workers", 0)), pin_memory=bool(cfg.get("pin_memory", False)), collate_fn=collate_scratch_batch)


def materialize_plan(plan: Mapping[str, Any], ctx: Mapping[str, Any]) -> tuple[ScratchPrepared, ScratchNoiseManifest | None]:
    runtime_limits = ctx.get("_runtime_limits", {}) if isinstance(ctx, Mapping) else {}
    if isinstance(runtime_limits, Mapping) and bool(runtime_limits.get("fixture")) and plan.get("data", {}).get("source") is None:
        fixture_plan = dict(plan)
        fixture_data = dict(plan.get("data", {}))
        fixture_data.update({"name": "synthetic", "train_size": 8, "test_size": 4,
                             "features": 4, "classes": int(plan.get("semantics", {}).get("num_classes", 10))})
        original_name = str(plan.get("data", {}).get("name", "")).lower()
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
        fixture_noise = dict(plan.get("noise", {}))
        if str(fixture_noise.get("name", "none")).lower() in {"external", "external_torch"}:
            # The fixture source is synthetic, so a paper artifact's labels
            # cannot be aligned to it. The real (non-fixture) path still
            # requires and validates the declared external manifest.
            fixture_noise = {"name": "symmetric", "rate": float(fixture_noise.get("rate", 0.0)),
                             "seed": int(fixture_noise.get("seed", 1))}
        fixture_plan["noise"] = fixture_noise
        train_source, official_validation, test_source = _load_source(fixture_plan, ctx)
    else:
        fixture_plan = plan
        train_source, official_validation, test_source = _load_source(plan, ctx)
    train, validation = _split_train(train_source, fixture_plan)
    if official_validation is not None and str(fixture_plan.get("split", {}).get("strategy", "")).lower() == "official":
        validation = official_validation
    labels = dict(fixture_plan.get("labels", {})); roles = list(fixture_plan.get("roles", ["train", "test"]))
    noise_config = dict(fixture_plan.get("noise", {}))
    trusted = None
    learning_train = train
    if "trusted_validation" in roles:
        external_clean_by_index: dict[int, int] = {}
        if not train.has_clean_targets and str(noise_config.get("name", "none")).lower() in {"external", "external_torch"}:
            _, _, external_clean_by_index = _noise_targets(train.samples, noise_config, train.num_classes)
            train = ScratchSplit(train.dataset, train.split, tuple(
                ScratchSample(sample.input, sample.index, sample.observed_target,
                              external_clean_by_index.get(sample.index)) for sample in train.samples),
                train.num_classes, train.version)
        count = int(fixture_plan.get("data", {}).get("num_clean", fixture_plan.get("data", {}).get("trusted_size", 0)))
        if count <= 0:
            raise ValueError("trusted_validation requires a positive num_clean/trusted_size")
        generator = _torch().Generator().manual_seed(int(fixture_plan.get("data", {}).get("trusted_seed", fixture_plan.get("seed", 1))))
        order = _torch().randperm(len(train.samples), generator=generator).tolist()
        trusted_positions = order[:min(count, len(order))]
        trusted = train.subset(trusted_positions, "trusted_validation")
        if not trusted.has_clean_targets:
            raise ValueError("trusted_validation requires complete clean_target supervision")
        trusted_indices = {sample.index for sample in trusted.samples}
        learning_train = ScratchSplit(train.dataset, train.split,
            tuple(sample for sample in train.samples if sample.index not in trusted_indices),
            train.num_classes, train.version)
    noisy_values, manifest, clean_by_index = _noise_targets(learning_train.samples, noise_config, learning_train.num_classes)
    noisy_by_index = {sample.index: value for sample, value in zip(learning_train.samples, noisy_values)}
    noisy_train = ScratchSplit(learning_train.dataset, learning_train.split,
        tuple(ScratchSample(s.input, s.index, noisy_by_index[s.index], clean_by_index.get(s.index, s.clean_target)) for s in learning_train.samples), learning_train.num_classes, learning_train.version)
    if labels.get("validation") == "observed" and validation.samples:
        val_values, _, val_clean_by_index = _noise_targets(validation.samples, noise_config, validation.num_classes)
        validation = ScratchSplit(validation.dataset, validation.split,
            tuple(ScratchSample(s.input, s.index, value, val_clean_by_index.get(s.index, s.clean_target)) for s, value in zip(validation.samples, val_values)), validation.num_classes, validation.version)
    if not validation.samples and any(role in roles for role in ("clean_validation", "noisy_validation")):
        validation = test_source
    weak, view_transforms = _transform_pair(fixture_plan, train_source)
    datasets: dict[str, ScratchRoleDataset] = {}
    for role in roles:
        if role == "train":
            datasets[role] = ScratchRoleDataset(noisy_train, transform=weak, views=view_transforms)
        elif role == "train_eval":
            datasets[role] = ScratchRoleDataset(noisy_train, transform=weak, views=view_transforms)
        elif role == "noisy_validation":
            if not validation.samples: raise ValueError("noisy_validation requires a validation/test source")
            datasets[role] = ScratchRoleDataset(validation, transform=weak, views=view_transforms)
        elif role == "clean_validation":
            if not validation.has_clean_targets: raise ValueError("clean_validation requires complete clean_target supervision")
            datasets[role] = ScratchRoleDataset(validation, transform=weak, views=view_transforms)
        elif role == "trusted_validation":
            datasets[role] = ScratchRoleDataset(trusted, transform=weak, views=view_transforms)
        elif role == "test":
            datasets[role] = ScratchRoleDataset(test_source, transform=weak, views=view_transforms)
        else:
            raise ValueError(f"unknown Scratch data role `{role}`")
    return ScratchPrepared(datasets, num_classes=train.num_classes, loader_config=fixture_plan.get("loader", {}), plan=fixture_plan, manifest=manifest), manifest


__all__ = ["ScratchBatch", "ScratchNoiseManifest", "ScratchPrepared", "ScratchRoleDataset", "ScratchSample", "ScratchSplit", "collate_scratch_batch", "materialize_plan"]
