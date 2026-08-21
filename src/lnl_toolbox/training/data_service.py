from __future__ import annotations

"""Single data-preparation entry point shared by every experiment runner."""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from lnl_toolbox.data.cifar_n import add_cifar_n_sources
from lnl_toolbox.data.contracts import (
    DataRequirements,
    DataRole,
    DataSpec,
    RawDatasetSplit,
)
from lnl_toolbox.data.mnist import add_mnist_sources
from lnl_toolbox.data.local_catalog import LocalDatasetCatalog, LocalDatasetRecord
from lnl_toolbox.data.real_noise import add_real_noise_sources
from lnl_toolbox.data.registry import DatasetRegistry
from lnl_toolbox.data.sources import add_existing_sources
from lnl_toolbox.data.torch_cifar import (
    build_cifar_transform,
    cifar_pixel_mean,
    train_validation_split,
)
from lnl_toolbox.data.views import IndexedDatasetView
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.noise.binary_rcn import generate_binary_asymmetric_rcn
from lnl_toolbox.training.noisy_labels import prepare_noise_manifest


def create_dataset_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    add_existing_sources(registry)
    add_cifar_n_sources(registry)
    add_mnist_sources(registry)
    add_real_noise_sources(registry)
    return registry


DATASETS = create_dataset_registry()


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _subset(indices: np.ndarray, labels: np.ndarray, maximum: Any, seed: int) -> np.ndarray:
    if maximum is None or int(maximum) >= len(indices):
        return indices.astype(np.int64, copy=True)
    maximum = int(maximum)
    if maximum <= 0:
        raise ValueError("dataset subset size must be positive")
    random = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    aligned = labels if len(labels) == len(indices) else labels[indices]
    unique = np.unique(aligned)
    base, remainder = divmod(maximum, len(unique))
    for offset, label in enumerate(unique):
        candidates = indices[aligned == label].copy()
        random.shuffle(candidates)
        selected.append(candidates[: base + (offset < remainder)])
    result = np.sort(np.concatenate(selected))
    if result.size != maximum:
        remaining = np.setdiff1d(indices, result, assume_unique=False)
        random.shuffle(remaining)
        result = np.sort(np.concatenate((result, remaining[: maximum - result.size])))
    return result.astype(np.int64, copy=False)


def _target_map(indices: np.ndarray, targets: np.ndarray) -> dict[int, int]:
    return {int(index): int(target) for index, target in zip(indices, targets)}


def _random_partition(indices: np.ndarray, sizes: Sequence[int], seed: int) -> list[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    np.random.RandomState(seed).shuffle(order)
    result: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        result.append(order[offset:offset + int(size)].copy())
        offset += int(size)
    if offset != order.size:
        raise ValueError("partition sizes do not cover the source indices")
    return result


def _is_image_split(split: RawDatasetSplit) -> bool:
    if len(split) == 0:
        return False
    value = split.inputs[0]
    return isinstance(value, (str, Path, Image.Image)) or (
        isinstance(value, np.ndarray) and value.ndim in {2, 3}
    )


def _transforms(
    split: RawDatasetSplit,
    data: Mapping[str, Any],
    requirements: DataRequirements,
    *,
    training: bool,
) -> dict[str, Any]:
    if not _is_image_split(split):
        return {name: None for name in requirements.views}
    name = split.dataset.replace("_", "").lower()
    normalization = dict(data.get("normalization", {}) or {})
    if name.startswith("cifar"):
        preprocessing = str(data.get("preprocessing", "standard"))
        pixel_mean = None
        if preprocessing == "gce2018":
            pixel_mean = cifar_pixel_mean(np.asarray(split.inputs))
        if preprocessing == "l2rw" or {"num_val", "num_clean", "input_seed"} <= set(data):
            from torchvision import transforms
            operations: list[Any] = []
            if training and bool(data.get("augment", True)):
                operations.extend((transforms.Pad(4), transforms.RandomCrop(32), transforms.RandomHorizontalFlip()))
            operations.extend((transforms.ToTensor(), transforms.Lambda(lambda value: (value - 0.5) * 2.0)))
            weak = transforms.Compose(operations)
        else:
            weak = build_cifar_transform(
                training,
                bool(data.get("augment", True)) if training else False,
                preprocessing=preprocessing,
                pixel_mean=pixel_mean,
                normalization_mean=normalization.get("mean", data.get("normalization_mean")),
                normalization_std=normalization.get("std", data.get("normalization_std")),
            )
        result = {view: weak for view in requirements.views}
        if "strong" in result:
            from lnl_toolbox.data.multi_view import build_strong_cifar_transform

            strong_options: dict[str, Any] = {
                "policy": str(data.get("strong_policy", "official_cifar10")),
                "magnitude": int(data.get("strong_magnitude", 10)),
            }
            if normalization.get("mean") is not None:
                strong_options["mean"] = normalization["mean"]
                strong_options["std"] = normalization["std"]
            result["strong"] = build_strong_cifar_transform(**strong_options)
        return result
    from torchvision import transforms

    if name in {"mnist", "fashionmnist"}:
        operations: list[Any] = []
        if training and bool(data.get("augment", False)):
            operations.append(transforms.RandomCrop(28, padding=4))
        operations.extend((transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))))
        base = transforms.Compose(operations)
        return {view: base for view in requirements.views}
    image_size = int(data.get("image_size", 224))
    mean = tuple(normalization.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(normalization.get("std", (0.229, 0.224, 0.225)))
    weak_ops: list[Any] = (
        [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
        if training
        else [transforms.Resize(image_size + 32), transforms.CenterCrop(image_size)]
    )
    weak_ops.extend((transforms.ToTensor(), transforms.Normalize(mean, std)))
    weak = transforms.Compose(weak_ops)
    result = {view: weak for view in requirements.views}
    if "strong" in result:
        result["strong"] = transforms.Compose((
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ))
    return result


@dataclass(slots=True)
class PreparedData:
    spec: DataSpec
    requirements: DataRequirements
    train_split: RawDatasetSplit
    test_split: RawDatasetSplit
    train_indices: np.ndarray
    validation_indices: np.ndarray
    trusted_indices: np.ndarray
    manifest: NoiseManifest | None
    manifest_path: Path | None
    datasets: dict[DataRole, Dataset]
    loader_config: Mapping[str, Any]
    seed: int
    data_manifest_path: Path
    data_fingerprint: str

    @property
    def num_classes(self) -> int:
        return self.train_split.num_classes

    @property
    def dataset(self) -> str:
        return self.train_split.dataset

    @property
    def noisy_targets(self) -> np.ndarray:
        if self.manifest is None:
            lookup = _target_map(self.train_split.global_indices, self.train_split.observed_targets)
        else:
            lookup = _target_map(self.manifest.global_indices, self.manifest.noisy_targets)
        return np.asarray([lookup[int(index)] for index in self.train_indices], dtype=np.int64)

    def dataset_for(self, role: DataRole | str) -> Dataset:
        key = role if isinstance(role, DataRole) else DataRole(str(role))
        try:
            return self.datasets[key]
        except KeyError as exc:
            raise KeyError(f"data role {key.value!r} was not requested") from exc

    def loader(
        self,
        role: DataRole | str,
        *,
        epoch: int = 0,
        stream: int = 0,
        shuffle: bool | None = None,
        drop_last: bool | None = None,
        batch_size: int | None = None,
        generator_seed: int | None = None,
    ) -> DataLoader:
        key = role if isinstance(role, DataRole) else DataRole(str(role))
        training = key == DataRole.TRAIN
        if shuffle is None:
            shuffle = training
        if drop_last is None:
            configured = self.requirements.train_drop_last
            drop_last = bool(self.loader_config.get("drop_last", False) if configured is None else configured) if training else False
        generator = torch.Generator().manual_seed(
            int(generator_seed) if generator_seed is not None else int(self.seed) + int(epoch) + int(stream)
        )
        return DataLoader(
            self.dataset_for(key),
            batch_size=int(batch_size or self.loader_config.get("batch_size", 128)),
            shuffle=bool(shuffle),
            num_workers=int(self.loader_config.get("num_workers", 0)),
            pin_memory=bool(self.loader_config.get("pin_memory", False)),
            drop_last=bool(drop_last),
            persistent_workers=bool(self.loader_config.get("persistent_workers", False))
            and int(self.loader_config.get("num_workers", 0)) > 0,
            generator=generator,
            worker_init_fn=_seed_worker,
        )

    def dynamic_dataset(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        views: tuple[str, ...] | None = None,
        targets_by_index: Mapping[int, int] | None = None,
        overlays: Mapping[str, Mapping[int, Any]] | None = None,
        training: bool = True,
    ) -> Dataset:
        requested = self.requirements if views is None else DataRequirements(
            roles=frozenset({DataRole.TRAIN}),
            views=views,
            validation_targets=self.requirements.validation_targets,
            needs_noise_manifest=self.requirements.needs_noise_manifest,
            manifest_scope=self.requirements.manifest_scope,
        )
        data_config = {"name": self.spec.name, **dict(self.spec.options)}
        return IndexedDatasetView(
            self.train_split,
            indices,
            targets_by_index=targets_by_index,
            transforms=_transforms(self.train_split, data_config, requested, training=training),
            overlays=overlays,
        )

    def loader_for_dataset(
        self,
        dataset: Dataset,
        *,
        epoch: int = 0,
        stream: int = 0,
        shuffle: bool = True,
        drop_last: bool | None = None,
        batch_size: int | None = None,
        generator_seed: int | None = None,
    ) -> DataLoader:
        if drop_last is None:
            drop_last = bool(self.loader_config.get("drop_last", False)) if shuffle else False
        generator = torch.Generator().manual_seed(
            int(generator_seed) if generator_seed is not None else int(self.seed) + int(epoch) + int(stream)
        )
        return DataLoader(
            dataset,
            batch_size=int(batch_size or self.loader_config.get("batch_size", 128)),
            shuffle=bool(shuffle),
            num_workers=int(self.loader_config.get("num_workers", 0)),
            pin_memory=bool(self.loader_config.get("pin_memory", False)),
            drop_last=bool(drop_last),
            persistent_workers=bool(self.loader_config.get("persistent_workers", False))
            and int(self.loader_config.get("num_workers", 0)) > 0,
            generator=generator,
            worker_init_fn=_seed_worker,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"data_fingerprint": self.data_fingerprint}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("data_fingerprint") != self.data_fingerprint:
            raise ValueError("checkpoint data identity mismatch")


def _write_data_manifest(
    path: Path,
    spec: DataSpec,
    splits: Mapping[str, RawDatasetSplit],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    trusted_indices: np.ndarray,
    noise_manifest: NoiseManifest | None,
    loader_config: Mapping[str, Any],
    requirements: DataRequirements,
) -> str:
    def jsonable(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    payload = {
        "format_version": 1,
        "dataset": spec.name,
        "source": {"root": None if spec.root is None else str(spec.root), "path": None if spec.path is None else str(spec.path)},
        "options": jsonable(spec.options),
        "splits": {name: split.identity.to_dict() for name, split in splits.items()},
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "trusted_indices": trusted_indices.tolist(),
        "noise": None if noise_manifest is None else {
            "mapping_hash": noise_manifest.mapping_hash,
            "noise_type": noise_manifest.noise_type,
            "seed": noise_manifest.seed,
            "requested_rate": noise_manifest.requested_rate,
        },
        "loader": {
            "batch_size": int(loader_config.get("batch_size", 128)),
            "num_workers": int(loader_config.get("num_workers", 0)),
            "drop_last": bool(loader_config.get("drop_last", False)),
            "epoch_seeded": True,
        },
        "requirements": {
            "roles": sorted(role.value for role in requirements.roles),
            "views": list(requirements.views),
            "validation_targets": requirements.validation_targets,
            "manifest_scope": requirements.manifest_scope,
            "train_drop_last": requirements.train_drop_last,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    payload["fingerprint"] = fingerprint
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("run data_manifest.json does not match configured data")
    else:
        path.write_text(encoded, encoding="utf-8")
    return fingerprint


def _prepare_experiment_data(
    config: Mapping[str, Any],
    *,
    requirements: DataRequirements,
    run_dir: str | Path,
    seed: int,
    checkpoint_payload: Mapping[str, Any] | None = None,
    registry: DatasetRegistry | None = None,
) -> PreparedData:
    """Normalize, validate, split, corrupt, view, and identify experiment data."""

    data_config = dict(config["data"])
    effective_config = dict(config)
    noise_config = dict(config.get("noise", {}) or {})
    if "name" not in noise_config and "type" in noise_config:
        noise_config["name"] = noise_config["type"]
    if "name" not in noise_config and {"rho_positive", "rho_negative"} <= set(noise_config):
        noise_config["name"] = "binary_asymmetric_rcn"
    if noise_config:
        effective_config["noise"] = noise_config
    spec = DataSpec.from_mapping(data_config)
    registry = registry or DATASETS
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train = registry.load(spec, "train", seed=seed)
    if requirements.class_subset is not None:
        selected = np.flatnonzero(np.isin(train.observed_targets, requirements.class_subset))
        remap = {value: offset for offset, value in enumerate(requirements.class_subset)}
        train = RawDatasetSplit(
            np.asarray(train.inputs)[selected],
            np.asarray([remap[int(value)] for value in train.observed_targets[selected]], dtype=np.int64),
            train.global_indices[selected],
            train.dataset,
            train.split,
            len(remap),
            train.version,
            None if train.clean_targets is None else np.asarray([remap[int(value)] for value in train.clean_targets[selected]], dtype=np.int64),
            tuple(train.class_names[value] for value in requirements.class_subset) if train.class_names else (),
            train.source,
        )
    try:
        native_validation = registry.load(spec, "validation", seed=seed)
    except (ValueError, FileNotFoundError):
        native_validation = None
    test = registry.load(spec, "test", seed=seed)
    source_train_indices = train.global_indices.copy()
    if requirements.subset_before_split:
        source_train_indices = _subset(
            source_train_indices,
            train.clean_targets if train.clean_targets is not None else train.observed_targets,
            data_config.get("max_train_samples"),
            seed + 1,
        )

    if native_validation is not None and native_validation.dataset == train.dataset:
        full_train_indices = source_train_indices.copy()
        validation_indices = native_validation.global_indices.copy()
        validation_split = native_validation
    else:
        validation_size = int(
            requirements.validation_size
            if requirements.validation_size is not None
            else data_config.get("validation_size", data_config.get("num_val", 0))
        )
        if validation_size:
            if "num_val" in data_config and "num_clean" in data_config:
                full_train_indices, validation_indices = _random_partition(
                    source_train_indices,
                    [len(source_train_indices) - validation_size, validation_size],
                    int(data_config.get("seed", seed)),
                )
            elif requirements.split_strategy == "numpy_choice_complement":
                np.random.seed(int(seed))
                chosen = np.random.choice(
                    source_train_indices.size,
                    source_train_indices.size - validation_size,
                    replace=False,
                )
                full_train_indices = source_train_indices[chosen]
                validation_indices = np.delete(source_train_indices, chosen)
            else:
                split_config = dict(data_config.get("validation_split", {}) or {})
                source_lookup = {int(index): position for position, index in enumerate(train.global_indices)}
                source_positions = np.asarray([source_lookup[int(index)] for index in source_train_indices], dtype=np.int64)
                train_positions, validation_positions = train_validation_split(
                    (train.clean_targets if train.clean_targets is not None else train.observed_targets)[source_positions],
                    validation_size,
                    int(data_config.get("split_seed", seed)),
                    strategy=str(data_config.get("split_strategy", split_config.get("strategy", "stratified"))),
                    rng=str(data_config.get("split_rng", split_config.get("rng", "default_rng"))),
                )
                full_train_indices = source_train_indices[train_positions]
                validation_indices = source_train_indices[validation_positions]
            validation_split = train
        else:
            full_train_indices = source_train_indices.copy()
            validation_indices = test.global_indices.copy()
            validation_split = test
    train_indices = _subset(
        full_train_indices,
        train.clean_targets if train.clean_targets is not None else train.observed_targets,
        None if requirements.subset_before_split else data_config.get("max_train_samples"),
        seed + 11,
    )
    trusted_indices = np.empty(0, dtype=np.int64)
    trusted_split = train
    trusted_target_map: dict[int, int] | None = None
    if DataRole.TRUSTED_VALIDATION in requirements.roles:
        trusted_config = dict(config.get("trusted_validation", {}) or {})
        trusted_source = str(trusted_config.get("source", "")).lower()
        if trusted_source == "audited_manifest":
            from lnl_toolbox.data.trusted import TrustedSupervisionManifest
            trusted_manifest = TrustedSupervisionManifest.load(trusted_config["manifest"])
            trusted_indices = trusted_manifest.global_indices.copy()
            trusted_target_map = _target_map(trusted_manifest.global_indices, trusted_manifest.targets)
            validation_available = set(map(int, validation_indices))
            if set(map(int, trusted_indices)) <= validation_available:
                trusted_split = validation_split
            elif not set(map(int, trusted_indices)) <= set(map(int, train.global_indices)):
                raise ValueError("trusted manifest indices are outside configured data")
        elif trusted_source == "synthetic_fixture":
            trusted_indices = validation_indices.copy()
            trusted_split = validation_split
            trusted_values = validation_split.clean_targets if validation_split.clean_targets is not None else validation_split.observed_targets
            trusted_target_map = _target_map(validation_split.global_indices, trusted_values)
        else:
            trusted_size = int(data_config.get("num_clean", data_config.get("trusted_size", 0)))
            if trusted_size <= 0 or trusted_size >= train_indices.size:
                raise ValueError("trusted_validation requires 0 < data.num_clean/trusted_size < training size")
            train_indices, trusted_indices = _random_partition(
                train_indices,
                [train_indices.size - trusted_size, trusted_size],
                int(data_config.get("seed", seed)),
            )
            if not ("num_val" in data_config and "num_clean" in data_config):
                labels = train.clean_targets if train.clean_targets is not None else train.observed_targets
                positions = {int(index): position for position, index in enumerate(train.global_indices)}
                combined = np.concatenate((train_indices, trusted_indices))
                aligned = np.asarray([labels[positions[int(index)]] for index in combined], dtype=np.int64)
                remaining_positions, trusted_positions = train_validation_split(
                    aligned, trusted_size, int(data_config.get("trusted_seed", seed)), strategy="stratified"
                )
                train_indices = combined[remaining_positions]
                trusted_indices = combined[trusted_positions]
    validation_indices = _subset(
        validation_indices,
        validation_split.clean_targets if validation_split.clean_targets is not None else validation_split.observed_targets,
        data_config.get("max_validation_samples"),
        seed + 12,
    )
    test_indices = _subset(
        test.global_indices,
        test.clean_targets if test.clean_targets is not None else test.observed_targets,
        data_config.get("max_test_samples"),
        seed + 13,
    )

    manifest: NoiseManifest | None = None
    manifest_path: Path | None = None
    source_clean = train.clean_targets
    if source_clean is not None:
        observed_differs = not np.array_equal(train.observed_targets, source_clean)
        if observed_differs:
            manifest = NoiseManifest(
                train.dataset,
                "real_world",
                0,
                float(np.mean(train.observed_targets != source_clean)),
                source_clean,
                train.observed_targets,
                metadata={"source": train.source},
                global_indices=train.global_indices,
                num_classes=train.num_classes,
            )
            manifest_path = run_dir / "noise_manifest.npz"
            if not manifest_path.exists():
                manifest.save(manifest_path)
        elif requirements.needs_noise_manifest:
            manifest_indices = (
                train_indices
                if requirements.manifest_scope == "effective_train"
                else full_train_indices
            )
            clean_lookup = _target_map(train.global_indices, source_clean)
            if requirements.validation_targets == "noisy":
                validation_clean = (
                    validation_split.clean_targets
                    if validation_split.clean_targets is not None
                    else validation_split.observed_targets
                )
                clean_lookup.update(_target_map(validation_split.global_indices, validation_clean))
                manifest_indices = np.unique(np.concatenate((manifest_indices, validation_indices)))
            manifest_clean = np.asarray([clean_lookup[int(index)] for index in manifest_indices], dtype=np.int64)
            dataset_targets = np.zeros(int(manifest_indices.max()) + 1, dtype=np.int64)
            for index, target in clean_lookup.items():
                if index < dataset_targets.size:
                    dataset_targets[index] = target
            noise_name = str(noise_config.get("name", "clean")).lower()
            if noise_name == "pdl":
                from lnl_toolbox.noise import generate_pdl_idn
                source_lookup = {int(index): position for position, index in enumerate(train.global_indices)}
                positions = np.asarray([source_lookup[int(index)] for index in manifest_indices], dtype=np.int64)
                values = np.asarray(train.inputs)[positions]
                if values.ndim == 4 and values.shape[1:] == (32, 32, 3):
                    values = np.transpose(values, (0, 3, 1, 2))
                if values.ndim != 2 and values.ndim != 4:
                    raise ValueError("PDL inputs must be flattened or CIFAR HWC images")
                raw_inputs = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
                manifest = generate_pdl_idn(
                    raw_inputs, manifest_clean, train.num_classes,
                    float(noise_config["rate"]), int(noise_config.get("seed", seed)), train.dataset,
                )
                manifest.global_indices = manifest_indices.copy()
                manifest.split = "train"
                manifest.num_classes = train.num_classes
                manifest_path = run_dir / str(noise_config.get("manifest_filename", "noise_manifest.npz"))
                if manifest_path.is_file():
                    stored = NoiseManifest.load(manifest_path)
                    if stored.mapping_hash != manifest.mapping_hash:
                        raise ValueError("existing PDL noise manifest changed")
                    manifest = stored
                else:
                    manifest.save(manifest_path)
            elif noise_name == "official_uniform_flip":
                rate = float(noise_config["rate"])
                noise_seed = int(noise_config.get("seed", seed))
                count = int(manifest_clean.size * rate)
                random_state = np.random.RandomState(noise_seed + 1)
                replacements = np.floor(
                    random_state.uniform(0.0, train.num_classes - 1, [count])
                ).astype(np.int64)
                noisy = np.concatenate((replacements, manifest_clean[count:]))
                order = np.arange(manifest_clean.size, dtype=np.int64)
                random_state.shuffle(order)
                manifest_indices = manifest_indices[order]
                manifest_clean = manifest_clean[order]
                noisy = noisy[order]
                manifest = NoiseManifest(
                    train.dataset, noise_name, noise_seed, rate,
                    manifest_clean, noisy, global_indices=manifest_indices,
                    num_classes=train.num_classes,
                    metadata={"source": "official_uniform_flip"},
                )
                train_indices = manifest_indices.copy()
                manifest_path = run_dir / str(noise_config.get("manifest_filename", "noise_manifest.npz"))
                if manifest_path.is_file():
                    stored = NoiseManifest.load(manifest_path)
                    if stored.mapping_hash != manifest.mapping_hash:
                        raise ValueError("existing official uniform-flip manifest changed")
                    manifest = stored
                else:
                    manifest.save(manifest_path)
            elif noise_name in {"binary_asymmetric_rcn", "asymmetric_rcn"}:
                manifest = generate_binary_asymmetric_rcn(
                    manifest_clean,
                    manifest_indices,
                    rho_positive=float(noise_config["rho_positive"]),
                    rho_negative=float(noise_config["rho_negative"]),
                    seed=int(noise_config.get("seed", seed)),
                    dataset=train.dataset,
                )
                manifest_path = run_dir / str(noise_config.get("manifest_filename", "noise_manifest.npz"))
                if manifest_path.is_file():
                    stored = NoiseManifest.load(manifest_path)
                    if stored.mapping_hash != manifest.mapping_hash:
                        raise ValueError("existing binary noise manifest does not match configured data")
                    manifest = stored
                else:
                    manifest.save(manifest_path)
            elif noise_name == "external_torch":
                source = Path(str(noise_config.get("path", ""))).expanduser().resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"external_torch noise file does not exist: {source}")
                payload = torch.load(source, map_location="cpu", weights_only=True)
                if not isinstance(payload, Mapping):
                    raise TypeError("external_torch noise file must contain a mapping")
                external_clean = np.asarray(payload[str(noise_config.get("clean_key", "clean_label_train"))], dtype=np.int64)
                external_noisy = np.asarray(payload[str(noise_config.get("noisy_key", "noise_label_train"))], dtype=np.int64)
                if external_clean.shape != external_noisy.shape or external_clean.ndim != 1:
                    raise ValueError("external clean/noisy labels must be aligned vectors")
                if manifest_indices.size and external_clean.size > int(manifest_indices.max()):
                    selected_clean, selected_noisy = external_clean[manifest_indices], external_noisy[manifest_indices]
                elif external_clean.size == manifest_clean.size:
                    selected_clean, selected_noisy = external_clean, external_noisy
                else:
                    raise ValueError("external noise labels do not cover requested training indices")
                if not np.array_equal(selected_clean, manifest_clean):
                    raise ValueError("external noise clean labels do not match the dataset")
                manifest = NoiseManifest(
                    train.dataset,
                    "external_instance_dependent",
                    int(noise_config.get("seed", seed)),
                    float(np.mean(selected_clean != selected_noisy)),
                    selected_clean,
                    selected_noisy,
                    metadata={"source": str(source)},
                    global_indices=manifest_indices,
                    num_classes=train.num_classes,
                )
                manifest_path = run_dir / str(noise_config.get("manifest_filename", "noise_manifest.npz"))
                if not manifest_path.is_file():
                    manifest.save(manifest_path)
            else:
                manifest, manifest_path = prepare_noise_manifest(
                effective_config,
                dataset=train.dataset,
                clean_targets=manifest_clean,
                global_indices=manifest_indices,
                num_classes=train.num_classes,
                run_dir=run_dir,
                checkpoint_payload=checkpoint_payload,
                dataset_targets=dataset_targets,
                )
    noisy_map = (
        _target_map(train.global_indices, train.observed_targets)
        if manifest is None
        else _target_map(manifest.global_indices, manifest.noisy_targets)
    )
    clean_train_map = None if train.clean_targets is None else _target_map(train.global_indices, train.clean_targets)
    validation_target_map = (
        noisy_map
        if requirements.validation_targets == "noisy"
        else _target_map(
            validation_split.global_indices,
            validation_split.clean_targets if validation_split.clean_targets is not None else validation_split.observed_targets,
        )
    )
    train_transforms = _transforms(train, data_config, requirements, training=True)
    eval_requirements = DataRequirements(
        roles=requirements.roles,
        views=("weak",),
        validation_targets=requirements.validation_targets,
        needs_noise_manifest=requirements.needs_noise_manifest,
    )
    train_eval_transforms = _transforms(train, data_config, eval_requirements, training=False)
    validation_transforms = _transforms(validation_split, data_config, eval_requirements, training=False)
    test_transforms = _transforms(test, data_config, eval_requirements, training=False)
    datasets: dict[DataRole, Dataset] = {}
    if DataRole.TRAIN in requirements.roles:
        datasets[DataRole.TRAIN] = IndexedDatasetView(train, train_indices, targets_by_index=noisy_map, transforms=train_transforms)
    if DataRole.TRAIN_EVAL in requirements.roles:
        datasets[DataRole.TRAIN_EVAL] = IndexedDatasetView(train, train_indices, targets_by_index=noisy_map, transforms=train_eval_transforms)
    validation_view = IndexedDatasetView(validation_split, validation_indices, targets_by_index=validation_target_map, transforms=validation_transforms)
    if DataRole.NOISY_VALIDATION in requirements.roles:
        datasets[DataRole.NOISY_VALIDATION] = validation_view
    if DataRole.CLEAN_VALIDATION in requirements.roles:
        if validation_split is train and clean_train_map is not None:
            datasets[DataRole.CLEAN_VALIDATION] = IndexedDatasetView(train, validation_indices, targets_by_index=clean_train_map, transforms=validation_transforms)
        else:
            datasets[DataRole.CLEAN_VALIDATION] = validation_view
    if DataRole.TRUSTED_VALIDATION in requirements.roles:
        if trusted_target_map is None:
            trusted_values = trusted_split.clean_targets if trusted_split.clean_targets is not None else trusted_split.observed_targets
            trusted_target_map = _target_map(trusted_split.global_indices, trusted_values)
        if trusted_target_map is None:
            raise ValueError("trusted_validation requires clean or trusted targets")
        datasets[DataRole.TRUSTED_VALIDATION] = IndexedDatasetView(
            trusted_split, trusted_indices, targets_by_index=trusted_target_map,
            transforms=_transforms(
                trusted_split, data_config, eval_requirements,
                training=trusted_source == "official_generated",
            ),
        )
    if DataRole.TEST in requirements.roles:
        clean_test = test.clean_targets if test.clean_targets is not None else test.observed_targets
        datasets[DataRole.TEST] = IndexedDatasetView(test, test_indices, targets_by_index=_target_map(test.global_indices, clean_test), transforms=test_transforms)
    data_manifest_path = run_dir / "data_manifest.json"
    fingerprint = _write_data_manifest(
        data_manifest_path,
        spec,
        {"train": train, "validation": validation_split, "test": test},
        train_indices,
        validation_indices,
        trusted_indices,
        manifest,
        dict(config.get("loader", {})),
        requirements,
    )
    adapter = registry.get(spec.name)
    artifact_provider = getattr(adapter, "identity_artifacts", None)
    if callable(artifact_provider):
        for filename, artifact in artifact_provider(spec, seed=seed).items():
            artifact_path = run_dir / str(filename)
            encoded = json.dumps(artifact, indent=2, sort_keys=True)
            if artifact_path.is_file():
                if json.loads(artifact_path.read_text(encoding="utf-8")) != artifact:
                    raise ValueError(f"{filename} identity mismatch")
            else:
                artifact_path.write_text(encoded, encoding="utf-8")
    prepared = PreparedData(
        spec=spec,
        requirements=requirements,
        train_split=train,
        test_split=test,
        train_indices=train_indices,
        validation_indices=validation_indices,
        trusted_indices=trusted_indices,
        manifest=manifest,
        manifest_path=manifest_path,
        datasets=datasets,
        loader_config=dict(config.get("loader", {})),
        seed=int(seed),
        data_manifest_path=data_manifest_path,
        data_fingerprint=fingerprint,
    )
    if checkpoint_payload is not None and "data" in checkpoint_payload:
        prepared.load_state_dict(checkpoint_payload["data"])
    return prepared


def _validate_data_config(
    config: Mapping[str, Any],
    registry: DatasetRegistry | None = None,
) -> DataSpec:
    spec = DataSpec.from_mapping(config["data"])
    if spec.root is not None and not spec.root.exists():
        raise FileNotFoundError(f"data path does not exist: {spec.root}")
    if spec.path is not None and not spec.path.exists():
        raise FileNotFoundError(f"data path does not exist: {spec.path}")
    active = registry or DATASETS
    active.validate(spec)
    active.load(spec, "train", seed=int(config.get("seed", 0)))
    active.load(spec, "test", seed=int(config.get("seed", 0)))
    return spec


@dataclass(frozen=True, slots=True)
class DatasetStatusReport:
    """Stable user-facing status produced by the shared data service."""

    name: str
    adapter: str
    status: str
    location: str | None = None
    train_samples: int | None = None
    test_samples: int | None = None
    classes: int | None = None
    train_fingerprint: str | None = None
    test_fingerprint: str | None = None
    fingerprint: str | None = None
    training_evidence: Mapping[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter": self.adapter,
            "status": self.status,
            "location": self.location,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "classes": self.classes,
            "train_fingerprint": self.train_fingerprint,
            "test_fingerprint": self.test_fingerprint,
            "fingerprint": self.fingerprint,
            "training_evidence": (
                None if self.training_evidence is None else dict(self.training_evidence)
            ),
            "error": self.error,
        }


class DataService:
    """Public data-management and experiment-data facade.

    Dataset adapters remain task-neutral.  This facade owns local registration,
    readiness reporting, full train/test inspection, and the compatibility
    entry used by experiment runners.
    """

    def __init__(
        self,
        registry: DatasetRegistry | None = None,
        catalog: LocalDatasetCatalog | None = None,
    ) -> None:
        self.registry = registry or DATASETS
        self._catalog = catalog

    @property
    def catalog(self) -> LocalDatasetCatalog:
        # Resolve the default path at call time so tests, CLI sessions, and the
        # Web server all honor the current LNL_DATA_CATALOG environment.
        return self._catalog or LocalDatasetCatalog()

    @staticmethod
    def _location(record: LocalDatasetRecord) -> str | None:
        value = record.data.get("root") or record.data.get("path")
        return None if value in {None, ""} else str(value)

    def _record_for(self, name: object) -> LocalDatasetRecord | None:
        key = str(name).strip().lower().replace(" ", "-")
        try:
            return self.catalog.get(key)
        except KeyError:
            adapter = self.registry.get(name).name
            matches = tuple(
                record for record in self.catalog.records() if record.adapter == adapter
            )
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"multiple local registrations use adapter {adapter!r}; "
                    "select one by alias"
                )
            return None

    def _shallow_report(
        self,
        name: str,
        adapter: str,
        record: LocalDatasetRecord | None,
    ) -> DatasetStatusReport:
        if record is None:
            return DatasetStatusReport(name, adapter, "missing")
        location = self._location(record)
        exists = location is not None and Path(location).expanduser().exists()
        effective = record.effective_state
        if not exists:
            status = "missing"
        elif effective in {"layout_validated", "training_verified"}:
            status = "ready"
        else:
            status = "incomplete"
        evidence = dict(record.evidence or {})
        return DatasetStatusReport(
            name=record.alias,
            adapter=record.adapter,
            status=status,
            location=location,
            train_samples=evidence.get("train_samples"),
            test_samples=evidence.get("test_samples"),
            classes=evidence.get("classes"),
            train_fingerprint=evidence.get("train_fingerprint"),
            test_fingerprint=evidence.get("test_fingerprint"),
            fingerprint=evidence.get("fingerprint") or evidence.get("data_fingerprint"),
            training_evidence=evidence if effective == "training_verified" else None,
            error=record.error,
        )

    def list_datasets(self) -> tuple[DatasetStatusReport, ...]:
        records = self.catalog.records()
        reports = [
            self._shallow_report(record.alias, record.adapter, record)
            for record in records
        ]
        registered_adapters = {record.adapter for record in records}
        reports.extend(
            DatasetStatusReport(name, name, "missing")
            for name in self.registry.names()
            if not name.startswith("synthetic_") and name not in registered_adapters
        )
        return tuple(sorted(reports, key=lambda item: (item.adapter, item.name)))

    def status(self, name: object | None = None):
        if name is None:
            return self.list_datasets()
        record = self._record_for(name)
        adapter = record.adapter if record is not None else self.registry.get(name).name
        return self._shallow_report(str(name), adapter, record)

    def path(self, name: object) -> Path | None:
        report = self.status(name)
        return None if report.location is None else Path(report.location)

    def register(
        self,
        alias: object,
        adapter: object,
        data: Mapping[str, Any],
    ) -> DatasetStatusReport:
        canonical = self.registry.get(adapter).name
        if canonical.startswith("synthetic_"):
            raise ValueError("synthetic datasets do not need local registration")
        payload = dict(data)
        payload["name"] = canonical
        if canonical == "uci_binary":
            if payload.get("path") in {None, ""}:
                raise ValueError("uci_binary registration requires --path")
            payload.setdefault(
                "preprocessing",
                {
                    "format": "whitespace",
                    "target_column": -1,
                    "has_header": False,
                    "standardize": False,
                },
            )
            payload.setdefault(
                "split",
                {"validation_fraction": 0.2, "test_fraction": 0.2},
            )
        elif payload.get("root") in {None, ""}:
            raise ValueError(f"{canonical} registration requires --root")
        if canonical in {"cifar10n", "cifar100n"} and payload.get("noise_path") in {
            None,
            "",
        }:
            raise ValueError(f"{canonical} registration requires --labels")
        record = self.catalog.register(alias, canonical, payload)
        return self._shallow_report(record.alias, record.adapter, record)

    def remove(self, name: object) -> None:
        self.catalog.remove(name)

    def record(self, name: object) -> LocalDatasetRecord:
        return self.catalog.get(name)

    def apply(self, config: Mapping[str, Any], name: object) -> dict[str, Any]:
        return self.catalog.apply(config, name)

    def resolve_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve one portable data contract through a unique local registration."""

        resolved = deepcopy(dict(config))
        data = resolved.get("data")
        if not isinstance(data, Mapping):
            return resolved
        data_config = dict(data)
        name = str(data_config.get("name", "")).strip()
        if not name or name.startswith("synthetic_"):
            return resolved
        if data_config.get("root") or data_config.get("path"):
            return resolved
        record = self._record_for(name)
        if record is None:
            raise ValueError(
                f"dataset {name!r} has no local registration; run "
                f"'lnl data register <alias> --adapter {name} ...' or pass --data"
            )
        return self.catalog.apply(resolved, record.alias)

    @staticmethod
    def _inspection_report(
        name: str,
        adapter: str,
        location: str | None,
        train: RawDatasetSplit,
        test: RawDatasetSplit,
    ) -> DatasetStatusReport:
        train_fingerprint = train.identity.fingerprint
        test_fingerprint = test.identity.fingerprint
        fingerprint = hashlib.sha256(
            f"{train_fingerprint}:{test_fingerprint}".encode("utf-8")
        ).hexdigest()
        return DatasetStatusReport(
            name=name,
            adapter=adapter,
            status="ready",
            location=location,
            train_samples=len(train),
            test_samples=len(test),
            classes=max(train.num_classes, test.num_classes),
            train_fingerprint=train_fingerprint,
            test_fingerprint=test_fingerprint,
            fingerprint=fingerprint,
        )

    def inspect(self, source: object, *, seed: int = 0) -> DatasetStatusReport:
        record: LocalDatasetRecord | None = None
        if isinstance(source, Mapping):
            config = dict(source)
            spec = DataSpec.from_mapping(config["data"])
            name = spec.name
            adapter = self.registry.get(spec.name).name
            location_value = spec.root or spec.path
            location = None if location_value is None else str(location_value)
        else:
            record = self._record_for(source)
            if record is None:
                adapter = self.registry.get(source).name
                return DatasetStatusReport(str(source), adapter, "missing")
            spec = DataSpec.from_mapping(record.data)
            name = record.alias
            adapter = record.adapter
            location = self._location(record)
        try:
            self.registry.validate(spec)
            train = self.registry.load(spec, "train", seed=seed)
            test = self.registry.load(spec, "test", seed=seed)
            report = self._inspection_report(name, adapter, location, train, test)
            if record is not None:
                evidence = report.to_dict()
                evidence.pop("training_evidence", None)
                if record.effective_state == "training_verified":
                    combined = {**dict(record.evidence or {}), **evidence}
                    self.catalog.mark_training_verified(record.alias, combined)
                    report = replace(report, training_evidence=combined)
                else:
                    self.catalog.mark_layout_validated(record.alias, evidence)
            return report
        except Exception as exc:
            if record is not None:
                self.catalog.mark_failed(record.alias, exc)
            return DatasetStatusReport(
                name=name,
                adapter=adapter,
                status="incomplete",
                location=location,
                error=f"{type(exc).__name__}: {exc}",
            )

    def verification_config(
        self,
        name: object,
        report: DatasetStatusReport,
        *,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Build a one-epoch smoke config from the registered data contract.

        This profile proves that the adapter can enter a real training runner;
        it is deliberately independent of paper recipes and their datasets.
        """

        record = self.record(name)
        data = dict(record.data)
        if record.adapter == "uci_binary":
            return {
                "schema_version": 1,
                "kind": "experiment",
                "seed": int(seed),
                "data": data,
                "loader": {"batch_size": 64, "num_workers": 0},
                "model": {"name": "linear"},
                "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.9},
                "trainer": {"epochs": 1, "device": "auto"},
                "execution": {"runner": "binary"},
            }
        if report.train_samples is None or report.train_samples < 2:
            raise ValueError("automatic image verification requires at least 2 train samples")
        validation_size = min(5000, max(1, report.train_samples // 10))
        data.update({
            "validation_size": validation_size,
            "max_train_samples": min(512, report.train_samples - validation_size),
            "max_validation_samples": min(256, validation_size),
            "max_test_samples": min(256, report.test_samples or 256),
            "augment": False,
        })
        return {
            "schema_version": 1,
            "kind": "experiment",
            "seed": int(seed),
            "data": data,
            "loader": {
                "batch_size": 128,
                "num_workers": 0,
                "pin_memory": bool(torch.cuda.is_available()),
            },
            "model": {"name": "tiny_cnn", "width": 32},
            "loss": {"name": "ce"},
            "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0005},
            "scheduler": {"name": "none"},
            "trainer": {"epochs": 1, "device": "auto", "progress": True},
            "execution": {"runner": "clean"},
        }

    def verify(
        self,
        name: object,
        config: Mapping[str, Any] | None,
        output_dir: str | Path,
        *,
        recipe: str | None = None,
    ) -> tuple[DatasetStatusReport, Path]:
        seed = int((config or {}).get("seed", 0))
        report = self.inspect(name, seed=seed)
        if report.status != "ready":
            raise ValueError(report.error or f"dataset is not ready: {name}")
        resolved_config = (
            self.verification_config(name, report, seed=seed)
            if config is None
            else dict(config)
        )
        from lnl_toolbox.training.service import ExperimentService

        service = ExperimentService(data_service=self)
        try:
            service.preflight(resolved_config, check_data=True)
            run_dir = Path(
                service.run(resolved_config, output_dir, recipe=recipe)
            ).resolve()
            metrics_path = run_dir / "metrics.jsonl"
            epoch_rows: list[dict[str, Any]] = []
            if metrics_path.is_file():
                for line in metrics_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if isinstance(row, dict) and row.get("event") == "epoch":
                            epoch_rows.append(row)
            legacy_path = run_dir / "metrics.json"
            if not epoch_rows and legacy_path.is_file():
                legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
                if isinstance(legacy, list):
                    epoch_rows = [
                        row for row in legacy
                        if isinstance(row, dict) and "epoch" in row
                    ]
            if not epoch_rows:
                raise ValueError("verification run did not complete an epoch")
            manifest_path = run_dir / "data_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            evidence = {
                **report.to_dict(),
                "recipe": recipe,
                "verification_profile": "automatic" if config is None else "recipe",
                "run_dir": str(run_dir),
                "data_fingerprint": manifest["fingerprint"],
                "result_metrics": str(
                    run_dir / "final_metrics.json"
                    if (run_dir / "final_metrics.json").is_file()
                    else metrics_path if metrics_path.is_file() else legacy_path
                ),
                "completed_epochs": len(epoch_rows),
            }
            record = self._record_for(name)
            assert record is not None
            self.catalog.mark_training_verified(record.alias, evidence)
            return replace(report, training_evidence=evidence), run_dir
        except Exception as exc:
            record = self._record_for(name)
            if record is not None:
                self.catalog.mark_training_failed(record.alias, exc)
            raise

    def validate_config(self, config: Mapping[str, Any]) -> DataSpec:
        return _validate_data_config(self.resolve_config(config), self.registry)

    def prepare_experiment_data(self, *args, **kwargs) -> PreparedData:
        kwargs.setdefault("registry", self.registry)
        if args:
            args = (self.resolve_config(args[0]), *args[1:])
        elif "config" in kwargs:
            kwargs["config"] = self.resolve_config(kwargs["config"])
        return _prepare_experiment_data(*args, **kwargs)


DEFAULT_DATA_SERVICE = DataService()


def prepare_experiment_data(*args, **kwargs) -> PreparedData:
    """Compatibility entry used by all existing experiment runners."""

    return DEFAULT_DATA_SERVICE.prepare_experiment_data(*args, **kwargs)


def validate_data_config(
    config: Mapping[str, Any],
    registry: DatasetRegistry | None = None,
) -> DataSpec:
    """Compatibility validation entry used by existing callers."""

    if registry is None:
        return DEFAULT_DATA_SERVICE.validate_config(config)
    return _validate_data_config(config, registry)


__all__ = [
    "DATASETS",
    "DEFAULT_DATA_SERVICE",
    "DataService",
    "DatasetStatusReport",
    "PreparedData",
    "create_dataset_registry",
    "prepare_experiment_data",
    "validate_data_config",
]
