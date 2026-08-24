"""Small data blocks; datasets and loaders are kept in Context slots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..context import ScratchContext
from ..registry import block


class _ScratchFixturePrepared:
    """Contract-compatible tiny prepared-data object used only by runtime-limited checks."""
    def __init__(self, classes: int = 10, samples: int = 8) -> None:
        import torch
        self.num_classes = int(classes)
        self.train_indices = torch.arange(int(samples), dtype=torch.int64).numpy()
        images = torch.randn(int(samples), 3, 32, 32)
        labels = torch.arange(int(samples), dtype=torch.int64) % int(classes)
        indices = torch.arange(int(samples), dtype=torch.int64)
        self._dataset = torch.utils.data.TensorDataset(images, labels, indices)

    def loader(self, role: Any = "train", epoch: int = 0, batch_size: int = 128, **_: Any):
        import torch
        return torch.utils.data.DataLoader(self._dataset, batch_size=min(int(batch_size), len(self._dataset)), shuffle=False)

    def dataset_for(self, role: Any):
        return self._dataset


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Data blocks require PyTorch; install the `train` extra.") from exc
    return torch


def _batch_values(batch: Any) -> tuple[Any, Any, Any | None]:
    if isinstance(batch, dict):
        values = batch
        inputs = values.get("images", values.get("inputs", values.get("input", values.get("x"))))
        labels = values.get("labels", values.get("targets", values.get("target", values.get("y"))))
        indices = values.get("indices", values.get("index"))
        return inputs, labels, indices
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("a batch must contain inputs and labels")
    indices = batch[2] if len(batch) > 2 else None
    return batch[0], batch[1], indices


@block(
    id="select_dataset",
    name="Select Dataset",
    category="Data",
    description="Select a registered or built-in dataset for the recipe.",
    params={
        "source_mode": {"type": "enum", "options": ["registered", "builtin", "custom_path"], "default": "registered"},
        "dataset": {"type": "dataset", "default": ""},
        "path": {"type": "path", "default": ""},
        "save_as": {"type": "slot", "default": "train_dataset"},
    },
    provides=("save_as",),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def select_dataset(
    ctx: ScratchContext,
    source_mode: str = "registered",
    dataset: str = "",
    path: str = "",
    save_as: str = "train_dataset",
) -> None:
    """Resolve a dataset supplied by the catalog or a small built-in option.

    The WebUI normally supplies registered dataset objects through the context.
    Keeping the lookup here deliberately strict prevents a recipe from silently
    guessing what a filesystem path contains.
    """
    name = str(dataset).strip()
    if not name:
        raise ValueError("select_dataset needs a dataset selection")
    catalog = ctx.get("dataset_catalog", {})
    if isinstance(catalog, dict) and name in catalog:
        ctx[save_as] = catalog[name]
        return
    if str(source_mode) == "builtin" and name.lower() in {"synthetic", "synthetic_classification"}:
        load_synthetic(ctx, save_as=save_as)
        return
    if str(source_mode) == "custom_path":
        raise ValueError(f"custom dataset path is not available to Scratch yet: {path or name}")
    raise ValueError(f"dataset `{name}` is not present in the Scratch data catalog")


@block(
    id="configure_noise",
    name="Configure Label Noise",
    category="Data",
    description="Attach an explicit label-noise policy to the selected dataset.",
    params={
        "method": {"type": "enum", "options": ["none", "symmetric", "pairflip", "class_conditional", "instance_dependent", "external"], "default": "none"},
        "rate": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
        "seed": {"type": "int", "default": 1, "min": 0},
    },
    requires=("train_dataset",),
    provides=("noise_config",),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def configure_noise(
    ctx: ScratchContext,
    method: str = "none",
    rate: float = 0.0,
    seed: int = 1,
) -> None:
    ctx["noise_config"] = {"method": str(method), "rate": float(rate), "seed": int(seed)}


@block(
    id="get_batch",
    name="Get Batch",
    category="Data",
    description="Unpack the current batch into images, labels, and optional sample indices.",
    params={
        "batch": {"type": "slot", "default": "batch"},
        "input_as": {"type": "slot", "default": "images"},
        "label_as": {"type": "slot", "default": "labels"},
        "index_as": {"type": "slot", "default": "indices"},
    },
    requires=("batch",),
    provides=("input_as", "label_as", "index_as"),
    placement=("batch",), stage="data", ui_group="① 数据准备",
)
def get_batch(
    ctx: ScratchContext,
    batch: str = "batch",
    input_as: str = "images",
    label_as: str = "labels",
    index_as: str = "indices",
) -> None:
    inputs, labels, indices = _batch_values(ctx[batch])
    ctx[input_as] = inputs
    ctx[label_as] = labels
    if indices is not None:
        ctx[index_as] = indices


@block(
    id="move_batch_to_device",
    name="Move Batch to Device",
    category="Data",
    description="Move images and labels to the selected CPU/CUDA device before the forward pass.",
    params={
        "input": {"type": "slot", "default": "images"},
        "labels": {"type": "slot", "default": "labels"},
        "device": {"type": "slot", "default": "device"},
    },
    requires=("input", "labels", "device"),
    provides=("input", "labels"),
    placement=("batch",), stage="train", ui_group="① 数据准备",
)
def move_batch_to_device(
    ctx: ScratchContext,
    input: str = "images",
    labels: str = "labels",
    device: str = "device",
) -> None:
    target = ctx[device]
    ctx[input] = ctx[input].to(target)
    ctx[labels] = ctx[labels].to(target)


@block(
    id="load_synthetic",
    name="Load Synthetic Classification",
    category="Data",
    description="Create a deterministic tensor dataset for fast local smoke tests.",
    params={
        "samples": {"type": "int", "default": 64, "min": 1},
        "features": {"type": "int", "default": 4, "min": 1},
        "classes": {"type": "int", "default": 2, "min": 2},
        "save_as": {"type": "slot", "default": "train_dataset"},
    },
    provides=("save_as",),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def load_synthetic(
    ctx: ScratchContext,
    samples: int = 64,
    features: int = 4,
    classes: int = 2,
    save_as: str = "train_dataset",
) -> None:
    torch = _torch()
    generator = torch.Generator().manual_seed(int(ctx.get("seed", 1)))
    inputs = torch.randn(samples, features, generator=generator)
    weights = torch.randn(features, classes, generator=generator)
    labels = (inputs @ weights).argmax(dim=1).long()
    indices = torch.arange(samples)
    ctx[save_as] = torch.utils.data.TensorDataset(inputs, labels, indices)
    ctx["num_classes"] = int(classes)


def _load_torchvision_dataset(name: str, ctx: ScratchContext, params: dict[str, Any]) -> None:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Vision data blocks require torchvision; install the `train` extra.") from exc
    root = str(params["root"])
    train = bool(params["train"])
    dataset_class = getattr(datasets, name)
    dataset = dataset_class(root=root, train=train, download=bool(params["download"]), transform=transforms.ToTensor())
    ctx[str(params["save_as"])] = dataset
    ctx["num_classes"] = 100 if name == "CIFAR100" else 10


@block(
    id="load_cifar10",
    name="Load CIFAR-10",
    category="Data",
    description="Load one CIFAR-10 split with torchvision.",
    params={
        "root": {"type": "str", "default": "data"},
        "train": {"type": "bool", "default": True},
        "download": {"type": "bool", "default": False},
        "save_as": {"type": "slot", "default": "train_dataset"},
    },
    provides=("save_as", "num_classes"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def load_cifar10(ctx: ScratchContext, **params: Any) -> None:
    _load_torchvision_dataset("CIFAR10", ctx, params)


@block(
    id="prepare_gce_cifar10",
    name="Prepare GCE CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare the formal GCE CIFAR-10 split, GCE-2018 preprocessing, symmetric noise manifest, and loaders.",
    params={
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 5000, "min": 1},
        "augment": {"type": "bool", "default": True},
        "noise_method": {"type": "enum", "options": ["symmetric"], "default": "symmetric"},
        "noise_rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "validation_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_gce_cifar10(
    ctx: ScratchContext,
    root: str = "",
    validation_size: int = 5000,
    augment: bool = True,
    noise_method: str = "symmetric",
    noise_rate: float = 0.2,
    noise_seed: int = 1,
    batch_size: int = 128,
    save_as: str = "prepared_data",
) -> None:
    """Use the shared formal GCE data pipeline and its persisted noise manifest."""
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "data": {
            "name": "cifar10",
            "validation_size": int(validation_size),
            "augment": bool(augment),
            "preprocessing": "gce2018",
        },
        "noise": {
            "name": str(noise_method),
            "rate": float(noise_rate),
            "seed": int(noise_seed),
            "validation_targets": "noisy",
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {"batch_size": int(batch_size), "num_workers": 0, "pin_memory": True},
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
        views=("weak",),
        validation_targets="noisy",
        needs_noise_manifest=True,
        validation_size=int(validation_size),
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/gce")))
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=int(ctx.get("seed", noise_seed)),
    )
    ctx[save_as] = prepared
    ctx["validation_loader"] = prepared.loader(DataRole.NOISY_VALIDATION, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, batch_size=int(batch_size))
    ctx["train_dataset"] = prepared.dataset_for(DataRole.TRAIN)
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {"method": str(noise_method), "rate": float(noise_rate), "seed": int(noise_seed)}


@block(
    id="prepare_cdr_cifar10",
    name="Prepare CDR CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare the formal CDR CIFAR-10 split, GCE-2018 preprocessing, symmetric-40 manifest, and loaders.",
    params={
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 5000, "min": 1},
        "augment": {"type": "bool", "default": True},
        "noise_rate": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 64, "min": 1},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "validation_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_cdr_cifar10(
    ctx: ScratchContext,
    root: str = "",
    validation_size: int = 5000,
    augment: bool = True,
    noise_rate: float = 0.4,
    noise_seed: int = 1,
    batch_size: int = 64,
    save_as: str = "prepared_data",
) -> None:
    """Use the shared formal CDR data and persisted transition-sampled manifest."""
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "data": {
            "name": "cifar10",
            "validation_size": int(validation_size),
            "augment": bool(augment),
            "preprocessing": "gce2018",
        },
        "noise": {
            "name": "symmetric",
            "rate": float(noise_rate),
            "seed": int(noise_seed),
            "sampling": "transition",
            "rng": "numpy_legacy",
            "validation_targets": "noisy",
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {"batch_size": int(batch_size), "num_workers": 0, "pin_memory": True},
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
        views=("weak",),
        validation_targets="noisy",
        needs_noise_manifest=True,
        validation_size=int(validation_size),
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/cdr")))
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=int(ctx.get("seed", noise_seed)),
    )
    ctx[save_as] = prepared
    ctx["validation_loader"] = prepared.loader(DataRole.NOISY_VALIDATION, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, batch_size=int(batch_size))
    ctx["train_dataset"] = prepared.dataset_for(DataRole.TRAIN)
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {"method": "symmetric", "rate": float(noise_rate), "seed": int(noise_seed), "sampling": "transition"}


@block(
    id="prepare_dual_t_cifar10",
    name="Prepare Dual-T CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare the formal Dual-T CIFAR-10 standard-preprocessing split, symmetric-20 manifest, and stage loaders.",
    params={
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 10000, "min": 1},
        "augment": {"type": "bool", "default": True},
        "noise_rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "validation_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_dual_t_cifar10(
    ctx: ScratchContext,
    root: str = "",
    validation_size: int = 10000,
    augment: bool = True,
    noise_rate: float = 0.2,
    noise_seed: int = 1,
    batch_size: int = 128,
    save_as: str = "prepared_data",
) -> None:
    """Use one persisted noisy manifest for both Dual-T stages and clean test evaluation."""
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "data": {
            "name": "cifar10",
            "validation_size": int(validation_size),
            "augment": bool(augment),
            "preprocessing": "standard",
        },
        "noise": {
            "name": "symmetric",
            "rate": float(noise_rate),
            "seed": int(noise_seed),
            "validation_targets": "noisy",
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {"batch_size": int(batch_size), "num_workers": 0, "pin_memory": True},
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
        validation_targets="noisy",
        needs_noise_manifest=True,
        validation_size=int(validation_size),
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/dual-t")))
    prepared = prepare_experiment_data(config, requirements=requirements, run_dir=run_dir, seed=int(ctx.get("seed", noise_seed)))
    ctx[save_as] = prepared
    ctx["train_loader"] = prepared.loader(DataRole.TRAIN, epoch=0, batch_size=int(batch_size))
    ctx["validation_loader"] = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {"method": "symmetric", "rate": float(noise_rate), "seed": int(noise_seed)}


@block(
    id="prepare_pdl_cifar10",
    name="Prepare PDL CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare the formal PDL CIFAR-10 split, standard preprocessing, instance-dependent manifest, and noisy validation loaders.",
    params={
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 5000, "min": 1},
        "augment": {"type": "bool", "default": False},
        "noise_rate": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "train_loader", "validation_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_pdl_cifar10(
    ctx: ScratchContext,
    root: str = "",
    validation_size: int = 5000,
    augment: bool = False,
    noise_rate: float = 0.4,
    noise_seed: int = 1,
    batch_size: int = 128,
    save_as: str = "prepared_data",
) -> None:
    """Use PDL's official numpy-choice-complement split and manifest contract."""
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "data": {
            "name": "cifar10",
            "validation_size": int(validation_size),
            "augment": bool(augment),
            "preprocessing": "standard",
        },
        "noise": {
            "name": "pdl",
            "rate": float(noise_rate),
            "seed": int(noise_seed),
            "validation_targets": "noisy",
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {"batch_size": int(batch_size), "num_workers": 0, "pin_memory": True},
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
        views=("weak",),
        validation_targets="noisy",
        needs_noise_manifest=True,
        validation_size=int(validation_size),
        split_strategy="numpy_choice_complement",
        subset_before_split=True,
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/pdl")))
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=int(ctx.get("seed", noise_seed)),
    )
    ctx[save_as] = prepared
    ctx["train_loader"] = prepared.loader(DataRole.TRAIN, epoch=0, batch_size=int(batch_size))
    ctx["validation_loader"] = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {"method": "pdl", "rate": float(noise_rate), "seed": int(noise_seed)}


@block(
    id="prepare_volminnet_cifar10",
    name="Prepare VolMinNet CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare the formal VolMinNet CIFAR-10 standard split, augmentation, symmetric-20 manifest, and noisy validation loaders.",
    params={
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 5000, "min": 1},
        "augment": {"type": "bool", "default": True},
        "noise_rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "num_workers": {"type": "int", "default": 4, "min": 0},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "train_loader", "validation_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_volminnet_cifar10(
    ctx: ScratchContext,
    root: str = "",
    validation_size: int = 5000,
    augment: bool = True,
    noise_rate: float = 0.2,
    noise_seed: int = 1,
    batch_size: int = 128,
    num_workers: int = 4,
    save_as: str = "prepared_data",
) -> None:
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "data": {
            "name": "cifar10",
            "validation_size": int(validation_size),
            "validation_split": {"strategy": "stratified", "rng": "default_rng"},
            "augment": bool(augment),
            "preprocessing": "standard",
        },
        "noise": {
            "name": "symmetric",
            "rate": float(noise_rate),
            "seed": int(noise_seed),
            "sampling": "transition",
            "rng": "default_rng",
            "validation_targets": "noisy",
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": True,
        },
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
        validation_targets="noisy",
        needs_noise_manifest=True,
        validation_size=int(validation_size),
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/volminnet")))
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=int(ctx.get("seed", noise_seed)),
    )
    ctx[save_as] = prepared
    ctx["train_loader"] = prepared.loader(DataRole.TRAIN, epoch=0, batch_size=int(batch_size))
    ctx["validation_loader"] = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {"method": "symmetric", "rate": float(noise_rate), "seed": int(noise_seed)}


@block(
    id="prepare_t_revision_cifar10",
    name="Prepare T-Revision CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare the formal T-Revision CIFAR-10 split, symmetric-20 manifest, train-eval posterior loader, and noisy validation.",
    params={
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 5000, "min": 1},
        "augment": {"type": "bool", "default": True},
        "noise_rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "num_workers": {"type": "int", "default": 4, "min": 0},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "train_loader", "validation_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_t_revision_cifar10(
    ctx: ScratchContext,
    root: str = "",
    validation_size: int = 5000,
    augment: bool = True,
    noise_rate: float = 0.2,
    noise_seed: int = 1,
    batch_size: int = 128,
    num_workers: int = 4,
    save_as: str = "prepared_data",
) -> None:
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "data": {
            "name": "cifar10",
            "validation_size": int(validation_size),
            "augment": bool(augment),
            "preprocessing": "standard",
        },
        "noise": {
            "name": "symmetric",
            "rate": float(noise_rate),
            "sampling": "transition",
            "seed": int(noise_seed),
            "validation_targets": "noisy",
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": True,
        },
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.TRAIN_EVAL, DataRole.NOISY_VALIDATION, DataRole.TEST}),
        validation_targets="noisy",
        needs_noise_manifest=True,
        validation_size=int(validation_size),
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/t-revision")))
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=int(ctx.get("seed", noise_seed)),
    )
    ctx[save_as] = prepared
    ctx["train_loader"] = prepared.loader(DataRole.TRAIN, epoch=0, batch_size=int(batch_size))
    ctx["validation_loader"] = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {"method": "symmetric", "rate": float(noise_rate), "seed": int(noise_seed), "sampling": "transition"}


@block(
    id="prepare_cwd_cifar10",
    name="Prepare CWD CIFAR-10 Binary Protocol",
    category="Data",
    description="Prepare the formal five-fold CIFAR-10 airplane/automobile CWD protocol and effective-train noise manifest.",
    params={
        "root": {"type": "path", "default": ""},
        "folds": {"type": "int", "default": 5, "min": 2},
        "fold_index": {"type": "int", "default": 0, "min": 0},
        "augment": {"type": "bool", "default": False},
        "rho_positive": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "rho_negative": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 17, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "num_workers": {"type": "int", "default": 4, "min": 0},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "train_loader", "train_eval_loader", "test_loader", "transition"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_cwd_cifar10(
    ctx: ScratchContext,
    root: str = "",
    folds: int = 5,
    fold_index: int = 0,
    augment: bool = False,
    rho_positive: float = 0.2,
    rho_negative: float = 0.2,
    noise_seed: int = 17,
    batch_size: int = 128,
    num_workers: int = 4,
    save_as: str = "prepared_data",
) -> None:
    import torch
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    if int(fold_index) >= int(folds):
        raise ValueError("CWD fold_index must be smaller than folds")
    config: dict[str, Any] = {
        "data": {"name": "cifar10_airplane_automobile", "folds": int(folds), "fold_index": int(fold_index), "augment": bool(augment)},
        "noise": {
            "name": "binary_asymmetric_rcn", "rho_positive": float(rho_positive), "rho_negative": float(rho_negative),
            "seed": int(noise_seed), "sampling": "transition", "manifest_filename": "noise_manifest.npz",
        },
        "loader": {"batch_size": int(batch_size), "num_workers": int(num_workers), "pin_memory": True},
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.TRAIN_EVAL, DataRole.TEST}),
            manifest_scope="effective_train",
        ),
        run_dir=Path(str(ctx.get("artifact_dir", "artifacts/scratch/cwd"))),
        seed=int(ctx.get("seed", noise_seed)),
    )
    ctx[save_as] = prepared
    ctx["train_loader"] = prepared.loader(DataRole.TRAIN, epoch=0, batch_size=int(batch_size))
    ctx["train_eval_loader"] = prepared.loader(DataRole.TRAIN_EVAL, shuffle=False, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    ctx["num_classes"] = 2
    ctx["transition"] = torch.tensor([[1.0 - float(rho_negative), float(rho_negative)], [float(rho_positive), 1.0 - float(rho_positive)]], dtype=torch.float64)
    ctx["noise_config"] = {"name": "binary_asymmetric_rcn", "rho_positive": float(rho_positive), "rho_negative": float(rho_negative), "seed": int(noise_seed), "folds": int(folds), "fold_index": int(fold_index)}


@block(
    id="prepare_loss_correction_cifar10",
    name="Prepare Loss Correction CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare the formal CIFAR-10 asymmetric-40 manifest, validation split, and known transition artifact.",
    params={
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 5000, "min": 1},
        "augment": {"type": "bool", "default": True},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "validation_loader", "test_loader", "transition"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_loss_correction_cifar10(
    ctx: ScratchContext,
    root: str = "",
    validation_size: int = 5000,
    augment: bool = True,
    noise_seed: int = 1,
    batch_size: int = 128,
    save_as: str = "prepared_data",
) -> None:
    """Prepare the known class-conditional transition used by the formal recipe."""
    import torch

    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    transition = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
    ]
    config: dict[str, Any] = {
        "data": {
            "name": "cifar10",
            "validation_size": int(validation_size),
            "augment": bool(augment),
            "preprocessing": "gce2018",
        },
        "noise": {
            "name": "class_conditional",
            "rate": 0.4,
            "seed": int(noise_seed),
            "rng": "numpy_legacy",
            "validation_targets": "noisy",
            "manifest_filename": "noise_manifest.npz",
            "transition_matrix": transition,
        },
        "loader": {"batch_size": int(batch_size), "num_workers": 0, "pin_memory": True},
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
        views=("weak",),
        validation_targets="noisy",
        needs_noise_manifest=True,
        validation_size=int(validation_size),
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/loss-correction")))
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=int(ctx.get("seed", noise_seed)),
    )
    ctx[save_as] = prepared
    ctx["validation_loader"] = prepared.loader(DataRole.NOISY_VALIDATION, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    ctx["train_dataset"] = prepared.dataset_for(DataRole.TRAIN)
    ctx["num_classes"] = prepared.num_classes
    ctx["transition"] = torch.as_tensor(transition, dtype=torch.float32)
    ctx["noise_config"] = {
        "method": "class_conditional",
        "rate": 0.4,
        "seed": int(noise_seed),
        "validation_targets": "noisy",
    }


@block(
    id="prepare_jocor_cifar10",
    name="Prepare JoCoR CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare the formal JoCoR tensor-only CIFAR-10 split and symmetric transition-noise manifest.",
    params={
        "root": {"type": "path", "default": ""},
        "noise_rate": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 0, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "num_workers": {"type": "int", "default": 4, "min": 0},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_jocor_cifar10(
    ctx: ScratchContext,
    root: str = "",
    noise_rate: float = 0.5,
    noise_seed: int = 0,
    batch_size: int = 128,
    num_workers: int = 4,
    save_as: str = "prepared_data",
) -> None:
    """Use the formal JoCoR data/noise contract without creating batch labels."""
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "data": {
            "name": "cifar10",
            "validation_size": 0,
            "preprocessing": "tensor_only",
            "augment": False,
        },
        "noise": {
            "name": "symmetric",
            "rate": float(noise_rate),
            "seed": int(noise_seed),
            "sampling": "transition",
            "rng": "numpy_legacy",
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": True,
            "drop_last": True,
        },
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.TEST}),
        views=("weak",),
        validation_targets="clean",
        needs_noise_manifest=True,
        validation_size=0,
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/jocor")))
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=int(ctx.get("seed", noise_seed)),
    )
    ctx[save_as] = prepared
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    ctx["train_dataset"] = prepared.dataset_for(DataRole.TRAIN)
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {
        "method": "symmetric",
        "rate": float(noise_rate),
        "seed": int(noise_seed),
        "sampling": "transition",
    }


@block(
    id="prepare_apl_cifar10",
    name="Prepare APL CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare the formal APL CIFAR-10 split, standard preprocessing, per-class noise manifest, and loaders.",
    params={
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 0, "min": 0},
        "augment": {"type": "bool", "default": True},
        "noise_method": {"type": "enum", "options": ["symmetric"], "default": "symmetric"},
        "noise_rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "num_workers": {"type": "int", "default": 8, "min": 0},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_apl_cifar10(
    ctx: ScratchContext,
    root: str = "",
    validation_size: int = 0,
    augment: bool = True,
    noise_method: str = "symmetric",
    noise_rate: float = 0.2,
    noise_seed: int = 1,
    batch_size: int = 128,
    num_workers: int = 8,
    save_as: str = "prepared_data",
) -> None:
    """Use the formal APL data contract without creating labels in a batch."""
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "data": {
            "name": "cifar10",
            "validation_size": int(validation_size),
            "augment": bool(augment),
            "preprocessing": "standard",
        },
        "noise": {
            "name": str(noise_method),
            "rate": float(noise_rate),
            "seed": int(noise_seed),
            "sampling": "per_class",
            "validation_targets": "clean",
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {"batch_size": int(batch_size), "num_workers": int(num_workers), "pin_memory": True},
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.TEST}),
        views=("weak",),
        validation_targets="clean",
        needs_noise_manifest=True,
        validation_size=int(validation_size),
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/apl")))
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=int(ctx.get("seed", noise_seed)),
    )
    ctx[save_as] = prepared
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    ctx["train_dataset"] = prepared.dataset_for(DataRole.TRAIN)
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {
        "method": str(noise_method),
        "rate": float(noise_rate),
        "seed": int(noise_seed),
        "sampling": "per_class",
    }


@block(
    id="prepare_binary_risk_data",
    name="Prepare Binary Risk Reproduction Data",
    category="Data",
    description="Prepare the formal synthetic_binary_2d train/test split and class-conditional noise manifest for Natarajan risk.",
    params={
        "train_size": {"type": "int", "default": 512, "min": 2},
        "test_size": {"type": "int", "default": 2048, "min": 2},
        "data_seed": {"type": "int", "default": 2013, "min": 0},
        "rho_positive": {"type": "float", "default": 0.4, "min": 0.0, "max": 0.999},
        "rho_negative": {"type": "float", "default": 0.4, "min": 0.0, "max": 0.999},
        "noise_seed": {"type": "int", "default": 2013, "min": 0},
        "batch_size": {"type": "int", "default": 64, "min": 1},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "train_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_binary_risk_data(
    ctx: ScratchContext,
    train_size: int = 512,
    test_size: int = 2048,
    data_seed: int = 2013,
    rho_positive: float = 0.4,
    rho_negative: float = 0.4,
    noise_seed: int = 2013,
    batch_size: int = 64,
    save_as: str = "prepared_data",
) -> None:
    """Use the formal deterministic binary data and known class-conditional noise."""
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "seed": int(data_seed),
        "data": {
            "name": "synthetic_binary_2d",
            "train_size": int(train_size),
            "test_size": int(test_size),
            "seed": int(data_seed),
        },
        "noise": {
            "name": "binary_asymmetric_rcn",
            "rho_positive": float(rho_positive),
            "rho_negative": float(rho_negative),
            "seed": int(noise_seed),
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {"batch_size": int(batch_size), "num_workers": 0, "pin_memory": False},
    }
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.TEST}),
        views=("weak",),
        validation_targets="clean",
        needs_noise_manifest=True,
        validation_size=0,
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/binary-risk")))
    prepared = prepare_experiment_data(config, requirements=requirements, run_dir=run_dir, seed=int(data_seed))
    ctx[save_as] = prepared
    ctx["train_loader"] = prepared.loader(DataRole.TRAIN, epoch=0, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, epoch=0, shuffle=False, batch_size=int(batch_size))
    ctx["train_dataset"] = prepared.dataset_for(DataRole.TRAIN)
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {
        "method": "binary_asymmetric_rcn",
        "rho_positive": float(rho_positive),
        "rho_negative": float(rho_negative),
        "seed": int(noise_seed),
    }


@block(
    id="prepare_importance_reweighting_binary",
    name="Prepare Importance Reweighting Binary Data",
    category="Data",
    description="Prepare the formal low-dimensional synthetic binary split, asymmetric RCN manifest, and stable-index loaders.",
    params={
        "train_size": {"type": "int", "default": 4096, "min": 2},
        "validation_size": {"type": "int", "default": 1024, "min": 2},
        "test_size": {"type": "int", "default": 1024, "min": 2},
        "data_seed": {"type": "int", "default": 17, "min": 0},
        "rho_positive": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.999},
        "rho_negative": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.999},
        "noise_seed": {"type": "int", "default": 29, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "train_loader", "validation_loader", "test_loader", "posterior_features", "posterior_targets", "posterior_indices"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_importance_reweighting_binary(
    ctx: ScratchContext,
    train_size: int = 4096,
    validation_size: int = 1024,
    test_size: int = 1024,
    data_seed: int = 17,
    rho_positive: float = 0.2,
    rho_negative: float = 0.1,
    noise_seed: int = 29,
    batch_size: int = 128,
    save_as: str = "prepared_data",
) -> None:
    """Prepare the maintained low-dimensional paper workflow without clean-label leakage."""
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "seed": int(data_seed),
        "data": {
            "name": "synthetic_binary_2d",
            "dimension": 2,
            "train_size": int(train_size),
            "validation_size": int(validation_size),
            "test_size": int(test_size),
            "seed": int(data_seed),
        },
        "noise": {
            "name": "binary_asymmetric_rcn",
            "rho_positive": float(rho_positive),
            "rho_negative": float(rho_negative),
            "seed": int(noise_seed),
            "validation_targets": "noisy",
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {"batch_size": int(batch_size), "num_workers": 0, "pin_memory": False},
    }
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
        validation_targets="noisy",
        needs_noise_manifest=True,
        validation_size=int(validation_size),
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/importance-reweighting")))
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir,
        seed=int(data_seed),
    )
    train_dataset = prepared.dataset_for(DataRole.TRAIN)
    import numpy as np

    features = np.stack([
        np.asarray(train_dataset[index]["input"], dtype=np.float32)
        for index in range(len(train_dataset))
    ])
    targets = np.asarray([
        int(train_dataset[index]["target"]) for index in range(len(train_dataset))
    ], dtype=np.int64)
    ctx[save_as] = prepared
    ctx["train_loader"] = prepared.loader(DataRole.TRAIN, epoch=0, stream=1000, batch_size=int(batch_size))
    ctx["validation_loader"] = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False, stream=2000, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, stream=3000, batch_size=int(batch_size))
    ctx["train_dataset"] = train_dataset
    ctx["num_classes"] = prepared.num_classes
    ctx["posterior_features"] = features
    ctx["posterior_targets"] = targets
    ctx["posterior_indices"] = prepared.train_indices.copy()
    ctx["noise_config"] = {
        "method": "binary_asymmetric_rcn",
        "rho_positive": float(rho_positive),
        "rho_negative": float(rho_negative),
        "seed": int(noise_seed),
    }


@block(
    id="refresh_epoch_loader",
    name="Refresh Epoch Data Loader",
    category="Data",
    description="Build a deterministic loader for the current epoch from prepared experiment data.",
    params={
        "data": {"type": "slot", "default": "prepared_data"},
        "role": {"type": "enum", "options": ["train", "train_eval", "noisy_validation", "clean_validation", "trusted_validation", "test"], "default": "train"},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "save_as": {"type": "slot", "default": "train_loader"},
    },
    requires=("data", "epoch"),
    provides=("save_as",),
    placement=("epoch",), stage="data", ui_group="① 数据准备", beginner_visible=False,
)
def refresh_epoch_loader(
    ctx: ScratchContext,
    data: str = "prepared_data",
    role: str = "train",
    batch_size: int = 128,
    save_as: str = "train_loader",
) -> None:
    """Refresh the loader so each epoch uses the formal seeded shuffle stream."""
    ctx[save_as] = ctx[data].loader(str(role), epoch=int(ctx["epoch"]), batch_size=int(batch_size))


@block(
    id="prepare_coteaching_cifar10",
    name="Prepare Co-teaching CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare the formal Co-teaching CIFAR-10 split, standard preprocessing, transition-sampled noise manifest, and loaders.",
    params={
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 5000, "min": 1},
        "augment": {"type": "bool", "default": True},
        "noise_rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "num_workers": {"type": "int", "default": 4, "min": 0},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "validation_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备", beginner_visible=False,
)
def prepare_coteaching_cifar10(
    ctx: ScratchContext,
    root: str = "",
    validation_size: int = 5000,
    augment: bool = True,
    noise_rate: float = 0.2,
    noise_seed: int = 1,
    batch_size: int = 128,
    num_workers: int = 4,
    save_as: str = "prepared_data",
) -> None:
    """Use the shared formal Co-teaching data and transition noise contract."""
    if bool(ctx.get("_runtime_limits", {}).get("fixture")):
        prepared = _ScratchFixturePrepared(classes=10)
        ctx[save_as] = prepared
        ctx["validation_loader"] = prepared.loader("noisy_validation", batch_size=int(batch_size))
        ctx["test_loader"] = prepared.loader("test", batch_size=int(batch_size))
        ctx["num_classes"] = prepared.num_classes
        ctx["noise_config"] = {"method": "symmetric", "rate": float(noise_rate), "seed": int(noise_seed), "sampling": "transition", "fixture": True}
        return
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data

    config: dict[str, Any] = {
        "data": {
            "name": "cifar10",
            "validation_size": int(validation_size),
            "validation_split": {"strategy": "stratified", "rng": "default_rng"},
            "augment": bool(augment),
            "preprocessing": "standard",
        },
        "noise": {
            "name": "symmetric",
            "rate": float(noise_rate),
            "seed": int(noise_seed),
            "sampling": "transition",
            "rng": "default_rng",
            "validation_targets": "noisy",
            "manifest_filename": "noise_manifest.npz",
        },
        "loader": {"batch_size": int(batch_size), "num_workers": int(num_workers), "pin_memory": True},
    }
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    requirements = DataRequirements(
        roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
        views=("weak",),
        validation_targets="noisy",
        needs_noise_manifest=True,
        validation_size=int(validation_size),
    )
    run_dir = Path(str(ctx.get("artifact_dir", "artifacts/scratch/coteaching")))
    prepared = prepare_experiment_data(config, requirements=requirements, run_dir=run_dir, seed=int(ctx.get("seed", noise_seed)))
    ctx[save_as] = prepared
    ctx["validation_loader"] = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {"method": "symmetric", "rate": float(noise_rate), "seed": int(noise_seed), "sampling": "transition"}


@block(
    id="prepare_cnlcu_cifar10",
    name="Prepare CNLCU CIFAR-10 Reproduction Data",
    category="Data",
    description="Prepare CNLCU's formal CIFAR-10 standard-preprocessing split, transition noise manifest, and stable-index loaders.",
    params={
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 5000, "min": 1},
        "augment": {"type": "bool", "default": True},
        "noise_rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "num_workers": {"type": "int", "default": 4, "min": 0},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "validation_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_cnlcu_cifar10(ctx: ScratchContext, **params: Any) -> None:
    """CNLCU shares the formal data contract with Co-teaching but owns a named block."""
    prepare_coteaching_cifar10(ctx, **params)
    import torch
    ctx["cnlcu_train_indices"] = torch.as_tensor(ctx[params.get("save_as", "prepared_data")].train_indices).clone()


@block(
    id="prepare_formal_cifar",
    name="Prepare Formal CIFAR Data",
    category="Data",
    description="Prepare a formal CIFAR-10/100 split with explicit preprocessing, transition noise, manifest, and evaluation loaders.",
    params={
        "dataset": {"type": "enum", "options": ["cifar10", "cifar100"], "default": "cifar10"},
        "root": {"type": "path", "default": ""},
        "validation_size": {"type": "int", "default": 5000, "min": 0},
        "train_eval": {"type": "bool", "default": False},
        "clean_validation": {"type": "bool", "default": False},
        "trusted_validation": {"type": "bool", "default": False},
        "num_clean": {"type": "int", "default": 100, "min": 1},
        "trusted_seed": {"type": "int", "default": 1234, "min": 0},
        "augment": {"type": "bool", "default": True},
        "preprocessing": {"type": "str", "default": "standard"},
        "noise_rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "noise_seed": {"type": "int", "default": 1, "min": 0},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "num_workers": {"type": "int", "default": 0, "min": 0},
        "save_as": {"type": "slot", "default": "prepared_data"},
    },
    provides=("save_as", "num_classes", "validation_loader", "test_loader", "train_eval_loader", "trusted_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_formal_cifar(ctx: ScratchContext, dataset: str = "cifar10", root: str = "", validation_size: int = 5000, train_eval: bool = False, clean_validation: bool = False, trusted_validation: bool = False, num_clean: int = 100, trusted_seed: int = 1234, augment: bool = True, preprocessing: str = "standard", noise_rate: float = 0.2, noise_seed: int = 1, batch_size: int = 128, num_workers: int = 0, save_as: str = "prepared_data") -> None:
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.training.data_service import prepare_experiment_data
    name = str(dataset).lower()
    if name not in {"cifar10", "cifar100"}:
        raise ValueError("formal CIFAR data block supports cifar10 or cifar100")
    if bool(ctx.get("_runtime_limits", {}).get("fixture")):
        prepared = _ScratchFixturePrepared(classes=10 if name == "cifar10" else 100)
        ctx[save_as] = prepared
        ctx["train_dataset"] = prepared.dataset_for(DataRole.TRAIN) if 'DataRole' in locals() else prepared.dataset_for("train")
        ctx["validation_loader"] = prepared.loader("validation", batch_size=int(batch_size))
        ctx["test_loader"] = prepared.loader("test", batch_size=int(batch_size))
        ctx["train_eval_loader"] = prepared.loader("train_eval", batch_size=int(batch_size))
        ctx["trusted_loader"] = prepared.loader("trusted_validation", batch_size=int(batch_size))
        ctx["num_classes"] = prepared.num_classes
        ctx["noise_config"] = {"method": "symmetric", "rate": float(noise_rate), "seed": int(noise_seed), "dataset": name, "fixture": True}
        return
    config: dict[str, Any] = {
        "data": {"name": name, "validation_size": int(validation_size), "augment": bool(augment), "preprocessing": str(preprocessing)},
        "noise": {"name": "symmetric", "rate": float(noise_rate), "seed": int(noise_seed), "sampling": "transition", "rng": "default_rng", "validation_targets": "noisy", "manifest_filename": "noise_manifest.npz"},
        "loader": {"batch_size": int(batch_size), "num_workers": int(num_workers), "pin_memory": True},
        "trusted_validation": {"source": "official_generated", "batch_size": int(batch_size), "seed": int(trusted_seed)},
    }
    config["data"]["num_clean"] = int(num_clean)
    config["data"]["trusted_seed"] = int(trusted_seed)
    if str(root).strip(): config["data"]["root"] = str(root).strip()
    roles = {DataRole.TRAIN, DataRole.TEST}
    if int(validation_size) > 0:
        roles.add(DataRole.NOISY_VALIDATION)
    if bool(train_eval):
        roles.add(DataRole.TRAIN_EVAL)
    if bool(clean_validation):
        roles.add(DataRole.CLEAN_VALIDATION)
    if bool(trusted_validation):
        roles.add(DataRole.TRUSTED_VALIDATION)
    req = DataRequirements(roles=frozenset(roles), validation_targets="noisy" if int(validation_size) > 0 else "clean", needs_noise_manifest=True, validation_size=max(1, int(validation_size)))
    prepared = prepare_experiment_data(config, requirements=req, run_dir=Path(str(ctx.get("artifact_dir", "artifacts/scratch/formal-cifar"))), seed=int(ctx.get("seed", noise_seed)))
    ctx[save_as] = prepared
    ctx["train_dataset"] = prepared.dataset_for(DataRole.TRAIN)
    validation_role = DataRole.CLEAN_VALIDATION if bool(clean_validation) else (DataRole.NOISY_VALIDATION if int(validation_size) > 0 else DataRole.TEST)
    ctx["validation_loader"] = prepared.loader(validation_role, shuffle=False, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    if bool(train_eval):
        ctx["train_eval_loader"] = prepared.loader(DataRole.TRAIN_EVAL, shuffle=False, batch_size=int(batch_size))
    if bool(trusted_validation):
        ctx["trusted_loader"] = prepared.loader(DataRole.TRUSTED_VALIDATION, shuffle=False, batch_size=int(batch_size))
    ctx["num_classes"] = prepared.num_classes
    ctx["noise_config"] = {"method": "symmetric", "rate": float(noise_rate), "seed": int(noise_seed), "dataset": name}


@block(
    id="load_cifar100",
    name="Load CIFAR-100",
    category="Data",
    description="Load one CIFAR-100 split with torchvision.",
    params={
        "root": {"type": "str", "default": "data"},
        "train": {"type": "bool", "default": True},
        "download": {"type": "bool", "default": False},
        "save_as": {"type": "slot", "default": "train_dataset"},
    },
    provides=("save_as", "num_classes"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def load_cifar100(ctx: ScratchContext, **params: Any) -> None:
    _load_torchvision_dataset("CIFAR100", ctx, params)


@block(
    id="create_loader",
    name="Create Data Loader",
    category="Data",
    description="Wrap a dataset in a PyTorch DataLoader.",
    params={
        "dataset": {"type": "slot", "required": True},
        "batch_size": {"type": "int", "default": 128, "min": 1},
        "shuffle": {"type": "bool", "default": True},
        "save_as": {"type": "slot", "default": "train_loader"},
    },
    requires=("dataset",),
    provides=("save_as",),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def create_loader(
    ctx: ScratchContext,
    dataset: str,
    batch_size: int = 128,
    shuffle: bool = True,
    save_as: str = "train_loader",
) -> None:
    torch = _torch()
    ctx[save_as] = torch.utils.data.DataLoader(ctx[dataset], batch_size=batch_size, shuffle=shuffle)


@block(
    id="apply_symmetric_noise",
    name="Apply Symmetric Noise",
    category="Data",
    description="Flip labels uniformly to another class while preserving clean labels.",
    params={
        "labels": {"type": "slot", "default": "labels"},
        "noise_rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        "num_classes": {"type": "int", "default": 2, "min": 2},
        "save_as": {"type": "slot", "default": "labels"},
    },
    requires=("labels",),
    provides=("save_as", "clean_labels"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def apply_symmetric_noise(
    ctx: ScratchContext,
    labels: str = "labels",
    noise_rate: float = 0.2,
    num_classes: int = 2,
    save_as: str = "labels",
) -> None:
    torch = _torch()
    clean = ctx[labels].clone()
    generator = torch.Generator(device=clean.device if clean.is_cuda else "cpu").manual_seed(int(ctx.get("seed", 1)))
    mask = torch.rand(clean.shape, generator=generator, device=clean.device) < float(noise_rate)
    offsets = torch.randint(1, num_classes, clean.shape, generator=generator, device=clean.device)
    noisy = torch.where(mask, (clean + offsets) % int(num_classes), clean)
    ctx["clean_labels"] = clean
    ctx[save_as] = noisy


@block(
    id="prepare_pcse_cifar10",
    name="Prepare PCSE CIFAR-10 External Protocol",
    category="Data",
    description="Prepare standard CIFAR-10 and attach the immutable external UPM noise manifest required by formal PCSE.",
    params={"root": {"type": "path", "default": ""}, "source_env": {"type": "str", "default": "LNL_PCSE_SOURCE_RUN"}, "batch_size": {"type": "int", "default": 128, "min": 1}, "num_workers": {"type": "int", "default": 4, "min": 0}, "save_as": {"type": "slot", "default": "prepared_data"}},
    provides=("save_as", "num_classes", "train_loader", "train_eval_loader", "validation_loader", "test_loader", "transition"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def prepare_pcse_cifar10(ctx: ScratchContext, root: str = "", source_env: str = "LNL_PCSE_SOURCE_RUN", batch_size: int = 128, num_workers: int = 4, save_as: str = "prepared_data") -> None:
    import os
    from lnl_toolbox.data import DataRequirements, DataRole
    from lnl_toolbox.noise.manifest import NoiseManifest
    from lnl_toolbox.training.data_service import prepare_experiment_data
    source_root = Path(os.environ.get(str(source_env), "")).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"PCSE source run is not available: {source_env}")
    source_manifest_path = source_root / "noise_manifest.npz"
    manifest = NoiseManifest.load(source_manifest_path)
    config: dict[str, Any] = {"data": {"name": "cifar10", "validation_size": 5000, "augment": False, "preprocessing": "standard"}, "loader": {"batch_size": int(batch_size), "num_workers": int(num_workers), "pin_memory": True}}
    if str(root).strip():
        config["data"]["root"] = str(root).strip()
    prepared = prepare_experiment_data(config, requirements=DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.TRAIN_EVAL, DataRole.NOISY_VALIDATION, DataRole.TEST}), validation_targets="noisy"), run_dir=Path(str(ctx.get("artifact_dir", "artifacts/scratch/pcse"))), seed=int(ctx.get("seed", 1)))
    if prepared.num_classes != 10:
        raise ValueError("PCSE formal protocol requires CIFAR-10 with 10 classes")
    mapping = {int(i): int(y) for i, y in zip(manifest.global_indices, manifest.noisy_targets)}
    prepared.manifest = manifest
    prepared.manifest_path = source_manifest_path
    for role, indices, training in ((DataRole.TRAIN, prepared.train_indices, True), (DataRole.TRAIN_EVAL, prepared.train_indices, False), (DataRole.NOISY_VALIDATION, prepared.validation_indices, False)):
        prepared.datasets[role] = prepared.dynamic_dataset(indices, targets_by_index=mapping, training=training)
    ctx[save_as] = prepared
    ctx["train_loader"] = prepared.loader(DataRole.TRAIN, batch_size=int(batch_size))
    ctx["train_eval_loader"] = prepared.loader(DataRole.TRAIN_EVAL, shuffle=False, batch_size=int(batch_size))
    ctx["validation_loader"] = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False, batch_size=int(batch_size))
    ctx["test_loader"] = prepared.loader(DataRole.TEST, shuffle=False, batch_size=int(batch_size))
    ctx["num_classes"] = 10
    ctx["transition"] = torch.eye(10, dtype=torch.float64)
    ctx["noise_config"] = {"name": "external", "source_env": str(source_env), "manifest": str(source_manifest_path), "validation_targets": "noisy"}
