"""Small data blocks; datasets and loaders are kept in Context slots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..context import ScratchContext
from ..registry import block


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
        "loader_as": {"type": "slot", "default": "train_loader"},
    },
    provides=("save_as", "loader_as", "num_classes", "validation_loader", "test_loader"),
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
    loader_as: str = "train_loader",
) -> None:
    """Build the GCE data contract without importing the legacy data service."""
    torch = _torch()
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - optional train extra
        raise RuntimeError("CIFAR preparation requires torchvision; install the `train` extra.") from exc

    data_root = str(root).strip() or "data"
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]) if augment else transforms.ToTensor()
    train_base = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    test_base = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transforms.ToTensor())
    if int(validation_size) <= 0 or int(validation_size) >= len(train_base):
        raise ValueError("validation_size must be smaller than the CIFAR-10 training set")
    generator = torch.Generator().manual_seed(int(ctx.get("seed", noise_seed)))
    permutation = torch.randperm(len(train_base), generator=generator).tolist()
    validation_indices = permutation[: int(validation_size)]
    train_indices = permutation[int(validation_size):]
    noisy_targets = list(train_base.targets)
    if str(noise_method).lower() == "symmetric" and float(noise_rate) > 0:
        noise_generator = torch.Generator().manual_seed(int(noise_seed))
        for index in range(len(noisy_targets)):
            if float(torch.rand((), generator=noise_generator)) < float(noise_rate):
                offset = int(torch.randint(1, 10, (), generator=noise_generator))
                noisy_targets[index] = (int(noisy_targets[index]) + offset) % 10
    elif str(noise_method).lower() not in {"none", "symmetric"}:
        raise ValueError(f"unsupported GCE noise method `{noise_method}`")
    train_base.targets = noisy_targets

    class IndexedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = list(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, position):
            inputs, label = self.dataset[self.indices[position]]
            return inputs, int(label), int(self.indices[position])

    train_dataset = IndexedSubset(train_base, train_indices)
    validation_dataset = IndexedSubset(train_base, validation_indices)
    test_dataset = IndexedSubset(test_base, range(len(test_base)))
    ctx[save_as] = {
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "num_classes": 10,
        "clean_train_targets": list(getattr(train_base, "targets", ())),
    }
    ctx[loader_as] = torch.utils.data.DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    ctx["validation_loader"] = torch.utils.data.DataLoader(validation_dataset, batch_size=int(batch_size), shuffle=False)
    ctx["test_loader"] = torch.utils.data.DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)
    ctx["num_classes"] = 10
    ctx["train_dataset"] = train_dataset
    ctx["noise_config"] = {"method": str(noise_method), "rate": float(noise_rate), "seed": int(noise_seed)}


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
