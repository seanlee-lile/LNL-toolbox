"""Small data blocks; datasets and loaders are kept in Context slots."""

from __future__ import annotations

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
        inputs = values.get("images", values.get("inputs", values.get("x")))
        labels = values.get("labels", values.get("targets", values.get("y")))
        indices = values.get("indices", values.get("index"))
        return inputs, labels, indices
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("a batch must contain inputs and labels")
    indices = batch[2] if len(batch) > 2 else None
    return batch[0], batch[1], indices


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
)
def load_cifar10(ctx: ScratchContext, **params: Any) -> None:
    _load_torchvision_dataset("CIFAR10", ctx, params)


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
