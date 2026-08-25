"""Small data blocks; datasets and loaders are kept in Context slots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..context import ScratchContext
from ..registry import block
from ..data_runtime import materialize_plan


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


# ---------------------------------------------------------------------------
# Scratch-native data plan
# ---------------------------------------------------------------------------
#
# These blocks deliberately keep a plain, serialisable plan in ScratchContext.
# They do not import legacy data packages. Materialisation below is a
# small Scratch-owned implementation used by the runtime and by fixture runs;
# the legacy pipeline remains available to tests as a formal-protocol oracle.

_VALID_LABEL_SOURCES = {"observed", "clean"}
_VALID_ROLES = {
    "train", "train_eval", "noisy_validation", "clean_validation",
    "trusted_validation", "test",
}


def _plan(ctx: ScratchContext, slot: str = "data_plan") -> dict[str, Any]:
    current = ctx.get(slot)
    if current is None:
        current = {"data": {}, "split": {}, "labels": {}, "noise": {},
                   "manifest": {}, "preprocessing": {}, "views": ["weak"],
                   "roles": ["train", "clean_validation", "test"],
                   "requirements": {"roles": ["train", "clean_validation", "test"],
                                    "views": ["weak"], "validation_targets": "clean"},
                   "loader": {}}
        ctx[slot] = current
    if not isinstance(current, dict):
        raise TypeError(f"{slot} must contain a mutable data plan")
    return current


def _role_name(role: Any) -> str:
    value = getattr(role, "value", role)
    value = str(value).strip().lower()
    aliases = {"validation": "clean_validation", "noisy-valid": "noisy_validation",
               "trusted": "trusted_validation", "train-eval": "train_eval"}
    return aliases.get(value, value)


def _dataset_num_classes(name: str) -> int:
    key = str(name).lower()
    return {"cifar10": 10, "cifar100": 100, "mnist": 10,
            "fashion_mnist": 10, "synthetic": 2,
            "synthetic_classification": 2}.get(key, 2)


@block(
    id="load_dataset", name="Load Dataset", category="Data",
    description="Create a Scratch-native dataset plan without selecting a paper-specific bundle.",
    params={"dataset": {"type": "dataset", "required": True},
            "root": {"type": "path", "default": ""},
            "path": {"type": "path", "default": ""},
            "options": {"type": "value", "default": {}},
            "save_as": {"type": "slot", "default": "data_plan"}},
    provides=("save_as", "data_spec"), placement=("top",), stage="data", ui_group="① 数据准备",
)
def load_dataset(ctx: ScratchContext, dataset: str, root: str = "", path: str = "",
                 options: Mapping[str, Any] | None = None, save_as: str = "data_plan") -> None:
    plan = _plan(ctx, save_as)
    opts = dict(options or {})
    data = {"name": str(dataset).strip(), "root": str(root), "path": str(path), **opts}
    if not data["name"]:
        raise ValueError("load_dataset needs a dataset name")
    catalog = ctx.get("dataset_catalog")
    if isinstance(catalog, Mapping) and data["name"] in catalog:
        data["source"] = catalog[data["name"]]
    plan["data"] = data
    plan.setdefault("seed", int(ctx.get("seed", 1)))
    ctx[save_as] = plan
    ctx["data_spec"] = {k: v for k, v in data.items() if k != "source"}


@block(
    id="inspect_dataset_semantics", name="Inspect Dataset Semantics", category="Data",
    description="Record dataset capabilities and label semantics for an explicit recipe audit.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "save_as": {"type": "slot", "default": "dataset_semantics"}},
    requires=("data_plan",), provides=("save_as",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def inspect_dataset_semantics(ctx: ScratchContext, data_plan: str = "data_plan",
                              save_as: str = "dataset_semantics") -> None:
    plan = _plan(ctx, data_plan)
    data = plan.get("data", {})
    name = str(data.get("name", ""))
    clean_available = data.get("clean_targets_available")
    if clean_available is None:
        source = data.get("source")
        if isinstance(source, Mapping):
            clean_available = source.get("clean_targets_available", True)
        else:
            # Built-in supervised datasets expose their ground-truth labels;
            # custom sources must opt out explicitly when they do not.
            clean_available = True
    num_classes = 2 if isinstance(data.get("binary_classes"), (list, tuple)) and len(data["binary_classes"]) == 2 else int(data.get("num_classes") or _dataset_num_classes(name))
    ctx[save_as] = {"dataset": name, "num_classes": num_classes,
                    "has_clean_targets": bool(clean_available), "has_observed_targets": True,
                    "source": "context" if data.get("source") is not None else "scratch-native"}
    plan["semantics"] = dict(ctx[save_as])


@block(
    id="create_dataset_split", name="Create Dataset Split", category="Data",
    description="Configure a deterministic train/validation split in the Scratch data plan.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "validation_size": {"type": "int", "default": 0, "min": 0},
            "split_strategy": {"type": "enum", "options": ["random", "stratified", "official"], "default": "random"},
            "split_seed": {"type": "int", "default": 1, "min": 0},
            "subset_before_split": {"type": "bool", "default": False}},
    requires=("data_plan",), provides=("data_plan",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def create_dataset_split(ctx: ScratchContext, data_plan: str = "data_plan", validation_size: int = 0,
                         split_strategy: str = "random", split_seed: int = 1,
                         subset_before_split: bool = False) -> None:
    plan = _plan(ctx, data_plan)
    plan["split"] = {"validation_size": int(validation_size), "strategy": str(split_strategy),
                     "seed": int(split_seed), "subset_before_split": bool(subset_before_split)}
    plan["requirements"].update({"validation_size": int(validation_size),
                                  "split_strategy": str(split_strategy),
                                  "subset_before_split": bool(subset_before_split)})
    plan["data"]["validation_size"] = int(validation_size)


@block(
    id="select_label_source", name="Select Label Source", category="Data",
    description="Declare observed or clean labels per role while preventing clean-label training leakage.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "train": {"type": "enum", "options": ["observed", "clean"], "default": "observed"},
            "validation": {"type": "enum", "options": ["clean", "observed"], "default": "clean"},
            "test": {"type": "enum", "options": ["clean", "observed"], "default": "clean"},
            "trusted": {"type": "enum", "options": ["clean", "observed"], "default": "clean"}},
    requires=("data_plan",), provides=("data_plan",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def select_label_source(ctx: ScratchContext, data_plan: str = "data_plan", train: str = "observed",
                        validation: str = "clean", test: str = "clean", trusted: str = "clean") -> None:
    values = {"train": str(train), "validation": str(validation), "test": str(test), "trusted": str(trusted)}
    if any(value not in _VALID_LABEL_SOURCES for value in values.values()):
        raise ValueError("label sources must be `observed` or `clean`")
    if values["train"] == "clean":
        raise ValueError("clean labels cannot be selected for the training role")
    plan = _plan(ctx, data_plan)
    plan["labels"] = values
    plan["requirements"]["label_sources"] = dict(values)
    plan["requirements"]["validation_targets"] = "noisy" if values["validation"] == "observed" else "clean"


@block(
    id="apply_noise", name="Apply Noise", category="Data",
    description="Declare a Scratch-owned label-noise transformation while retaining clean targets.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "name": {"type": "str", "default": "none"},
            "rate": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            "seed": {"type": "int", "default": 1, "min": 0},
            "sampling": {"type": "str", "default": "transition"},
            "options": {"type": "value", "default": {}}},
    requires=("data_plan",), provides=("data_plan",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def apply_noise(ctx: ScratchContext, data_plan: str = "data_plan", name: str = "none",
                rate: float = 0.0, seed: int = 1, sampling: str = "transition",
                options: Mapping[str, Any] | None = None) -> None:
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("noise rate must be between 0 and 1")
    plan = _plan(ctx, data_plan)
    plan["noise"] = {"name": str(name), "rate": float(rate), "seed": int(seed),
                     "sampling": str(sampling), **dict(options or {})}


@block(
    id="build_noise_manifest", name="Build Noise Manifest", category="Data",
    description="Declare the aligned noise manifest required by the recipe protocol.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "required": {"type": "bool", "default": True},
            "scope": {"type": "enum", "options": ["train_split", "effective_train"], "default": "train_split"},
            "filename": {"type": "str", "default": "noise_manifest.npz"},
            "external_path": {"type": "path", "default": ""}},
    requires=("data_plan",), provides=("data_plan",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def build_noise_manifest(ctx: ScratchContext, data_plan: str = "data_plan", required: bool = True,
                         scope: str = "train_split", filename: str = "noise_manifest.npz",
                         external_path: str = "") -> None:
    plan = _plan(ctx, data_plan)
    plan["manifest"] = {"required": bool(required), "scope": str(scope), "filename": str(filename),
                        "external_path": str(external_path)}
    plan["requirements"]["needs_noise_manifest"] = bool(required)
    plan["requirements"]["manifest_scope"] = str(scope)


@block(
    id="configure_preprocessing", name="Configure Preprocessing", category="Data",
    description="Record preprocessing and augmentation as an explicit data operation.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "preprocessing": {"type": "str", "default": "standard"},
            "augment": {"type": "bool", "default": False},
            "strong_augment": {"type": "bool", "default": False},
            "options": {"type": "value", "default": {}}},
    requires=("data_plan",), provides=("data_plan",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def configure_preprocessing(ctx: ScratchContext, data_plan: str = "data_plan", preprocessing: str = "standard",
                            augment: bool = False, strong_augment: bool = False,
                            options: Mapping[str, Any] | None = None) -> None:
    plan = _plan(ctx, data_plan)
    plan["preprocessing"] = {"name": str(preprocessing), "augment": bool(augment),
                             "strong_augment": bool(strong_augment), **dict(options or {})}


@block(
    id="configure_views", name="Configure Views", category="Data",
    description="Declare weak/strong or other explicit dataset views.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "views": {"type": "value", "default": ["weak"]}},
    requires=("data_plan",), provides=("data_plan",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def configure_views(ctx: ScratchContext, data_plan: str = "data_plan", views: Sequence[str] = ("weak",)) -> None:
    values = [str(value) for value in views]
    if not values or len(values) != len(set(values)):
        raise ValueError("views must be a non-empty list of unique names")
    plan = _plan(ctx, data_plan)
    plan["views"] = values
    plan["requirements"]["views"] = list(values)


@block(
    id="assign_data_roles", name="Assign Data Roles", category="Data",
    description="Select the train, validation, trusted, and test roles materialised by the plan.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "roles": {"type": "value", "default": ["train", "clean_validation", "test"]},
            "train_drop_last": {"type": "bool", "default": False}},
    requires=("data_plan",), provides=("data_plan",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def assign_data_roles(ctx: ScratchContext, data_plan: str = "data_plan",
                      roles: Sequence[str] = ("train", "clean_validation", "test"),
                      train_drop_last: bool = False) -> None:
    values = [_role_name(value) for value in roles]
    unknown = set(values) - _VALID_ROLES
    if unknown or "train" not in values or "test" not in values:
        raise ValueError(f"roles must include train/test and use known names; invalid={sorted(unknown)}")
    plan = _plan(ctx, data_plan)
    plan["roles"] = list(dict.fromkeys(values))
    plan["requirements"]["roles"] = list(plan["roles"])
    plan["loader"]["drop_last"] = bool(train_drop_last)


@block(
    id="configure_loader", name="Configure Loader", category="Data",
    description="Set loader parameters independently of dataset preparation.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "batch_size": {"type": "int", "default": 128, "min": 1},
            "num_workers": {"type": "int", "default": 0, "min": 0},
            "pin_memory": {"type": "bool", "default": False},
            "drop_last": {"type": "bool", "default": False},
            "options": {"type": "value", "default": {}}},
    requires=("data_plan",), provides=("data_plan",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def configure_loader(ctx: ScratchContext, data_plan: str = "data_plan", batch_size: int = 128,
                     num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False,
                     options: Mapping[str, Any] | None = None) -> None:
    plan = _plan(ctx, data_plan)
    plan["loader"] = {"batch_size": int(batch_size), "num_workers": int(num_workers),
                      "pin_memory": bool(pin_memory), "drop_last": bool(drop_last), **dict(options or {})}


@block(
    id="build_prepared_data", name="Build Prepared Data", category="Data",
    description="Materialise the Scratch data plan into Scratch-owned role datasets.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "artifact_dir": {"type": "path", "default": ""},
            "save_as": {"type": "slot", "default": "prepared_data"}},
    requires=("data_plan",), provides=("save_as", "num_classes", "transition", "posterior_features", "posterior_targets", "posterior_indices"), placement=("top",), stage="data", ui_group="① 数据准备",
)
def build_prepared_data(ctx: ScratchContext, data_plan: str = "data_plan", artifact_dir: str = "",
                        save_as: str = "prepared_data") -> None:
    plan = _plan(ctx, data_plan)
    prepared, manifest = materialize_plan(plan, ctx)
    ctx[save_as] = prepared
    ctx["noise_manifest"] = manifest
    ctx["num_classes"] = int(prepared.num_classes)
    torch = _torch()
    ctx["transition"] = torch.eye(prepared.num_classes)
    ctx["posterior_features"] = torch.empty((0, 0))
    ctx["posterior_targets"] = torch.empty((0,), dtype=torch.long)
    ctx["posterior_indices"] = torch.empty((0,), dtype=torch.long)
    ctx["data_materialization"] = "scratch-native"


@block(
    id="build_loaders", name="Build Loaders", category="Data",
    description="Expose role-specific loaders from Scratch PreparedData.",
    params={"prepared_data": {"type": "slot", "default": "prepared_data"},
            "batch_size": {"type": "int", "default": 0, "min": 0}},
    requires=("prepared_data",), provides=("train_loader", "train_eval_loader", "validation_loader", "trusted_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def build_loaders(ctx: ScratchContext, prepared_data: str = "prepared_data", batch_size: int = 0) -> None:
    prepared = ctx[prepared_data]
    size = int(batch_size or getattr(prepared, "loader_config", {}).get("batch_size", 128))
    ctx["train_loader"] = prepared.loader("train", batch_size=size)
    ctx["test_loader"] = prepared.loader("test", shuffle=False, batch_size=size)
    targets = getattr(prepared, "plan", {}).get("requirements", {}).get("validation_targets", "clean")
    validation = "noisy_validation" if targets == "noisy" else "clean_validation"
    if validation in getattr(prepared, "datasets", {}):
        ctx["validation_loader"] = prepared.loader(validation, shuffle=False, batch_size=size)
    for role, slot in (("train_eval", "train_eval_loader"), ("trusted_validation", "trusted_loader")):
        if role in getattr(prepared, "datasets", {}):
            ctx[slot] = prepared.loader(role, shuffle=False, batch_size=size)
    ctx["train_dataset"] = prepared.dataset_for("train")
    ctx["num_classes"] = int(prepared.num_classes)


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
        "clean_as": {"type": "slot", "default": "clean_targets"},
    },
    requires=("batch",),
    provides=("input_as", "label_as", "index_as", "clean_as"),
    placement=("batch",), stage="data", ui_group="① 数据准备",
)
def get_batch(
    ctx: ScratchContext,
    batch: str = "batch",
    input_as: str = "images",
    label_as: str = "labels",
    index_as: str = "indices",
    clean_as: str = "clean_targets",
) -> None:
    inputs, labels, indices = _batch_values(ctx[batch])
    ctx[input_as] = inputs
    ctx[label_as] = labels
    if indices is not None:
        ctx[index_as] = indices
    if isinstance(ctx[batch], Mapping):
        ctx[clean_as] = ctx[batch].get("clean_targets")


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
    id="refresh_epoch_loader", name="Refresh Epoch Loader", category="Data",
    description="Rebuild a role loader for the current epoch without leaving Scratch runtime.",
    params={"data": {"type": "slot", "default": "prepared_data"},
            "role": {"type": "str", "default": "train"},
            "batch_size": {"type": "int", "default": 128, "min": 1},
            "save_as": {"type": "slot", "default": "train_loader"}},
    requires=("data",), provides=("save_as",), placement=("top", "epoch", "batch"),
    stage="data", ui_group="① 数据准备",
)
def refresh_epoch_loader(ctx: ScratchContext, data: str = "prepared_data", role: str = "train",
                         batch_size: int = 128, save_as: str = "train_loader") -> None:
    prepared = ctx[data]
    ctx[save_as] = prepared.loader(role, epoch=int(ctx.get("epoch", 0)), batch_size=int(batch_size))


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
