"""Small data blocks; datasets and loaders are kept in Context slots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..context import ScratchContext
from ..registry import block
from ..data_runtime import (
    ScratchSplit,
    apply_noise_to_split,
    assemble_prepared,
    build_transforms,
    build_role_datasets,
    load_sources,
    registered_dataset_config,
    split_source,
)


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


def _serializable(value: Any) -> Any:
    """Keep ``data_plan`` an audit/config object, never a runtime object.

    Recipe options are user supplied, so this boundary is deliberately
    defensive: primitive values and nested containers are retained while
    tensors, datasets, transforms and other live objects are represented by a
    stable type marker.  Runtime values always travel through Context slots.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    try:
        from pathlib import Path
        if isinstance(value, Path):
            return str(value)
    except Exception:
        pass
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _serializable_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _serializable(dict(plan))


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
    # Existing plans may have been produced by an older runtime.  Scrub live
    # objects once at the boundary; subsequent Blocks use Context slots for
    # every runtime value and only append serialisable configuration here.
    cleaned = _serializable_plan(current)
    if cleaned != current:
        current.clear()
        current.update(cleaned)
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
    requires=(), provides=("save_as", "data_spec", "train_source", "validation_source", "test_source", "num_classes"), placement=("top",), stage="data", ui_group="① 数据准备",
)
def load_dataset(ctx: ScratchContext, dataset: str, root: str = "", path: str = "",
                 options: Mapping[str, Any] | None = None, save_as: str = "data_plan") -> None:
    plan = _plan(ctx, save_as)
    opts = dict(options or {})
    data = {"name": str(dataset).strip(), "root": str(root), "path": str(path), **opts}
    if not data["name"]:
        raise ValueError("load_dataset needs a dataset name")
    registered = registered_dataset_config(data["name"])
    if registered is not None:
        registered_data = dict(registered.get("data", {}))
        for key, value in registered_data.items():
            if key == "name" or value is None or value == "":
                continue
            if data.get(key) is None or data.get(key) == "":
                data[key] = value
        data["adapter"] = str(registered.get("adapter") or data.get("adapter") or "").strip().lower()
        data["registered_alias"] = str(registered.get("alias") or data["name"])
    catalog = ctx.get("dataset_catalog")
    if isinstance(catalog, Mapping) and data["name"] in catalog:
        data["source"] = catalog[data["name"]]
    plan["data"] = _serializable({key: value for key, value in data.items() if key != "source"})
    plan.setdefault("seed", int(ctx.get("seed", 1)))
    load_plan = dict(plan)
    load_plan["data"] = data
    effective, train_source, validation_source, test_source = load_sources(load_plan, ctx)
    # Fixture substitution is a source-loading concern. Keep the requested
    # recipe metadata, but make the concrete sources available immediately.
    if effective is not plan:
        plan["data"] = _serializable({key: value for key, value in dict(effective.get("data", {})).items()
                                       if key != "source"})
        plan["split"] = _serializable(dict(effective.get("split", plan.get("split", {}))))
        plan["loader"] = _serializable(dict(effective.get("loader", plan.get("loader", {}))))
        plan["noise"] = _serializable(dict(effective.get("noise", plan.get("noise", {}))))
    plan["num_classes"] = int(train_source.num_classes)
    ctx[save_as] = plan
    ctx["data_spec"] = {k: v for k, v in data.items() if k != "source"}
    ctx["train_source"] = train_source
    ctx["validation_source"] = validation_source
    ctx["test_source"] = test_source
    ctx["num_classes"] = int(train_source.num_classes)


@block(
    id="inspect_dataset_semantics", name="Inspect Dataset Semantics", category="Data",
    description="Record dataset capabilities and label semantics for an explicit recipe audit.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "save_as": {"type": "slot", "default": "dataset_semantics"}},
    requires=("train_source", "test_source"), provides=("save_as",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def inspect_dataset_semantics(ctx: ScratchContext, data_plan: str = "data_plan",
                              save_as: str = "dataset_semantics") -> None:
    plan = _plan(ctx, data_plan)
    data = plan.get("data", {})
    name = str(data.get("name", ""))
    train_source = ctx.get("train_source")
    test_source = ctx.get("test_source")
    if train_source is None or test_source is None:
        raise ValueError("inspect_dataset_semantics requires loaded source slots")
    clean_available = bool(getattr(train_source, "has_clean_targets", False))
    num_classes = int(getattr(train_source, "num_classes", data.get("num_classes") or _dataset_num_classes(name)))
    ctx[save_as] = {"dataset": name, "num_classes": num_classes,
                    "has_clean_targets": bool(clean_available), "has_observed_targets": True,
                    "has_test_source": test_source is not None,
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
    requires=("train_source",), provides=("train_split", "validation_split"), placement=("top",), stage="data", ui_group="① 数据准备",
)
def create_dataset_split(ctx: ScratchContext, data_plan: str = "data_plan", validation_size: int = 0,
                         split_strategy: str = "random", split_seed: int = 1,
                         subset_before_split: bool = False) -> None:
    plan = _plan(ctx, data_plan)
    runtime_limits = ctx.get("_runtime_limits", {})
    if isinstance(runtime_limits, Mapping) and runtime_limits.get("fixture") and int(validation_size) > 2:
        validation_size = 2
    plan["split"] = {"validation_size": int(validation_size), "strategy": str(split_strategy),
                     "seed": int(split_seed), "subset_before_split": bool(subset_before_split)}
    plan["requirements"].update({"validation_size": int(validation_size),
                                  "split_strategy": str(split_strategy),
                                  "subset_before_split": bool(subset_before_split)})
    plan["data"]["validation_size"] = int(validation_size)
    train_source = ctx.get("train_source")
    if train_source is None:
        raise ValueError("create_dataset_split requires train_source from load_dataset")
    train_split, validation_split = split_source(train_source, ctx.get("validation_source"), plan)
    ctx["train_split"] = train_split
    ctx["validation_split"] = validation_split


@block(
    id="select_label_source", name="Select Label Source", category="Data",
    description="Declare observed or clean labels per role while preventing clean-label training leakage.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "train": {"type": "enum", "options": ["observed", "clean"], "default": "observed"},
            "validation": {"type": "enum", "options": ["clean", "observed"], "default": "clean"},
            "test": {"type": "enum", "options": ["clean", "observed"], "default": "clean"},
            "trusted": {"type": "enum", "options": ["clean", "observed"], "default": "clean"}},
    requires=("train_split", "validation_split", "test_source"), provides=("label_policy",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def select_label_source(ctx: ScratchContext, data_plan: str = "data_plan", train: str = "observed",
                        validation: str = "clean", test: str = "clean", trusted: str = "clean") -> None:
    values = {"train": str(train), "validation": str(validation), "test": str(test), "trusted": str(trusted)}
    if any(value not in _VALID_LABEL_SOURCES for value in values.values()):
        raise ValueError("label sources must be `observed` or `clean`")
    if values["train"] == "clean":
        raise ValueError("clean labels cannot be selected for the training role")
    for slot in ("train_split", "validation_split", "test_source"):
        if slot not in ctx:
            raise ValueError(f"select_label_source requires {slot}")
    plan = _plan(ctx, data_plan)
    plan["labels"] = values
    plan["requirements"]["label_sources"] = dict(values)
    plan["requirements"]["validation_targets"] = "noisy" if values["validation"] == "observed" else "clean"
    ctx["label_policy"] = dict(values)


@block(
    id="apply_noise", name="Apply Noise", category="Data",
    description="Declare a Scratch-owned label-noise transformation while retaining clean targets.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "name": {"type": "str", "default": "none"},
            "rate": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            "seed": {"type": "int", "default": 1, "min": 0},
            "sampling": {"type": "str", "default": "transition"},
            "options": {"type": "value", "default": {}},
            # Optional runtime evidence for instance-dependent noise.  When
            # supplied, the value must name an existing Context slot; when
            # omitted the block fails explicitly if the selected noise type
            # requires evidence rather than silently fabricating it.
            "class_scores": {"type": "slot"},
            "features": {"type": "slot"}},
    requires=("train_split",), provides=("noisy_train_split", "clean_train_split", "noise_state", "transition"), placement=("top",), stage="data", ui_group="① 数据准备",
)
def apply_noise(ctx: ScratchContext, data_plan: str = "data_plan", name: str = "none",
                rate: float = 0.0, seed: int = 1, sampling: str = "transition",
                options: Mapping[str, Any] | None = None,
                class_scores: str = "", features: str = "") -> None:
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("noise rate must be between 0 and 1")
    plan = _plan(ctx, data_plan)
    raw_options = dict(options or {})
    plan["noise"] = _serializable({"name": str(name), "rate": float(rate), "seed": int(seed),
                                    "sampling": str(sampling), **raw_options})
    runtime_limits = ctx.get("_runtime_limits", {})
    if isinstance(runtime_limits, Mapping) and runtime_limits.get("fixture") and str(name).lower() in {"external", "external_torch"}:
        plan["noise"] = {"name": "symmetric", "rate": float(rate), "seed": int(seed)}
    train_split = ctx.get("train_split")
    if train_split is None:
        raise ValueError("apply_noise requires train_split from create_dataset_split")
    # Runtime-only score/feature matrices are explicit Context inputs.  They
    # are never copied into the serialisable audit plan.
    noise_config = {"name": str(name), "rate": float(rate), "seed": int(seed),
                    "sampling": str(sampling), **raw_options}
    if isinstance(runtime_limits, Mapping) and runtime_limits.get("fixture") and str(name).lower() in {"external", "external_torch"}:
        noise_config = {"name": "symmetric", "rate": float(rate), "seed": int(seed)}
    if class_scores:
        if class_scores not in ctx:
            raise ValueError(f"apply_noise requires class-score slot `{class_scores}`")
        noise_config["class_scores"] = ctx[class_scores]
    if features:
        if features not in ctx:
            raise ValueError(f"apply_noise requires feature slot `{features}`")
        noise_config["features"] = ctx[features]
    if class_scores:
        plan["noise"]["class_scores_slot"] = str(class_scores)
    if features:
        plan["noise"]["features_slot"] = str(features)
    noisy_train, manifest, clean_by_index = apply_noise_to_split(train_split, noise_config)
    clean_train = train_split
    if clean_by_index and not train_split.has_clean_targets:
        clean_train = ScratchSplit(
            train_split.dataset,
            train_split.split,
            tuple(type(sample)(sample.input, sample.index, sample.observed_target,
                               clean_by_index[int(sample.index)]) for sample in train_split.samples),
            train_split.num_classes,
            train_split.version,
        )
    noise_state = {"manifest": manifest, "clean_by_index": clean_by_index,
                   "clean_train_split": clean_train}
    ctx["noisy_train_split"] = noisy_train
    ctx["noise_state"] = noise_state
    ctx["clean_train_split"] = clean_train
    transition = getattr(manifest, "transition_matrix", None)
    if transition is not None:
        ctx["transition"] = _torch().as_tensor(transition, dtype=_torch().float32)
    labels = dict(plan.get("labels", {}))
    validation = ctx.get("validation_split")
    if validation is not None and labels.get("validation") == "observed" and validation.samples:
        noisy_validation, _, _ = apply_noise_to_split(validation, plan["noise"])
        ctx["validation_split"] = noisy_validation


@block(
    id="build_noise_manifest", name="Build Noise Manifest", category="Data",
    description="Declare the aligned noise manifest required by the recipe protocol.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "required": {"type": "bool", "default": True},
            "scope": {"type": "enum", "options": ["train_split", "effective_train"], "default": "train_split"},
            "filename": {"type": "str", "default": "noise_manifest.npz"},
            "external_path": {"type": "path", "default": ""}},
    requires=("noise_state", "noisy_train_split"), provides=("noise_manifest",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def build_noise_manifest(ctx: ScratchContext, data_plan: str = "data_plan", required: bool = True,
                         scope: str = "train_split", filename: str = "noise_manifest.npz",
                         external_path: str = "") -> None:
    plan = _plan(ctx, data_plan)
    state = ctx.get("noise_state")
    if state is None or state.get("manifest") is None:
        if required:
            raise ValueError("build_noise_manifest requires noise_state from apply_noise")
    manifest = None if state is None else state.get("manifest")
    noisy_train = ctx.get("noisy_train_split")
    if manifest is not None and noisy_train is not None:
        manifest_indices = [int(value) for value in getattr(manifest, "global_indices", [])]
        split_indices = [int(sample.index) for sample in noisy_train.samples]
        manifest_targets = [int(value) for value in getattr(manifest, "noisy_targets", [])]
        split_targets = [int(sample.observed_target) for sample in noisy_train.samples]
        if manifest_indices != split_indices or manifest_targets != split_targets:
            raise ValueError("noise manifest is not aligned with noisy_train_split")
    plan["manifest"] = {"required": bool(required), "scope": str(scope), "filename": str(filename),
                        "external_path": str(external_path)}
    plan["requirements"]["needs_noise_manifest"] = bool(required)
    plan["requirements"]["manifest_scope"] = str(scope)
    if external_path:
        plan["manifest"]["external_path"] = str(external_path)
    ctx["noise_manifest"] = manifest


@block(
    id="configure_preprocessing", name="Configure Preprocessing", category="Data",
    description="Record preprocessing and augmentation as an explicit data operation.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "preprocessing": {"type": "str", "default": "standard"},
            "augment": {"type": "bool", "default": False},
            "strong_augment": {"type": "bool", "default": False},
            "options": {"type": "value", "default": {}}},
    requires=("train_split",), provides=("preprocessing_transform",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def configure_preprocessing(ctx: ScratchContext, data_plan: str = "data_plan", preprocessing: str = "standard",
                            augment: bool = False, strong_augment: bool = False,
                            options: Mapping[str, Any] | None = None) -> None:
    plan = _plan(ctx, data_plan)
    plan["preprocessing"] = _serializable({"name": str(preprocessing), "augment": bool(augment),
                                            "strong_augment": bool(strong_augment), **dict(options or {})})
    source = ctx.get("train_split")
    if source is None:
        raise ValueError("configure_preprocessing requires a loaded/split source")
    weak, views = build_transforms(plan["preprocessing"], source, plan.get("views", ["weak"]))
    ctx["preprocessing_transform"] = weak


@block(
    id="configure_views", name="Configure Views", category="Data",
    description="Declare weak/strong or other explicit dataset views.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "views": {"type": "value", "default": ["weak"]}},
    requires=("train_split", "preprocessing_transform"), provides=("view_transforms",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def configure_views(ctx: ScratchContext, data_plan: str = "data_plan", views: Sequence[str] = ("weak",)) -> None:
    values = [str(value) for value in views]
    if not values or len(values) != len(set(values)):
        raise ValueError("views must be a non-empty list of unique names")
    plan = _plan(ctx, data_plan)
    plan["views"] = list(values)
    plan["requirements"]["views"] = list(values)
    source = ctx.get("train_split")
    if source is None:
        raise ValueError("configure_views requires a loaded/split source")
    preprocessing_transform = ctx.get("preprocessing_transform")
    if "preprocessing_transform" not in ctx:
        raise ValueError("configure_views requires preprocessing_transform from configure_preprocessing")
    _, view_transforms = build_transforms(plan.get("preprocessing", {}), source, values,
                                          preprocessing_transform=preprocessing_transform)
    ctx["view_transforms"] = view_transforms


@block(
    id="assign_data_roles", name="Assign Data Roles", category="Data",
    description="Select the train, validation, trusted, and test roles materialised by the plan.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "roles": {"type": "value", "default": ["train", "clean_validation", "test"]},
            "train_drop_last": {"type": "bool", "default": False}},
    requires=("train_split", "clean_train_split", "noisy_train_split", "validation_split", "test_source", "preprocessing_transform", "view_transforms"), provides=("role_datasets",), placement=("top",), stage="data", ui_group="① 数据准备",
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
    if ctx.get("train_split") is None or ctx.get("noisy_train_split") is None:
        raise ValueError("assign_data_roles requires split and noise Block outputs")
    if ctx.get("test_source") is None and "test" in plan["roles"]:
        raise ValueError("test role requires an explicit test source")
    train_split = ctx.get("clean_train_split") or ctx["train_split"]
    data_config = dict(plan.get("data", {}))
    # Fixture sources are intentionally tiny.  When a recipe requests a
    # trusted subset, keep at least one sample in the learning-train role so
    # downstream fixture blocks can observe a real train batch.  This is
    # strictly a bounded-runtime adjustment; formal recipe values are left
    # untouched outside ``runtime_limits.fixture``.
    runtime_limits = ctx.get("_runtime_limits", {})
    if isinstance(runtime_limits, Mapping) and runtime_limits.get("fixture") and "trusted_validation" in plan["roles"]:
        train_count = len(ctx["train_split"].samples)
        if train_count > 1:
            requested = int(data_config.get("num_clean", data_config.get("trusted_size", 0)))
            capped = min(requested, train_count - 1)
            if "num_clean" in data_config:
                data_config["num_clean"] = capped
            else:
                data_config["trusted_size"] = capped
    datasets = build_role_datasets(
        train_split=train_split,
        noisy_train=ctx["noisy_train_split"],
        validation_split=ctx.get("validation_split") or ScratchSplit(
            ctx["train_split"].dataset, "validation", tuple(), ctx["train_split"].num_classes,
            ctx["train_split"].version,
        ),
        test_split=ctx["test_source"],
        roles=plan["roles"],
        data_config=data_config,
        preprocessing_transform=ctx["preprocessing_transform"],
        view_transforms=ctx["view_transforms"],
    )
    ctx["role_datasets"] = datasets


@block(
    id="configure_loader", name="Configure Loader", category="Data",
    description="Set loader parameters independently of dataset preparation.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "batch_size": {"type": "int", "default": 128, "min": 1},
            "num_workers": {"type": "int", "default": 0, "min": 0},
            "pin_memory": {"type": "bool", "default": False},
            "drop_last": {"type": "bool", "default": False},
            "options": {"type": "value", "default": {}}},
    requires=(), provides=("loader_spec",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def configure_loader(ctx: ScratchContext, data_plan: str = "data_plan", batch_size: int = 128,
                     num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False,
                     options: Mapping[str, Any] | None = None) -> None:
    plan = _plan(ctx, data_plan)
    runtime_limits = ctx.get("_runtime_limits", {})
    if isinstance(runtime_limits, Mapping) and runtime_limits.get("fixture"):
        num_workers = 0
        # Bounded fixture sources are intentionally tiny.  Keeping formal
        # ``drop_last`` here would silently skip every batch when the recipe
        # batch size exceeds the fixture size, preventing downstream Blocks
        # from observing the canonical batch contract.
        drop_last = False
    plan["loader"] = _serializable({"batch_size": int(batch_size), "num_workers": int(num_workers),
                                     "pin_memory": bool(pin_memory), "drop_last": bool(drop_last), **dict(options or {})})
    plan["loader_spec"] = dict(plan["loader"])
    ctx["loader_spec"] = plan["loader_spec"]


@block(
    id="build_prepared_data", name="Build Prepared Data", category="Data",
    description="Materialise the Scratch data plan into Scratch-owned role datasets.",
    params={"data_plan": {"type": "slot", "default": "data_plan"},
            "artifact_dir": {"type": "path", "default": ""},
            "save_as": {"type": "slot", "default": "prepared_data"}},
    requires=("role_datasets", "noise_manifest", "loader_spec", "num_classes"), provides=("save_as",), placement=("top",), stage="data", ui_group="① 数据准备",
)
def build_prepared_data(ctx: ScratchContext, data_plan: str = "data_plan", artifact_dir: str = "",
                        save_as: str = "prepared_data") -> None:
    plan = _plan(ctx, data_plan)
    datasets = ctx.get("role_datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("build_prepared_data requires completed role datasets")
    prepared = assemble_prepared(
        datasets,
        num_classes=int(ctx["num_classes"]),
        loader_config=dict(ctx["loader_spec"]),
        plan=plan,
        manifest=ctx.get("noise_manifest"),
    )
    ctx[save_as] = prepared
    ctx["num_classes"] = int(prepared.num_classes)
    # A transition is an explicit upstream artifact.  Assembly may expose it
    # when one was already materialised, but must never infer an identity
    # matrix (or fabricate posterior snapshots) as a side effect.
    ctx["data_materialization"] = "scratch-native"


@block(
    id="build_loaders", name="Build Loaders", category="Data",
    description="Expose role-specific loaders from Scratch PreparedData.",
    params={"role_datasets": {"type": "slot", "default": "role_datasets"},
            "loader_spec": {"type": "slot", "default": "loader_spec"},
            "batch_size": {"type": "int", "default": 0, "min": 0}},
    requires=("role_datasets", "loader_spec"), provides=("train_loader", "train_eval_loader", "validation_loader", "trusted_loader", "test_loader"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def build_loaders(ctx: ScratchContext, role_datasets: str = "role_datasets",
                  loader_spec: str = "loader_spec", batch_size: int = 0) -> None:
    datasets = ctx[role_datasets]
    config = dict(ctx[loader_spec])
    size = int(batch_size or config.get("batch_size", 128))
    import torch
    from ..data_runtime import collate_scratch_batch

    def make_loader(role: str, *, shuffle: bool = False) -> Any:
        dataset = datasets[role]
        return torch.utils.data.DataLoader(
            dataset, batch_size=size, shuffle=bool(shuffle and len(dataset) > 0),
            drop_last=bool(config.get("drop_last", False)),
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=bool(config.get("pin_memory", False)),
            collate_fn=collate_scratch_batch,
        )

    ctx["train_loader"] = make_loader("train", shuffle=True)
    ctx["test_loader"] = make_loader("test", shuffle=False)
    targets = ctx.get("label_policy", {}).get("validation", "clean")
    validation = "noisy_validation" if targets in {"observed", "noisy"} else "clean_validation"
    if validation in datasets:
        ctx["validation_loader"] = make_loader(validation, shuffle=False)
    for role, slot in (("train_eval", "train_eval_loader"), ("trusted_validation", "trusted_loader")):
        if role in datasets:
            ctx[slot] = make_loader(role, shuffle=False)


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
    id="get_next_batch",
    name="Get Next Batch",
    category="Data",
    description="Fetch and unpack the next batch from any configured role loader.",
    params={"loader": {"type": "slot", "default": "loader"}, "input_as": {"type": "slot", "default": "images"}, "label_as": {"type": "slot", "default": "labels"}, "index_as": {"type": "slot", "default": "indices"}, "clean_as": {"type": "slot", "default": "clean_targets"}},
    requires=("loader",), provides=("input_as", "label_as", "index_as", "clean_as"), placement=("batch",), stage="data", ui_group="① 数据准备",
)
def get_next_batch(ctx: ScratchContext, loader: str = "loader", input_as: str = "images", label_as: str = "labels", index_as: str = "indices", clean_as: str = "clean_targets") -> None:
    iterator_key = f"_scratch_iterator:{loader}"
    iterator = ctx.get(iterator_key)
    if iterator is None:
        iterator = iter(ctx[loader]); ctx[iterator_key] = iterator
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(ctx[loader]); ctx[iterator_key] = iterator; batch = next(iterator)
    inputs, labels, indices = _batch_values(batch)
    ctx[input_as], ctx[label_as] = inputs, labels
    if indices is not None:
        ctx[index_as] = indices
    if isinstance(batch, Mapping):
        ctx[clean_as] = batch.get("clean_targets")


@block(
    id="select_batch_view",
    name="Select Batch View",
    category="Data",
    description="Select a named view from the current batch, falling back to the primary input when absent.",
    params={"batch": {"type": "slot", "default": "batch"}, "input": {"type": "slot", "default": "images"}, "view": {"type": "str", "default": "strong_input"}, "save_as": {"type": "slot", "default": "view_input"}},
    requires=("batch", "input"), provides=("save_as",), placement=("batch",), stage="data", ui_group="① 数据准备",
)
def select_batch_view(ctx: ScratchContext, batch: str = "batch", input: str = "images", view: str = "strong_input", save_as: str = "view_input") -> None:
    value = ctx[batch].get(view) if isinstance(ctx[batch], Mapping) else None
    if value is None and isinstance(ctx[batch], Mapping) and view == "strong_input":
        value = ctx[batch].get("strong_images")
    ctx[save_as] = ctx[input] if value is None else value


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
    description="Create a deterministic Scratch source for fixture/example recipes; continue through the canonical data vocabulary.",
    params={
        "samples": {"type": "int", "default": 64, "min": 1},
        "features": {"type": "int", "default": 4, "min": 1},
        "classes": {"type": "int", "default": 2, "min": 2},
        "save_as": {"type": "slot", "default": "data_plan"},
    },
    provides=("save_as", "data_spec", "train_source", "test_source", "num_classes"),
    placement=("top",), stage="data", ui_group="① 数据准备",
)
def load_synthetic(
    ctx: ScratchContext,
    samples: int = 64,
    features: int = 4,
    classes: int = 2,
    save_as: str = "data_plan",
) -> None:
    # Synthetic data is a source constructor only.  It deliberately delegates
    # to the same source loader used by ``load_dataset`` so it cannot create a
    # second ``dataset -> loader`` dialect.
    load_dataset(
        ctx,
        dataset="synthetic",
        options={"samples": int(samples), "features": int(features), "classes": int(classes)},
        save_as=save_as,
    )


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
