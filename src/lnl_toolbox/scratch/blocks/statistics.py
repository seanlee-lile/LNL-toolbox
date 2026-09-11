"""Scratch-native statistics and transition operations.

These blocks intentionally stop at one statistical operation.  Snapshot
collection remains separate from estimation, and estimators never import the
legacy data/algorithm packages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from ..context import ScratchContext
from ..registry import block


def _torch():
    import torch
    return torch


def _as_indices(value):
    return _torch().as_tensor(value, dtype=_torch().long).reshape(-1)


def _row_normalize(value):
    torch = _torch()
    return value / value.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(value.dtype).tiny)


@block(
    id="estimate_transition", name="Estimate Transition Matrix", category="Statistics",
    description="Estimate a class-conditional transition matrix from model predictions and observed labels.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "estimated_transition"}},
    requires=("logits", "labels"), provides=("save_as",), placement=("batch", "top"), stage="setup", ui_group="⑥ 后验与权重",
)
def estimate_transition(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", save_as: str = "estimated_transition") -> None:
    """Estimate P(observed | clean) from detached posterior predictions.

    This is intentionally a single class-conditional estimator.  It does not
    collect snapshots, align indices, or apply the resulting matrix; those
    operations are separate public blocks.
    """
    torch = _torch()
    probabilities = torch.softmax(torch.as_tensor(ctx[logits]).detach(), dim=-1)
    targets = torch.as_tensor(ctx[labels], device=probabilities.device).long().reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.numel():
        raise ValueError("estimate_transition expects aligned [N,C] logits and labels")
    classes = int(probabilities.shape[-1])
    result = torch.zeros((classes, classes), dtype=probabilities.dtype, device=probabilities.device)
    for label in range(classes):
        mask = targets == label
        result[label] = probabilities[mask].mean(0) if bool(mask.any()) else torch.full(
            (classes,), 1.0 / classes, dtype=probabilities.dtype, device=probabilities.device
        )
    ctx[save_as] = _row_normalize(result)


@block(
    id="collect_posterior_snapshot", name="Collect Posterior Snapshot", category="Statistics",
    description="Collect detached class probabilities, observed targets and stable indices from a model/loader pair.",
    params={"model": {"type": "slot", "default": "model"}, "loader": {"type": "slot", "default": "loader"}, "device": {"type": "slot", "default": "device"}, "dataset": {"type": "str", "default": "dataset"}, "split": {"type": "str", "default": "split"}, "save_as": {"type": "slot", "default": "posterior_snapshot"}, "probabilities_as": {"type": "slot", "default": "posterior"}, "targets_as": {"type": "slot", "default": "snapshot_targets"}, "indices_as": {"type": "slot", "default": "snapshot_indices"}},
    requires=("model", "loader"), provides=("save_as", "probabilities_as", "targets_as", "indices_as"), placement=("top", "epoch"), stage="evaluate", ui_group="⑥ 后验与权重",
)
def collect_posterior_snapshot(ctx: ScratchContext, model: str = "model", loader: str = "loader", device: str = "device", dataset: str = "dataset", split: str = "split", save_as: str = "posterior_snapshot", probabilities_as: str = "posterior", targets_as: str = "snapshot_targets", indices_as: str = "snapshot_indices") -> None:
    from ..native_stats import collect_posterior_snapshot as collect
    target_device = ctx.get(device, "cpu")
    value = collect(ctx[model], ctx[loader], target_device, dataset=str(ctx.get(dataset, dataset)), split=str(ctx.get(split, split)))
    ctx[save_as] = value
    ctx[probabilities_as] = value.noisy_probabilities
    ctx[targets_as] = value.noisy_targets
    ctx[indices_as] = value.global_indices


@block(
    id="collect_feature_snapshot", name="Collect Feature Snapshot", category="Statistics",
    description="Collect detached feature vectors, observed targets and stable indices from a model/loader pair.",
    params={"model": {"type": "slot", "default": "model"}, "loader": {"type": "slot", "default": "loader"}, "device": {"type": "slot", "default": "device"}, "dataset": {"type": "str", "default": "dataset"}, "split": {"type": "str", "default": "split"}, "layers": {"type": "value", "default": []}, "save_as": {"type": "slot", "default": "feature_snapshot"}, "features_as": {"type": "slot", "default": "features"}, "targets_as": {"type": "slot", "default": "snapshot_targets"}, "indices_as": {"type": "slot", "default": "snapshot_indices"}},
    requires=("model", "loader"), provides=("save_as", "features_as", "targets_as", "indices_as"), placement=("top", "epoch"), stage="evaluate", ui_group="⑥ 后验与权重",
)
def collect_feature_snapshot(ctx: ScratchContext, model: str = "model", loader: str = "loader", device: str = "device", dataset: str = "dataset", split: str = "split", layers: Sequence[str] = (), save_as: str = "feature_snapshot", features_as: str = "features", targets_as: str = "snapshot_targets", indices_as: str = "snapshot_indices") -> None:
    from ..native_stats import PCSEFeatureLayerConfig, collect_feature_snapshot as collect, collect_pcse_features
    target_device = ctx.get(device, "cpu")
    layer_names = tuple(str(value) for value in (layers or ()))
    if layer_names:
        value = collect_pcse_features(
            ctx[model], ctx[loader], target_device,
            dataset=str(ctx.get(dataset, dataset)), split=str(ctx.get(split, split)),
            layers=tuple(PCSEFeatureLayerConfig(name, "global_average") for name in layer_names),
        ).snapshots
    else:
        value = collect(ctx[model], ctx[loader], target_device, dataset=str(ctx.get(dataset, dataset)), split=str(ctx.get(split, split)))
    ctx[save_as] = value
    if isinstance(value, tuple):
        ctx[features_as] = tuple(item.features for item in value)
        ctx[targets_as] = tuple(item.noisy_targets for item in value)
        ctx[indices_as] = tuple(item.global_indices for item in value)
    else:
        ctx[features_as] = value.features
        ctx[targets_as] = value.noisy_targets
        ctx[indices_as] = value.global_indices


@block(
    id="collect_loss_snapshot", name="Collect Loss Snapshot", category="Statistics",
    description="Collect one detached per-example loss for an entire loader, aligned by stable sample index.",
    params={
        "model": {"type": "slot", "default": "model"},
        "loader": {"type": "slot", "default": "loader"},
        "device": {"type": "slot", "default": "device"},
        "save_as": {"type": "slot", "default": "loss_snapshot"},
        "losses_as": {"type": "slot", "default": "snapshot_losses"},
        "targets_as": {"type": "slot", "default": "snapshot_targets"},
        "indices_as": {"type": "slot", "default": "snapshot_indices"},
    },
    requires=("model", "loader"),
    provides=("save_as", "losses_as", "targets_as", "indices_as"),
    placement=("top", "epoch"), stage="evaluate", ui_group="⑥ 后验与权重",
)
def collect_loss_snapshot(
    ctx: ScratchContext,
    model: str = "model",
    loader: str = "loader",
    device: str = "device",
    save_as: str = "loss_snapshot",
    losses_as: str = "snapshot_losses",
    targets_as: str = "snapshot_targets",
    indices_as: str = "snapshot_indices",
) -> None:
    """Collect full-dataset losses without fitting a downstream estimator.

    DivideMix's co-divide stage fits its mixture on the complete training
    loss distribution.  Keeping this snapshot operation separate from GMM
    fitting prevents a minibatch from silently becoming the estimator's
    population and makes the stable-index contract explicit.
    """
    torch = _torch()
    import torch.nn.functional as F

    network = ctx[model]
    target_device = ctx.get(device, "cpu")
    was_training = bool(getattr(network, "training", False))
    losses, targets, indices = [], [], []
    network.eval()
    try:
        with torch.no_grad():
            for batch in ctx[loader]:
                if isinstance(batch, Mapping):
                    inputs = batch.get("inputs", batch.get("images", batch.get("input")))
                    labels = batch.get("targets", batch.get("labels", batch.get("target")))
                    sample_indices = batch.get("indices", batch.get("index"))
                else:
                    inputs, labels, sample_indices = batch[0], batch[1], batch[2] if len(batch) > 2 else None
                if inputs is None or labels is None or sample_indices is None:
                    raise ValueError("collect_loss_snapshot requires inputs, targets and stable indices")
                inputs = inputs.to(target_device) if hasattr(inputs, "to") else inputs
                labels = torch.as_tensor(labels, device=target_device, dtype=torch.long).reshape(-1)
                sample_indices = torch.as_tensor(sample_indices, dtype=torch.long).reshape(-1)
                logits = network(inputs)
                per_sample = F.cross_entropy(logits, labels, reduction="none")
                if per_sample.numel() != sample_indices.numel():
                    raise ValueError("collect_loss_snapshot outputs are not sample aligned")
                losses.append(per_sample.detach().cpu())
                targets.append(labels.detach().cpu())
                indices.append(sample_indices.detach().cpu())
    finally:
        network.train(was_training)
    if not losses:
        raise ValueError("collect_loss_snapshot loader is empty")
    loss_values = torch.cat(losses)
    target_values = torch.cat(targets)
    index_values = torch.cat(indices)
    order = torch.argsort(index_values, stable=True)
    index_values = index_values[order]
    if torch.unique(index_values).numel() != index_values.numel():
        raise ValueError("collect_loss_snapshot indices must be unique")
    loss_values, target_values = loss_values[order], target_values[order]
    ctx[losses_as] = loss_values
    ctx[targets_as] = target_values
    ctx[indices_as] = index_values
    ctx[save_as] = SimpleNamespace(losses=loss_values, targets=target_values, global_indices=index_values)


def _resolve_snapshot_values(ctx: ScratchContext, snapshots: Any) -> tuple[Any, ...]:
    values = ctx[snapshots] if isinstance(snapshots, str) else snapshots
    if not isinstance(values, (list, tuple)):
        values = (values,)
    resolved = tuple(ctx[item] if isinstance(item, str) else item for item in values)
    if not resolved:
        raise ValueError("snapshot merge requires at least one snapshot")
    return resolved


@block(
    id="merge_feature_snapshots", name="Merge Feature Snapshots", category="Statistics",
    description="Concatenate public feature snapshots without rerunning inference or an estimator.",
    params={"snapshots": {"type": "value", "default": []}, "save_as": {"type": "slot", "default": "feature_snapshot"}},
    requires=(), provides=("save_as",), placement=("top", "epoch"), stage="evaluate", ui_group="⑥ 后验与权重",
)
def merge_feature_snapshots(ctx: ScratchContext, snapshots: Any = (), save_as: str = "feature_snapshot") -> None:
    from ..native_stats import merge_feature_snapshots as merge
    ctx[save_as] = merge(_resolve_snapshot_values(ctx, snapshots))


@block(
    id="merge_posterior_snapshots", name="Merge Posterior Snapshots", category="Statistics",
    description="Concatenate public posterior snapshots without fitting a transition estimator.",
    params={"snapshots": {"type": "value", "default": []}, "save_as": {"type": "slot", "default": "posterior_snapshot"}},
    requires=(), provides=("save_as",), placement=("top", "epoch"), stage="evaluate", ui_group="⑥ 后验与权重",
)
def merge_posterior_snapshots(ctx: ScratchContext, snapshots: Any = (), save_as: str = "posterior_snapshot") -> None:
    from ..native_stats import merge_posterior_snapshots as merge
    ctx[save_as] = merge(_resolve_snapshot_values(ctx, snapshots))


@block(
    id="estimate_noisy_posterior", name="Estimate Noisy Posterior", category="Statistics",
    description="Estimate noisy-label posteriors from features and aligned labels/indices.",
    params={"features": {"type": "slot", "default": "posterior_features"}, "labels": {"type": "slot", "default": "posterior_targets"}, "indices": {"type": "slot", "default": "posterior_indices"}, "backend": {"type": "enum", "options": ["kde", "kliep"], "default": "kde"}, "bandwidth": {"type": "float", "default": 0.15, "min": 0.0}, "save_as": {"type": "slot", "default": "posterior_snapshot"}},
    requires=("features", "labels", "indices"), provides=("save_as", "posterior_probabilities", "posterior_indices", "posterior_targets"), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
)
def estimate_noisy_posterior(ctx: ScratchContext, features: str = "posterior_features", labels: str = "posterior_targets", indices: str = "posterior_indices", backend: str = "kde", bandwidth: float = 0.15, save_as: str = "posterior_snapshot") -> None:
    import torch
    from ..native_stats import build_binary_noisy_posterior_backend
    config: dict[str, Any] = {"name": str(backend), "bandwidth": float(bandwidth)}
    if str(backend).strip().lower() == "kliep":
        config.update({"max_centers": 24, "max_iterations": 200, "learning_rate": 0.02, "tolerance": 1.0e-7, "epsilon": 1.0e-12, "seed": int(ctx.get("seed", 1))})
    snapshot = build_binary_noisy_posterior_backend(config).fit_predict(ctx[features], ctx[labels], ctx[indices], dataset="synthetic_binary_2d", split="train")
    ctx[save_as] = snapshot
    ctx["posterior_probabilities"] = torch.as_tensor(snapshot.noisy_probabilities, dtype=torch.float32)
    ctx["posterior_indices"] = torch.as_tensor(snapshot.global_indices, dtype=torch.long)
    ctx["posterior_targets"] = torch.as_tensor(snapshot.noisy_targets, dtype=torch.long)


@block(
    id="estimate_raw_min_noise_rates", name="Estimate Raw-min Noise Rates", category="Statistics",
    description="Estimate binary class-dependent noise rates from the minimum noisy posterior by observed class.",
    params={"posterior": {"type": "slot", "default": "posterior_snapshot"}, "positive_as": {"type": "slot", "default": "rho_positive"}, "negative_as": {"type": "slot", "default": "rho_negative"}},
    requires=("posterior",), provides=("positive_as", "negative_as"), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
)
def estimate_raw_min_noise_rates(ctx: ScratchContext, posterior: str = "posterior_snapshot", positive_as: str = "rho_positive", negative_as: str = "rho_negative") -> None:
    from ..native_stats import PaperRawMinNoiseRateEstimator
    artifact = PaperRawMinNoiseRateEstimator().estimate(ctx[posterior])
    ctx[positive_as] = float(artifact.rho_positive)
    ctx[negative_as] = float(artifact.rho_negative)
    ctx["noise_rate_artifact"] = artifact


@block(
    id="estimate_class_prior", name="Estimate Class Prior", category="Statistics",
    description="Estimate a normalized class prior from integer labels or a snapshot.",
    params={"labels": {"type": "slot", "default": "labels"}, "num_classes": {"type": "int", "default": 10, "min": 2}, "save_as": {"type": "slot", "default": "class_prior"}},
    requires=("labels",), provides=("save_as",), placement=("batch", "top"), stage="setup", ui_group="⑥ 后验与权重",
)
def estimate_class_prior(ctx: ScratchContext, labels: str = "labels", num_classes: int = 10, save_as: str = "class_prior") -> None:
    torch = _torch()
    value = ctx[labels]
    if hasattr(value, "noisy_targets"):
        value = value.noisy_targets
    elif hasattr(value, "observed_targets"):
        value = value.observed_targets
    elif hasattr(value, "samples"):
        value = [sample.observed_target for sample in value.samples]
    elif isinstance(value, Mapping):
        value = value.get("observed_targets", value.get("noisy_targets", value.get("targets", value)))
    value = torch.as_tensor(value, dtype=torch.long).reshape(-1)
    if value.numel() == 0 or bool((value < 0).any()) or bool((value >= int(num_classes)).any()):
        raise ValueError("estimate_class_prior labels are outside the class range")
    counts = torch.bincount(value, minlength=int(num_classes)).to(torch.float32)
    ctx[save_as] = counts / counts.sum().clamp_min(1.0)


@block(
    id="align_by_stable_index", name="Align By Stable Index", category="Statistics",
    description="Reorder an artifact to a requested stable-index namespace without changing values.",
    params={"artifact": {"type": "slot", "default": "artifact"}, "indices": {"type": "slot", "default": "indices"}, "save_as": {"type": "slot", "default": "aligned_artifact"}},
    requires=("artifact", "indices"), provides=("save_as",), placement=("top", "batch"), stage="setup", ui_group="⑥ 后验与权重",
)
def align_by_stable_index(ctx: ScratchContext, artifact: str = "artifact", indices: str = "indices", save_as: str = "aligned_artifact") -> None:
    torch = _torch(); source = ctx[artifact]; requested = _as_indices(ctx[indices])
    source_indices = getattr(source, "global_indices", None)
    if source_indices is None and isinstance(source, dict):
        source_indices = source.get("global_indices", source.get("indices"))
    if source_indices is None:
        raise ValueError("align_by_stable_index requires artifact global_indices")
    source_indices = _as_indices(source_indices)
    order = torch.searchsorted(source_indices, requested)
    if bool((order >= source_indices.numel()).any()) or not bool(torch.equal(source_indices[order], requested)):
        raise KeyError("artifact does not cover requested stable indices")
    if isinstance(source, dict):
        result = dict(source)
        for key, value in source.items():
            if hasattr(value, "shape") and getattr(value, "shape", ()) and int(value.shape[0]) == source_indices.numel():
                result[key] = value[order]
        result["global_indices"] = requested
    else:
        try:
            import dataclasses
            fields = {field.name: getattr(source, field.name) for field in dataclasses.fields(source)}
            for key, value in list(fields.items()):
                if hasattr(value, "shape") and getattr(value, "shape", ()) and int(value.shape[0]) == source_indices.numel():
                    fields[key] = value[order.cpu().numpy() if hasattr(value, "__array__") else order]
            fields["global_indices"] = requested.cpu().numpy()
            result = type(source)(**fields)
        except Exception as exc:
            raise TypeError("artifact must be a mapping or dataclass with aligned fields") from exc
    ctx[save_as] = result


@block(
    id="classwise_feature_statistics", name="Classwise Feature Statistics", category="Statistics",
    description="Compute per-class counts, means and a shared covariance from features and labels.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "num_classes": {"type": "int", "default": 10, "min": 2}, "save_as": {"type": "slot", "default": "feature_statistics"}},
    requires=("features", "labels"), provides=("save_as",), placement=("top", "batch"), stage="setup", ui_group="⑥ 后验与权重",
)
def classwise_feature_statistics(ctx: ScratchContext, features: str = "features", labels: str = "labels", num_classes: int = 10, save_as: str = "feature_statistics") -> None:
    torch = _torch(); values = torch.as_tensor(ctx[features]); targets = torch.as_tensor(ctx[labels]).long().reshape(-1)
    if values.ndim != 2 or values.shape[0] != targets.numel():
        raise ValueError("classwise_feature_statistics expects aligned [N,D] features and labels")
    counts = torch.bincount(targets, minlength=int(num_classes)).to(values.dtype)
    means = torch.zeros((int(num_classes), values.shape[1]), dtype=values.dtype, device=values.device)
    for cls in range(int(num_classes)):
        selected = values[targets == cls]
        if selected.numel():
            means[cls] = selected.mean(0)
    centered = values - means[targets]
    covariance = centered.T @ centered / max(int(values.shape[0]) - 1, 1)
    ctx[save_as] = SimpleNamespace(counts=counts, means=means, covariance=covariance)


@block(
    id="recover_clean_prior", name="Recover Clean Prior", category="Statistics",
    description="Solve the forward transition equation for a nonnegative clean class prior.",
    params={"noisy_prior": {"type": "slot", "default": "noisy_prior"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "clean_prior"}},
    requires=("noisy_prior", "transition"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
)
def recover_clean_prior(ctx: ScratchContext, noisy_prior: str = "noisy_prior", transition: str = "transition", save_as: str = "clean_prior") -> None:
    torch = _torch(); observed = torch.as_tensor(ctx[noisy_prior]); matrix = torch.as_tensor(ctx[transition], device=observed.device, dtype=observed.dtype)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or observed.numel() != matrix.shape[0]:
        raise ValueError("recover_clean_prior expects a square transition and matching prior")
    clean = torch.linalg.lstsq(matrix.T, observed.reshape(-1, 1)).solution[:, 0].clamp_min(0)
    ctx[save_as] = clean / clean.sum().clamp_min(torch.finfo(clean.dtype).tiny)


@block(
    id="matrix_pseudoinverse", name="Matrix Pseudoinverse", category="Statistics",
    description="Compute the Moore-Penrose pseudoinverse of a matrix.",
    params={"matrix": {"type": "slot", "default": "matrix"}, "save_as": {"type": "slot", "default": "pseudoinverse"}},
    requires=("matrix",), provides=("save_as",), placement=("top", "batch"), stage="setup", ui_group="⑥ 后验与权重",
)
def matrix_pseudoinverse(ctx: ScratchContext, matrix: str = "matrix", save_as: str = "pseudoinverse") -> None:
    ctx[save_as] = _torch().linalg.pinv(ctx[matrix])


@block(
    id="fit_shared_covariance_gda", name="Fit Shared-covariance GDA", category="Statistics",
    description="Fit one shared-covariance GDA estimator per recovered class-statistics mapping.",
    params={"statistics": {"type": "slot", "default": "statistics"}, "covariance_ridge": {"type": "float", "default": 0.1, "min": 0.0}, "save_as": {"type": "slot", "default": "gda"}},
    requires=("statistics",), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
)
def fit_shared_covariance_gda(ctx: ScratchContext, statistics: str = "statistics", covariance_ridge: float = 0.1, save_as: str = "gda") -> None:
    """Fit GDA estimators from already materialized class statistics.

    Statistics recovery and estimator construction are deliberately separate:
    this block never collects a loader snapshot or reinterprets raw labels.
    A mapping produces one estimator per feature layer, which is the contract
    consumed by ``predict_gda_layers`` and the public ensemble operations.
    """
    from ..native_stats import fit_gda_layers
    source = ctx[statistics]
    if not isinstance(source, Mapping) and not hasattr(source, "means"):
        raise TypeError("fit_shared_covariance_gda expects recovered class statistics")
    ctx[save_as] = fit_gda_layers(source, covariance_ridge=float(covariance_ridge))


@block(
    id="predict_gda_layers", name="Predict GDA Layers", category="Statistics",
    description="Evaluate fitted GDA estimators on aligned feature snapshots and publish [N,E,C] predictions.",
    params={"estimators": {"type": "slot", "default": "gda"}, "snapshots": {"type": "slot", "default": "feature_snapshots"}, "save_as": {"type": "slot", "default": "predictions"}, "targets_as": {"type": "slot", "default": "targets"}},
    requires=("estimators", "snapshots"), provides=("save_as", "targets_as"), placement=("top", "epoch"), stage="evaluate", ui_group="⑥ 后验与权重",
)
def predict_gda_layers(ctx: ScratchContext, estimators: str = "gda", snapshots: str = "feature_snapshots", save_as: str = "predictions", targets_as: str = "targets") -> None:
    import numpy as np
    estimators_value = ctx[estimators]
    snapshots_value = ctx[snapshots]
    if not isinstance(estimators_value, (list, tuple)) or not isinstance(snapshots_value, (list, tuple)):
        raise TypeError("predict_gda_layers expects estimator and snapshot sequences")
    if len(estimators_value) != len(snapshots_value) or not estimators_value:
        raise ValueError("predict_gda_layers estimator/snapshot counts must match")
    reference = snapshots_value[0]
    reference_indices = np.asarray(reference.global_indices, dtype=np.int64)
    reference_targets = np.asarray(reference.noisy_targets, dtype=np.int64)
    predictions = []
    for estimator, snapshot in zip(estimators_value, snapshots_value):
        indices = np.asarray(snapshot.global_indices, dtype=np.int64)
        targets = np.asarray(snapshot.noisy_targets, dtype=np.int64)
        if not np.array_equal(indices, reference_indices) or not np.array_equal(targets, reference_targets):
            raise ValueError("predict_gda_layers snapshots must share stable indices and targets")
        if not hasattr(estimator, "posterior"):
            raise TypeError("GDA estimator must expose posterior(features)")
        predictions.append(np.asarray(estimator.posterior(snapshot.features), dtype=np.float32))
    values = _torch().as_tensor(np.stack(predictions, axis=1), dtype=_torch().float32)
    ctx[save_as] = values
    ctx[targets_as] = _torch().as_tensor(reference_targets, dtype=_torch().long)


@block(
    id="fit_simplex_ensemble_weights", name="Fit Simplex Ensemble Weights", category="Statistics",
    description="Optimize nonnegative simplex weights against validation negative log-likelihood.",
    params={"predictions": {"type": "slot", "default": "predictions"}, "targets": {"type": "slot", "default": "targets"}, "epochs": {"type": "int", "default": 5, "min": 1}, "learning_rate": {"type": "float", "default": 0.05, "min": 0.0}, "save_as": {"type": "slot", "default": "ensemble_weights"}},
    requires=("predictions", "targets"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
)
def fit_simplex_ensemble_weights(ctx: ScratchContext, predictions: str = "predictions", targets: str = "targets", epochs: int = 5, learning_rate: float = 0.05, save_as: str = "ensemble_weights") -> None:
    torch = _torch(); values = torch.as_tensor(ctx[predictions]); target = torch.as_tensor(ctx[targets]).long().reshape(-1)
    if values.ndim != 3 or values.shape[0] != target.numel():
        raise ValueError("fit_simplex_ensemble_weights expects [N,E,C] probabilities/logits")
    probabilities = values.softmax(-1) if values.min() < 0 or values.max() > 1 else values
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError("fit_simplex_ensemble_weights predictions must be finite probabilities or logits")
    raw = torch.zeros(int(probabilities.shape[1]), dtype=probabilities.dtype, device=probabilities.device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=float(learning_rate))
    for _ in range(int(epochs)):
        weights = torch.softmax(raw, dim=0)
        mixture = (probabilities * weights[None, :, None]).sum(dim=1)
        loss = -mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log().gather(1, target[:, None]).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    ctx[save_as] = torch.softmax(raw.detach(), dim=0)


@block(
    id="evaluate_weighted_ensemble", name="Evaluate Weighted Ensemble", category="Evaluation",
    description="Evaluate a weighted ensemble of class predictions.",
    params={"predictions": {"type": "slot", "default": "predictions"}, "weights": {"type": "slot", "default": "ensemble_weights"}, "targets": {"type": "slot", "default": "targets"}, "save_as": {"type": "slot", "default": "accuracy"}},
    requires=("predictions", "weights", "targets"), provides=("save_as",), placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
)
def evaluate_weighted_ensemble(ctx: ScratchContext, predictions: str = "predictions", weights: str = "ensemble_weights", targets: str = "targets", save_as: str = "accuracy") -> None:
    torch = _torch(); values = torch.as_tensor(ctx[predictions]); raw_weights = ctx[weights];
    if isinstance(raw_weights, Mapping) and "weights" in raw_weights:
        raw_weights = raw_weights["weights"]
    factors = torch.as_tensor(raw_weights, device=values.device, dtype=values.dtype); target = torch.as_tensor(ctx[targets], device=values.device).long()
    if values.ndim != 3 or values.shape[1] != factors.numel() or values.shape[0] != target.numel():
        raise ValueError("evaluate_weighted_ensemble input shapes are inconsistent")
    probabilities = values.softmax(-1) if values.min() < 0 or values.max() > 1 else values
    ctx[save_as] = float((probabilities.mul(factors[None, :, None]).sum(1).argmax(-1) == target).float().mean())


class _SigmoidOffDiagonalTransition:
    def __init__(self, classes: int, device: Any = "cpu", initial_weight: float = 0.0):
        torch = _torch(); self.num_classes = int(classes); self.parameter = torch.nn.Parameter(torch.full((self.num_classes, self.num_classes), float(initial_weight), device=device))
    def parameters(self):
        return [self.parameter]
    def state_dict(self):
        return {"parameter": self.parameter.detach().clone()}
    def load_state_dict(self, state):
        if isinstance(state, dict) and "parameter" in state:
            self.parameter.data.copy_(state["parameter"].to(self.parameter))
        return self
    def train(self, mode: bool = True):
        return self
    def eval(self):
        return self
    def matrix(self, dtype=None, device=None):
        torch = _torch(); values = torch.sigmoid(self.parameter)
        eye = torch.eye(self.num_classes, device=values.device, dtype=values.dtype)
        values = values * (1.0 - eye) + eye
        values = values / values.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)
        if device is not None: values = values.to(device)
        return values.to(dtype=dtype) if dtype is not None else values
    def to(self, device):
        self.parameter.data = self.parameter.data.to(device); return self


@block(
    id="create_sigmoid_offdiagonal_transition", name="Create Sigmoid Off-diagonal Transition", category="Transition",
    description="Create a trainable row-stochastic transition with sigmoid off-diagonal entries.",
    params={"num_classes": {"type": "int", "default": 10, "min": 2}, "initial_weight": {"type": "float", "default": 0.0}, "seed": {"type": "int", "default": 0, "min": 0}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "transition"}},
    requires=(), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
)
def create_sigmoid_offdiagonal_transition(ctx: ScratchContext, num_classes: int = 10, initial_weight: float = 0.0, seed: int = 0, device: str = "device", save_as: str = "transition") -> None:
    # ``seed`` is part of the public construction contract for reproducible
    # estimators.  The parameterization itself is deterministic, so seeding is
    # deliberately scoped and does not mutate the caller's global RNG state.
    ctx[save_as] = _SigmoidOffDiagonalTransition(int(num_classes), ctx.get(device, "cpu"), initial_weight=float(initial_weight))


class _AdditiveTransitionRevision:
    def __init__(self, transition: Any, *, project_nonnegative: bool = True, normalize: bool = True):
        torch = _torch(); self.base = transition.matrix() if hasattr(transition, "matrix") else torch.as_tensor(transition); self.delta = torch.nn.Parameter(torch.zeros_like(self.base))
        self.project_nonnegative = bool(project_nonnegative)
        self.normalize = bool(normalize)
    def parameters(self):
        return [self.delta]
    def state_dict(self):
        return {"base": self.base.detach().clone(), "delta": self.delta.detach().clone(), "project_nonnegative": self.project_nonnegative, "normalize": self.normalize}
    def load_state_dict(self, state):
        if isinstance(state, dict) and "delta" in state:
            self.delta.data.copy_(state["delta"].to(self.delta))
        if isinstance(state, dict) and "base" in state:
            self.base = state["base"].to(self.delta)
        if isinstance(state, dict):
            if "project_nonnegative" in state: self.project_nonnegative = bool(state["project_nonnegative"])
            if "normalize" in state: self.normalize = bool(state["normalize"])
        return self
    def train(self, mode: bool = True):
        return self
    def eval(self):
        return self
    def to(self, device):
        self.base = self.base.to(device)
        self.delta.data = self.delta.data.to(device)
        return self
    def matrix(self, dtype=None, device=None):
        torch = _torch(); value = self.base.to(self.delta) + self.delta
        if self.project_nonnegative:
            value = value.clamp_min(0)
        if self.normalize:
            value = value / value.sum(-1, keepdim=True).clamp_min(torch.finfo(value.dtype).tiny)
        if device is not None: value = value.to(device)
        return value.to(dtype=dtype) if dtype is not None else value
    def __call__(self):
        return self.matrix()


@block(
    id="create_additive_transition_revision", name="Create Additive Transition Revision", category="Transition",
    description="Create a zero-initialized additive transition revision around a fixed transition.",
    params={"transition": {"type": "slot", "default": "transition"}, "project_nonnegative": {"type": "bool", "default": True}, "normalize": {"type": "bool", "default": True}, "save_as": {"type": "slot", "default": "revision"}},
    requires=("transition",), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
)
def create_additive_transition_revision(ctx: ScratchContext, transition: str = "transition", project_nonnegative: bool = True, normalize: bool = True, save_as: str = "revision") -> None:
    ctx[save_as] = _AdditiveTransitionRevision(ctx[transition], project_nonnegative=bool(project_nonnegative), normalize=bool(normalize))


def _batch(batch):
    if isinstance(batch, dict):
        return batch.get("input", batch.get("inputs", batch.get("images"))), batch.get("target", batch.get("targets", batch.get("labels")))
    return batch[0], batch[1]


@block(
    id="evaluate_transition_accuracy", name="Evaluate Transition Accuracy", category="Evaluation",
    description="Evaluate observed-label accuracy or noisy-transition NLL after an explicit transition matrix.",
    params={"model": {"type": "slot", "default": "model"}, "transition": {"type": "slot", "default": "transition"}, "loader": {"type": "slot", "default": "test_loader"}, "device": {"type": "slot", "default": "device"}, "metric": {"type": "enum", "options": ["accuracy", "loss"], "default": "accuracy"}, "denominator_floor": {"type": "float", "default": 1.0e-12, "min": 0.0}, "revision": {"type": "bool", "default": False}, "final": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "accuracy"}},
    requires=("model", "transition", "loader"), provides=("save_as", "validation_accuracy", "validation_loss", "metrics"), placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
)
def evaluate_transition_accuracy(ctx: ScratchContext, model: str = "model", transition: str = "transition", loader: str = "test_loader", device: str = "device", metric: str = "accuracy", denominator_floor: float = 1.0e-12, revision: bool = False, final: bool = False, save_as: str = "accuracy") -> None:
    torch = _torch(); network = ctx[model]; matrix_source = ctx[transition]
    if bool(final) and bool((ctx.get("_runtime_limits") or {}).get("skip_final_test")) and str(loader) == "test_loader":
        ctx[save_as] = float("nan"); ctx["transition_test_skipped"] = True; return
    target_device = ctx.get(device, next(network.parameters()).device)
    shared_matrix = None if hasattr(matrix_source, "transition_for") else (matrix_source.matrix() if hasattr(matrix_source, "matrix") else (matrix_source() if callable(matrix_source) else matrix_source))
    if shared_matrix is not None:
        shared_matrix = torch.as_tensor(shared_matrix, device=target_device)
    was_training = network.training; network.eval(); correct = total = 0; loss_sum = 0.0
    with torch.no_grad():
        for batch in ctx[loader]:
            inputs, labels = _batch(batch); logits = network(inputs.to(target_device)); probabilities = torch.softmax(logits, -1)
            if hasattr(matrix_source, "transition_for"):
                batch_indices = batch.get("index", batch.get("indices")) if isinstance(batch, dict) else batch[2]
                matrix = matrix_source.transition_for(None, batch_indices, device=probabilities.device, dtype=probabilities.dtype)
            else:
                matrix = shared_matrix.to(probabilities)
            if bool(revision):
                revision_head = getattr(network, "T_revision", None)
                if revision_head is None or not hasattr(revision_head, "weight"):
                    raise AttributeError("revision evaluation requires model.T_revision.weight")
                matrix = matrix + revision_head.weight.to(probabilities)
                matrix = matrix.abs()
                matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(matrix.dtype).tiny)
            observed = torch.bmm(probabilities.unsqueeze(1), matrix).squeeze(1) if matrix.ndim == 3 else probabilities @ matrix
            labels = labels.to(probabilities.device).long(); correct += int(observed.argmax(-1).eq(labels).sum()); total += int(labels.numel())
            target_probability = observed.gather(1, labels[:, None]).squeeze(1)
            if bool((target_probability <= float(denominator_floor)).any()) or not bool(torch.isfinite(target_probability).all()):
                raise ValueError("evaluate_transition_accuracy encountered an invalid observed-label probability")
            loss_sum += float((-target_probability.log()).sum().item())
    network.train(was_training)
    if total == 0:
        raise ValueError("evaluate_transition_accuracy loader is empty")
    accuracy = float(correct / total)
    loss = float(loss_sum / total)
    ctx["validation_accuracy"] = accuracy
    ctx["validation_loss"] = loss
    ctx[save_as] = loss if str(metric) == "loss" else accuracy
    ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "validation_loss": loss, "validation_accuracy": accuracy})


@block(
    id="evaluate_binary_accuracy", name="Evaluate Binary Accuracy", category="Evaluation",
    description="Evaluate a scalar or two-logit binary classifier with an explicit zero threshold.",
    params={"model": {"type": "slot", "default": "model"}, "loader": {"type": "slot", "default": "test_loader"}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "accuracy"}, "final": {"type": "bool", "default": False}, "target_source": {"type": "enum", "options": ["observed", "clean"], "default": "observed"}},
    requires=("model", "loader"), provides=("save_as",), placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
)
def evaluate_binary_accuracy(ctx: ScratchContext, model: str = "model", loader: str = "test_loader", device: str = "device", save_as: str = "accuracy", final: bool = False, target_source: str = "observed") -> None:
    torch = _torch(); network = ctx[model]; target_device = ctx.get(device, next(network.parameters()).device); was_training = network.training; network.eval(); correct = total = 0
    if bool(final) and bool((ctx.get("_runtime_limits") or {}).get("skip_final_test")) and str(loader) == "test_loader":
        ctx[save_as] = float("nan"); ctx["binary_test_skipped"] = True; return
    with torch.no_grad():
        for batch in ctx[loader]:
            inputs, labels = _batch(batch)
            if str(target_source) == "clean":
                labels = batch.get("clean_targets") if isinstance(batch, dict) else None
                if labels is None:
                    raise ValueError("clean evaluation requires complete clean_targets")
            logits = network(inputs.to(target_device)); score = logits[:, 0] if logits.ndim > 1 and logits.shape[1] == 1 else logits[:, 1] - logits[:, 0] if logits.ndim > 1 else logits.reshape(-1); prediction = (score >= 0).long(); labels = labels.to(target_device).long(); correct += int(prediction.eq(labels).sum()); total += int(labels.numel())
    network.train(was_training); ctx[save_as] = float(correct / total) if total else 0.0
    ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "binary_accuracy": ctx[save_as]})
