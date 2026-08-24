"""Independent paper-level noisy-label operations.

These blocks intentionally operate on Context slots instead of calling the
legacy algorithm package.  They are small, semantic operations that can be
composed in a recipe and are useful for shape/value smoke checks.
"""

from __future__ import annotations

from typing import Any

from ...context import ScratchContext
from ...registry import block


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional train extra
        raise RuntimeError("Paper blocks require PyTorch; install the `train` extra.") from exc
    return torch, F


def _row_normalize(value):
    torch, _ = _torch()
    return value / value.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(value.dtype).tiny)


def _ce(logits, labels):
    _, F = _torch()
    return F.cross_entropy(logits, labels.long(), reduction="none")


@block(
    id="prepare_paper_batch",
    name="Prepare Paper Smoke Batch",
    category="Paper Specific",
    description="Create deterministic logits, labels, features, and a transition matrix for recipe shape checks.",
    params={
        "samples": {"type": "int", "default": 16, "min": 2},
        "features": {"type": "int", "default": 8, "min": 2},
        "classes": {"type": "int", "default": 3, "min": 2},
    },
    provides=("labels", "logits", "logits_a", "logits_b", "features", "probabilities", "loss_per_sample", "transition", "transition_a", "transition_b"),
)
def prepare_paper_batch(ctx: ScratchContext, samples: int = 16, features: int = 8, classes: int = 3) -> None:
    torch, F = _torch()
    generator = torch.Generator().manual_seed(int(ctx.get("seed", 1)))
    labels = torch.randint(classes, (samples,), generator=generator)
    logits = torch.randn(samples, classes, generator=generator, requires_grad=True)
    logits_a = torch.randn(samples, classes, generator=generator, requires_grad=True)
    logits_b = torch.randn(samples, classes, generator=generator, requires_grad=True)
    features_value = torch.randn(samples, features, generator=generator)
    transition = torch.rand(classes, classes, generator=generator) + 0.2
    transition = _row_normalize(transition)
    ctx.update({
        "labels": labels,
        "logits": logits,
        "logits_a": logits_a,
        "logits_b": logits_b,
        "features": features_value,
        "probabilities": F.softmax(logits, dim=-1),
        "loss_per_sample": _ce(logits, labels),
        "transition": transition,
        "transition_a": transition,
        "transition_b": torch.eye(classes),
        "num_classes": int(classes),
    })


@block(
    id="pdl_instance_transition",
    name="PDL: Instance Transition",
    category="Paper Specific",
    description="Build feature-dependent row-stochastic transition matrices and corrected per-sample risk.",
    params={"logits": {"type": "slot", "default": "logits"}, "features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "pdl_loss"}},
    requires=("logits", "features", "labels"),
    provides=("save_as", "instance_transition"),
)
def pdl_instance_transition(ctx: ScratchContext, logits: str = "logits", features: str = "features", labels: str = "labels", save_as: str = "pdl_loss") -> None:
    torch, _ = _torch()
    classes = ctx[logits].shape[-1]
    scale = torch.sigmoid(ctx[features].mean(dim=-1, keepdim=True))
    base = torch.eye(classes, device=ctx[logits].device).expand(ctx[logits].shape[0], -1, -1).clone()
    base = base * (0.7 + 0.2 * scale.unsqueeze(-1)) + (1.0 - base) * (0.3 - 0.2 * scale.unsqueeze(-1)) / (classes - 1)
    base = _row_normalize(base)
    observed = torch.bmm(torch.softmax(ctx[logits], -1).unsqueeze(1), base).squeeze(1)
    ctx["instance_transition"] = base
    ctx[save_as] = -torch.log(observed.gather(1, ctx[labels].long()[:, None]).squeeze(1).clamp_min(1e-12))


def _pdl_subset_snapshot(snapshot, indices):
    import numpy as np
    from lnl_toolbox.noise.estimators import PosteriorSnapshot
    requested = np.asarray(indices, dtype=np.int64)
    sorted_indices = np.sort(requested, kind="stable")
    positions = np.searchsorted(snapshot.global_indices, sorted_indices)
    if np.any(positions >= snapshot.global_indices.size) or not np.array_equal(
        snapshot.global_indices[positions], sorted_indices
    ):
        raise KeyError("PDL snapshot does not cover requested split indices")
    return PosteriorSnapshot(
        snapshot.noisy_probabilities[positions],
        snapshot.noisy_targets[positions],
        sorted_indices,
        dataset=snapshot.dataset,
        split=snapshot.split,
    )


def _pdl_subset_features(snapshot, indices):
    import numpy as np
    from lnl_toolbox.training.snapshots import FeatureSnapshot
    requested = np.asarray(indices, dtype=np.int64)
    sorted_indices = np.sort(requested, kind="stable")
    positions = np.searchsorted(snapshot.global_indices, sorted_indices)
    if np.any(positions >= snapshot.global_indices.size) or not np.array_equal(
        snapshot.global_indices[positions], sorted_indices
    ):
        raise KeyError("PDL feature snapshot does not cover requested split indices")
    return FeatureSnapshot(
        snapshot.features[positions],
        snapshot.noisy_targets[positions],
        sorted_indices,
        dataset=snapshot.dataset,
        split=snapshot.split,
    )


@block(
    id="snapshot_pdl_features",
    name="Snapshot PDL Features and Posteriors",
    category="Transition",
    description="Collect stable-index feature and noisy-posterior snapshots for PDL train and noisy-validation splits.",
    params={
        "model": {"type": "slot", "default": "model"},
        "prepared_data": {"type": "slot", "default": "prepared_data"},
        "device": {"type": "slot", "default": "device"},
    },
    requires=("model", "prepared_data", "device"),
    provides=("pdl_train_features", "pdl_train_posteriors", "pdl_validation_features", "pdl_validation_posteriors", "pdl_representation_features", "pdl_representation_posteriors"),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="q(x)=softmax(f_theta*(x)); h(x)=representation(f_theta*,x)",
    formula_ref="PDL warm-up feature/posterior snapshot lifecycle",
    paper="Part-dependent Label Noise",
)
def snapshot_pdl_features(
    ctx: ScratchContext,
    model: str = "model",
    prepared_data: str = "prepared_data",
    device: str = "device",
) -> None:
    import itertools
    import numpy as np
    from lnl_toolbox.data import DataRole
    from lnl_toolbox.training.snapshots import collect_feature_snapshot, collect_posterior_snapshot

    prepared = ctx[prepared_data]
    manifest = prepared.manifest
    if manifest is None:
        raise ValueError("PDL snapshot requires a persisted noise manifest")
    noisy_map = {int(index): int(target) for index, target in zip(manifest.global_indices, manifest.noisy_targets)}
    train_indices = np.asarray(prepared.train_indices, dtype=np.int64)
    validation_indices = np.asarray(prepared.validation_indices, dtype=np.int64)
    union_indices = np.concatenate([train_indices, validation_indices])
    union = prepared.dynamic_dataset(union_indices, targets_by_index=noisy_map, training=False)
    loader = prepared.loader_for_dataset(union, shuffle=False)
    runtime_limits = ctx.get("_runtime_limits") or {}
    # A runtime cap may shorten training loops, but the artifact must still cover
    # every train/validation index used by the formal corrected lifecycle.
    if runtime_limits.get("snapshot_batches") is not None:
        loader = itertools.islice(loader, int(runtime_limits["snapshot_batches"]))
    posterior = collect_posterior_snapshot(
        ctx[model], loader, ctx[device], dataset="cifar10", split="train"
    )
    loader = prepared.loader_for_dataset(union, shuffle=False)
    if runtime_limits.get("snapshot_batches") is not None:
        loader = itertools.islice(loader, int(runtime_limits["snapshot_batches"]))
    features = collect_feature_snapshot(
        ctx[model], loader, ctx[device], dataset="cifar10", split="train",
        feature_extractor=lambda network, inputs: network.forward_with_features(inputs).features,
    )
    if not np.array_equal(posterior.global_indices, features.global_indices):
        raise ValueError("PDL feature and posterior snapshots are not index aligned")
    ctx["pdl_train_posteriors"] = _pdl_subset_snapshot(posterior, train_indices)
    ctx["pdl_validation_posteriors"] = _pdl_subset_snapshot(posterior, validation_indices)
    ctx["pdl_train_features"] = _pdl_subset_features(features, train_indices)
    ctx["pdl_validation_features"] = _pdl_subset_features(features, validation_indices)
    ctx["pdl_representation_features"] = features
    ctx["pdl_representation_posteriors"] = posterior


@block(
    id="pdl_fit_part_representation",
    name="PDL Part Representation",
    category="Transition",
    description="Fit PDL's multiplicative-update nonnegative part representation once on train plus noisy-validation features.",
    params={
        "features": {"type": "slot", "default": "pdl_representation_features"},
        "num_parts": {"type": "int", "default": 20, "min": 1},
        "iterations": {"type": "int", "default": 10, "min": 1},
        "error_tolerance": {"type": "float", "default": 1.0e-5, "min": 0.0},
        "representation_seed": {"type": "int", "default": 1, "min": 0},
        "official_raw": {"type": "bool", "default": True},
        "parts_as": {"type": "slot", "default": "pdl_parts"},
        "coefficients_as": {"type": "slot", "default": "pdl_coefficients"},
        "indices_as": {"type": "slot", "default": "pdl_representation_indices"},
    },
    requires=("features",),
    provides=("parts_as", "coefficients_as", "indices_as"),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="H,W = argmin_{H,W>=0} ||X-WH||^2; normalize W rows",
    formula_ref="PDL official train_m multiplicative updates",
    paper="Part-dependent Label Noise",
)
def pdl_fit_part_representation(
    ctx: ScratchContext,
    features: str = "pdl_representation_features",
    num_parts: int = 20,
    iterations: int = 10,
    error_tolerance: float = 1.0e-5,
    representation_seed: int = 1,
    official_raw: bool = True,
    parts_as: str = "pdl_parts",
    coefficients_as: str = "pdl_coefficients",
    indices_as: str = "pdl_representation_indices",
) -> None:
    from lnl_toolbox.noise.pdl import fit_part_representation
    snapshot = ctx[features]
    seed = None if bool(official_raw) else int(representation_seed)
    parts, coefficients = fit_part_representation(
        snapshot.features, int(num_parts), seed=seed,
        iterations=int(iterations), error_tolerance=float(error_tolerance),
    )
    ctx[parts_as] = parts
    ctx[coefficients_as] = coefficients
    ctx[indices_as] = snapshot.global_indices.copy()


@block(
    id="pdl_select_anchor_candidates",
    name="PDL Anchor Candidates",
    category="Transition",
    description="Select PDL high-posterior anchor samples independently for train and noisy validation.",
    params={
        "train_posterior": {"type": "slot", "default": "pdl_train_posteriors"},
        "validation_posterior": {"type": "slot", "default": "pdl_validation_posteriors"},
        "percentages": {"type": "value", "default": [97.0, 97.1052631579, 97.2105263158, 97.3157894737, 97.4210526316, 97.5263157895, 97.6315789474, 97.7368421053, 97.8421052632, 97.9473684211, 98.0526315789, 98.1578947368, 98.2631578947, 98.3684210526, 98.4736842105, 98.5789473684, 98.6842105263, 98.7894736842, 98.8947368421, 99.0]},
        "train_as": {"type": "slot", "default": "pdl_train_anchor_positions"},
        "validation_as": {"type": "slot", "default": "pdl_validation_anchor_positions"},
    },
    requires=("train_posterior", "validation_posterior"),
    provides=("train_as", "validation_as"),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="a_c(r)=argmax_i h_i 1[q_i(c)>=percentile_r(q(c))]",
    formula_ref="PDL official tools.fit(filter_outlier=True)",
    paper="Part-dependent Label Noise",
)
def pdl_select_anchor_candidates(
    ctx: ScratchContext,
    train_posterior: str = "pdl_train_posteriors",
    validation_posterior: str = "pdl_validation_posteriors",
    percentages: list[float] | tuple[float, ...] = (),
    train_as: str = "pdl_train_anchor_positions",
    validation_as: str = "pdl_validation_anchor_positions",
) -> None:
    from lnl_toolbox.noise.pdl import select_pdl_anchor_candidates
    levels = list(percentages) if percentages else list(__import__("numpy").linspace(97.0, 99.0, 20))
    ctx[train_as] = select_pdl_anchor_candidates(ctx[train_posterior].noisy_probabilities, levels)
    ctx[validation_as] = select_pdl_anchor_candidates(ctx[validation_posterior].noisy_probabilities, levels)


@block(
    id="pdl_fit_basis_matrices",
    name="PDL Basis Matrices",
    category="Transition",
    description="Fit PDL's class-conditioned basis transition matrices with the shared Adam lifecycle.",
    params={
        "coefficients": {"type": "slot", "default": "pdl_coefficients"},
        "representation_indices": {"type": "slot", "default": "pdl_representation_indices"},
        "train_posterior": {"type": "slot", "default": "pdl_train_posteriors"},
        "validation_posterior": {"type": "slot", "default": "pdl_validation_posteriors"},
        "train_anchors": {"type": "slot", "default": "pdl_train_anchor_positions"},
        "validation_anchors": {"type": "slot", "default": "pdl_validation_anchor_positions"},
        "basis_epochs": {"type": "int", "default": 1500, "min": 1},
        "basis_learning_rate": {"type": "float", "default": 0.001, "min": 0.0},
        "basis_loss_threshold": {"type": "float", "default": 0.02, "min": 0.0},
        "representation_seed": {"type": "int", "default": 1, "min": 0},
        "train_as": {"type": "slot", "default": "pdl_train_basis"},
        "validation_as": {"type": "slot", "default": "pdl_validation_basis"},
    },
    requires=("coefficients", "representation_indices", "train_posterior", "validation_posterior", "train_anchors", "validation_anchors"),
    provides=("train_as", "validation_as"),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="M_c=argmin_M sum_r ||W_{a_c(r)} M - q_{a_c(r)}||^2; normalize |M| rows",
    formula_ref="PDL official Matrix_optimize basis_matrix_optimize",
    paper="Part-dependent Label Noise",
)
def pdl_fit_basis_matrices(
    ctx: ScratchContext,
    coefficients: str = "pdl_coefficients",
    representation_indices: str = "pdl_representation_indices",
    train_posterior: str = "pdl_train_posteriors",
    validation_posterior: str = "pdl_validation_posteriors",
    train_anchors: str = "pdl_train_anchor_positions",
    validation_anchors: str = "pdl_validation_anchor_positions",
    basis_epochs: int = 1500,
    basis_learning_rate: float = 0.001,
    basis_loss_threshold: float = 0.02,
    representation_seed: int = 1,
    train_as: str = "pdl_train_basis",
    validation_as: str = "pdl_validation_basis",
) -> None:
    import numpy as np
    from lnl_toolbox.noise.pdl import fit_pdl_basis_matrices_pair
    coeff = ctx[coefficients]
    rep_indices = np.asarray(ctx[representation_indices], dtype=np.int64)
    train = ctx[train_posterior]
    validation = ctx[validation_posterior]
    train_positions = np.searchsorted(rep_indices, train.global_indices)
    validation_positions = np.searchsorted(rep_indices, validation.global_indices)
    if np.any(train_positions >= rep_indices.size) or np.any(validation_positions >= rep_indices.size):
        raise KeyError("PDL representation does not cover train/validation snapshots")
    limits = ctx.get("_runtime_limits") or {}
    effective_epochs = int(basis_epochs)
    if limits:
        effective_epochs = min(effective_epochs, max(1, int(limits.get("max_epochs", 1))))
    train_basis, validation_basis = fit_pdl_basis_matrices_pair(
        coeff[train_positions][ctx[train_anchors]], train.noisy_probabilities[ctx[train_anchors]],
        coeff[validation_positions][ctx[validation_anchors]], validation.noisy_probabilities[ctx[validation_anchors]],
        epochs=effective_epochs, learning_rate=float(basis_learning_rate),
        loss_threshold=float(basis_loss_threshold), seed=int(representation_seed), official_raw=True,
    )
    ctx[train_as] = train_basis
    ctx[validation_as] = validation_basis


@block(
    id="pdl_estimate_instance_transition",
    name="PDL Instance Transition",
    category="Transition",
    description="Compose PDL part coefficients with fitted basis matrices into stable-index train and validation transition artifacts.",
    params={
        "parts": {"type": "slot", "default": "pdl_parts"},
        "coefficients": {"type": "slot", "default": "pdl_coefficients"},
        "representation_indices": {"type": "slot", "default": "pdl_representation_indices"},
        "train_features": {"type": "slot", "default": "pdl_train_features"},
        "train_posterior": {"type": "slot", "default": "pdl_train_posteriors"},
        "validation_features": {"type": "slot", "default": "pdl_validation_features"},
        "validation_posterior": {"type": "slot", "default": "pdl_validation_posteriors"},
        "train_basis": {"type": "slot", "default": "pdl_train_basis"},
        "validation_basis": {"type": "slot", "default": "pdl_validation_basis"},
        "num_parts": {"type": "int", "default": 20, "min": 1},
        "representation_seed": {"type": "int", "default": 1, "min": 0},
        "train_as": {"type": "slot", "default": "pdl_transition"},
        "validation_as": {"type": "slot", "default": "pdl_validation_transition"},
        "revision_validation_as": {"type": "slot", "default": "pdl_revision_validation_transition"},
    },
    requires=("parts", "coefficients", "representation_indices", "train_features", "train_posterior", "validation_features", "validation_posterior", "train_basis", "validation_basis"),
    provides=("train_as", "validation_as", "revision_validation_as"),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="T(x)=sum_r beta_r(x) M_r",
    formula_ref="PDL part-dependent instance transition composition",
    paper="Part-dependent Label Noise",
)
def pdl_estimate_instance_transition(
    ctx: ScratchContext,
    parts: str = "pdl_parts",
    coefficients: str = "pdl_coefficients",
    representation_indices: str = "pdl_representation_indices",
    train_features: str = "pdl_train_features",
    train_posterior: str = "pdl_train_posteriors",
    validation_features: str = "pdl_validation_features",
    validation_posterior: str = "pdl_validation_posteriors",
    train_basis: str = "pdl_train_basis",
    validation_basis: str = "pdl_validation_basis",
    num_parts: int = 20,
    representation_seed: int = 1,
    train_as: str = "pdl_transition",
    validation_as: str = "pdl_validation_transition",
    revision_validation_as: str = "pdl_revision_validation_transition",
) -> None:
    from lnl_toolbox.noise.pdl import PartTransitionEstimator
    estimator = PartTransitionEstimator(int(num_parts), int(num_parts), representation_seed=int(representation_seed))
    train_artifact = estimator.estimate_from_shared_representation(
        ctx[train_features], ctx[train_posterior],
        representation_parts=ctx[parts], representation_coefficients=ctx[coefficients],
        representation_indices=ctx[representation_indices], part_matrices=ctx[train_basis], official_raw_basis=True,
    )
    validation_artifact = estimator.estimate_from_shared_representation(
        ctx[validation_features], ctx[validation_posterior],
        representation_parts=ctx[parts], representation_coefficients=ctx[coefficients],
        representation_indices=ctx[representation_indices], part_matrices=ctx[validation_basis], official_raw_basis=True,
    )
    ctx[train_as] = train_artifact
    ctx[validation_as] = validation_artifact
    ctx[revision_validation_as] = validation_artifact.with_part_matrices(
        train_artifact.part_matrices, role="revision_validation",
        source_artifact_hash=train_artifact.artifact_hash,
    )


@block(
    id="attach_pdl_revision_head",
    name="Attach PDL Revision Head",
    category="Model",
    description="Attach the official bias-free global T_revision parameter to the classifier.",
    params={"model": {"type": "slot", "default": "model"}, "num_classes": {"type": "int", "default": 10}},
    requires=("model",),
    provides=(),
    placement=("top",), stage="setup", ui_group="② 初始化",
    formula="T_revision in R^{C x C}, bias-free",
    formula_ref="PDL official train_revision parameterization",
    paper="Part-dependent Label Noise",
)
def attach_pdl_revision_head(ctx: ScratchContext, model: str = "model", num_classes: int = 10) -> None:
    import torch.nn as nn
    if not hasattr(ctx[model], "T_revision"):
        setattr(ctx[model], "T_revision", nn.Linear(int(num_classes), int(num_classes), bias=False))


@block(
    id="reset_pdl_revision",
    name="Reset PDL Revision",
    category="Model",
    description="Reset T_revision to zero before the official revision phase.",
    params={"model": {"type": "slot", "default": "model"}},
    requires=("model",), placement=("top",), stage="setup", ui_group="② 初始化",
)
def reset_pdl_revision(ctx: ScratchContext, model: str = "model") -> None:
    with _torch()[0].no_grad():
        ctx[model].T_revision.weight.zero_()


def _pdl_matrices(ctx, transition: str, indices: str, device, dtype, revision: bool = False, model: str = "model"):
    matrices = ctx[transition].transition_for(None, ctx[indices], device=device, dtype=dtype)
    if revision:
        matrices = matrices + ctx[model].T_revision.weight.to(device=device, dtype=dtype).unsqueeze(0)
        matrices = matrices.clamp_min(0.0)
    return matrices


@block(
    id="pdl_corrected_loss",
    name="PDL Corrected Risk",
    category="Correction",
    description="Compute PDL beta-weighted corrected risk using the estimated instance transition.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "indices": {"type": "slot", "default": "indices"}, "transition": {"type": "slot", "default": "pdl_transition"}, "save_as": {"type": "slot", "default": "loss_per_sample"}},
    requires=("logits", "labels", "indices", "transition"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 纠错风险",
    formula="beta=p(y|x)/p(yt|x); l=beta[-log p(yt|x)]",
    formula_ref="PDL official train_correction objective",
    paper="Part-dependent Label Noise",
)
def pdl_corrected_loss(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", indices: str = "indices", transition: str = "pdl_transition", save_as: str = "loss_per_sample") -> None:
    import torch
    clean = torch.softmax(ctx[logits], dim=1)
    matrices = ctx[transition].transition_for(None, ctx[indices], device=clean.device, dtype=clean.dtype)
    observed = torch.bmm(clean.unsqueeze(1), matrices).squeeze(1)
    clean_y = clean.gather(1, ctx[labels].long()[:, None]).squeeze(1)
    observed_y = observed.gather(1, ctx[labels].long()[:, None]).squeeze(1)
    ctx[save_as] = (clean_y / observed_y.clamp_min(torch.finfo(clean.dtype).tiny)) * (-torch.log(clean_y.clamp_min(torch.finfo(clean.dtype).tiny)))


@block(
    id="pdl_revision_loss",
    name="PDL Revision Risk",
    category="Correction",
    description="Compute PDL's revision-phase corrected risk with trainable T_revision.",
    params={"model": {"type": "slot", "default": "model"}, "logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "indices": {"type": "slot", "default": "indices"}, "transition": {"type": "slot", "default": "pdl_transition"}, "save_as": {"type": "slot", "default": "loss_per_sample"}},
    requires=("model", "logits", "labels", "indices", "transition"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 纠错风险",
    formula="T'=row_normalize(max(0,T(x)+T_revision)); l=beta[-log p(yt|x)]",
    formula_ref="PDL official train_revision objective",
    paper="Part-dependent Label Noise",
)
def pdl_revision_loss(ctx: ScratchContext, model: str = "model", logits: str = "logits", labels: str = "labels", indices: str = "indices", transition: str = "pdl_transition", save_as: str = "loss_per_sample") -> None:
    import torch
    clean = torch.softmax(ctx[logits], dim=1)
    matrices = ctx[transition].transition_for(None, ctx[indices], device=clean.device, dtype=clean.dtype)
    matrices = (matrices + ctx[model].T_revision.weight.to(clean)).clamp_min(0.0)
    matrices = matrices / matrices.sum(dim=2, keepdim=True).clamp_min(torch.finfo(clean.dtype).tiny)
    observed = torch.bmm(clean.unsqueeze(1), matrices).squeeze(1)
    clean_y = clean.gather(1, ctx[labels].long()[:, None]).squeeze(1)
    observed_y = observed.gather(1, ctx[labels].long()[:, None]).squeeze(1)
    ctx[save_as] = (clean_y / observed_y.clamp_min(torch.finfo(clean.dtype).tiny)) * (-torch.log(clean_y.clamp_min(torch.finfo(clean.dtype).tiny)))


@block(
    id="evaluate_pdl_accuracy",
    name="Evaluate PDL Accuracy",
    category="Evaluation",
    description="Evaluate PDL corrected probabilities on noisy validation or test data.",
    params={"model": {"type": "slot", "default": "model"}, "loader": {"type": "slot", "default": "validation_loader"}, "transition": {"type": "slot", "default": "pdl_validation_transition"}, "device": {"type": "slot", "default": "device"}, "revision": {"type": "bool", "default": False}, "final": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "validation_accuracy"}},
    requires=("model", "loader", "transition", "device"), provides=("save_as",), placement=("epoch", "top"), stage="evaluate", ui_group="⑨ 评估",
)
def evaluate_pdl_accuracy(ctx: ScratchContext, model: str = "model", loader: str = "validation_loader", transition: str = "pdl_validation_transition", device: str = "device", revision: bool = False, final: bool = False, save_as: str = "validation_accuracy") -> None:
    import torch
    if final and ctx.get("_runtime_limits", {}).get("skip_final_test") and str(loader) == "test_loader":
        ctx[save_as] = float("nan")
        ctx["pdl_test_skipped"] = True
        return
    network = ctx[model]
    was_training = network.training
    network.eval()
    correct = total = 0
    with torch.inference_mode():
        for batch in ctx[loader]:
            inputs = batch["input"].to(ctx[device])
            labels = batch["target"].to(ctx[device])
            logits = network(inputs)
            matrices = ctx[transition].transition_for(None, batch["index"], device=logits.device, dtype=logits.dtype)
            if revision:
                matrices = matrices + network.T_revision.weight.to(logits)
            matrices = matrices.abs()
            matrices = matrices / matrices.sum(dim=2, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)
            observed = torch.bmm(torch.softmax(logits, dim=1).unsqueeze(1), matrices).squeeze(1)
            correct += int(observed.argmax(1).eq(labels).sum())
            total += int(labels.numel())
    network.train(was_training)
    ctx[save_as] = float(correct / total) if total else 0.0


@block(
    id="jocor_agreement",
    name="JoCoR: Joint Agreement",
    category="Paper Specific",
    description="Combine two peer CE losses with symmetric prediction agreement.",
    params={"logits_a": {"type": "slot", "default": "logits_a"}, "logits_b": {"type": "slot", "default": "logits_b"}, "labels": {"type": "slot", "default": "labels"}, "agreement": {"type": "float", "default": 0.1, "min": 0.0}, "save_as": {"type": "slot", "default": "joint_loss_per_sample"}},
    requires=("logits_a", "logits_b", "labels"),
    provides=("save_as",),
)
def jocor_agreement(ctx: ScratchContext, logits_a: str = "logits_a", logits_b: str = "logits_b", labels: str = "labels", agreement: float = 0.1, save_as: str = "joint_loss_per_sample") -> None:
    torch, F = _torch()
    pa, pb = F.softmax(ctx[logits_a], -1), F.softmax(ctx[logits_b], -1)
    kl = F.kl_div(pa.clamp_min(1e-12).log(), pb, reduction="none").sum(-1) + F.kl_div(pb.clamp_min(1e-12).log(), pa, reduction="none").sum(-1)
    ctx[save_as] = (_ce(ctx[logits_a], ctx[labels]) + _ce(ctx[logits_b], ctx[labels])) / 2 + float(agreement) * kl / 2


@block(
    id="jocor_symmetric_kl",
    name="JoCoR Symmetric KL",
    category="Paper Specific",
    description="Compute the per-sample symmetric prediction agreement term for two peers.",
    params={"logits_a": {"type": "slot", "default": "logits_a"}, "logits_b": {"type": "slot", "default": "logits_b"}, "save_as": {"type": "slot", "default": "agreement_per_sample"}},
    requires=("logits_a", "logits_b"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="D_SKL(p_a,p_b)=KL(p_a||p_b)+KL(p_b||p_a)",
    formula_ref="JoCoR CVPR 2020, co-regularization term",
    paper="Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization",
)
def jocor_symmetric_kl(ctx: ScratchContext, logits_a: str = "logits_a", logits_b: str = "logits_b", save_as: str = "agreement_per_sample") -> None:
    torch, F = _torch()
    log_a = F.log_softmax(ctx[logits_a], dim=-1)
    log_b = F.log_softmax(ctx[logits_b], dim=-1)
    prob_a, prob_b = log_a.exp(), log_b.exp()
    values = F.kl_div(log_a, prob_b, reduction="none").sum(-1) + F.kl_div(log_b, prob_a, reduction="none").sum(-1)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("JoCoR symmetric KL produced non-finite values")
    ctx[save_as] = values


@block(
    id="jocor_joint_composition",
    name="JoCoR Joint Loss",
    category="Paper Specific",
    description="Compose the two supervised peer losses and symmetric KL term before sample selection.",
    params={"loss_a": {"type": "slot", "default": "loss_a_per_sample"}, "loss_b": {"type": "slot", "default": "loss_b_per_sample"}, "agreement": {"type": "slot", "default": "agreement_per_sample"}, "lambda_": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "joint_loss_per_sample"}},
    requires=("loss_a", "loss_b", "agreement"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="J=(1-λ)(L_a+L_b)+λD_SKL",
    formula_ref="JoCoR CVPR 2020, joint objective",
    paper="Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization",
)
def jocor_joint_composition(ctx: ScratchContext, loss_a: str = "loss_a_per_sample", loss_b: str = "loss_b_per_sample", agreement: str = "agreement_per_sample", lambda_: float = 0.9, save_as: str = "joint_loss_per_sample") -> None:
    coefficient = float(lambda_)
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("JoCoR lambda must be in [0, 1]")
    ctx[save_as] = (1.0 - coefficient) * (ctx[loss_a] + ctx[loss_b]) + coefficient * ctx[agreement]


@block(
    id="jocor_keep_rate_formula",
    name="JoCoR Keep-rate Formula",
    category="Paper Specific",
    description="Compute JoCoR's linear small-loss keep rate from the formal epoch schedule.",
    params={"epoch": {"type": "slot", "default": "epoch"}, "start": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0}, "end": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}, "warmup_epochs": {"type": "int", "default": 9, "min": 1}, "save_as": {"type": "slot", "default": "keep_rate"}},
    requires=("epoch",),
    provides=("save_as",),
    placement=("epoch",), stage="train", ui_group="⑩ 论文专用",
    formula="R(t)=start+min(t/T_k,1)(end-start)",
    formula_ref="JoCoR formal selector keep_rate linear schedule",
    paper="Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization",
)
def jocor_keep_rate_formula(ctx: ScratchContext, epoch: str = "epoch", start: float = 1.0, end: float = 0.5, warmup_epochs: int = 9, save_as: str = "keep_rate") -> None:
    progress = min(max(float(ctx[epoch]), 0.0) / max(int(warmup_epochs), 1), 1.0)
    ctx[save_as] = float(start) + progress * (float(end) - float(start))


@block(
    id="dss_evidence",
    name="DSS: Evidence Update",
    category="Paper Specific",
    description="Convert per-sample loss evidence into a stable debiased selection score.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "dss_score"}},
    requires=("losses",),
    provides=("save_as",),
)
def dss_evidence(ctx: ScratchContext, losses: str = "loss_per_sample", save_as: str = "dss_score") -> None:
    values = ctx[losses].detach()
    normalized = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
    ctx[save_as] = torch_sigmoid(-normalized)


def torch_sigmoid(value):
    torch, _ = _torch()
    return torch.sigmoid(value)


@block(
    id="cdr_parameter_mask",
    name="CDR: Critical Parameter Mask",
    category="Paper Specific",
    description="Keep the lower-loss half as a simple critical-update mask for smoke execution.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "keep_rate": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "critical_mask"}},
    requires=("losses",),
    provides=("save_as",),
)
def cdr_parameter_mask(ctx: ScratchContext, losses: str = "loss_per_sample", keep_rate: float = 0.8, save_as: str = "critical_mask") -> None:
    torch, _ = _torch()
    values = ctx[losses].reshape(-1)
    count = max(1, min(values.numel(), int(torch.ceil(torch.tensor(values.numel() * float(keep_rate))).item())))
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[torch.argsort(values, stable=True)[:count]] = True
    ctx[save_as] = mask


@block(
    id="cdr_criticality_score",
    name="CDR Criticality Score",
    category="Parameter Update",
    description="Compute CDR scalar criticality |gradient times parameter| and select the top ceil((1-tau)m) trainable scalars.",
    params={
        "model": {"type": "slot", "default": "model"},
        "noise_rate": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0},
        "save_as": {"type": "slot", "default": "cdr_masks"},
    },
    requires=("model",),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="s_i = |g_i theta_i|; S = top-ceil((1-tau)m)(s)",
    formula_ref="Xia et al., Robust Early-Learning: Hindering the Memorization of Noisy Labels, Eq. (3)-(4)",
    paper="Robust Early-Learning: Hindering the Memorization of Noisy Labels",
)
def cdr_criticality_score(
    ctx: ScratchContext,
    model: str = "model",
    noise_rate: float = 0.4,
    save_as: str = "cdr_masks",
) -> None:
    import math

    torch, _ = _torch()
    tau = float(noise_rate)
    if not 0.0 <= tau < 1.0:
        raise ValueError("CDR noise_rate must satisfy 0 <= noise_rate < 1")
    values = sorted(
        ((str(name), parameter) for name, parameter in ctx[model].named_parameters()
         if parameter.requires_grad and parameter.grad is not None),
        key=lambda item: item[0],
    )
    if not values:
        raise ValueError("CDR requires trainable parameters with gradients")
    scores = torch.cat([
        (parameter.grad.detach() * parameter.detach()).abs().reshape(-1).to(torch.float64)
        for _, parameter in values
    ])
    critical_count = int(math.ceil((1.0 - tau) * int(scores.numel())))
    order = torch.argsort(scores, descending=True, stable=True)
    selected = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    selected[order[:critical_count]] = True
    masks: dict[str, torch.Tensor] = {}
    offset = 0
    for name, parameter in values:
        size = int(parameter.numel())
        masks[name] = selected[offset:offset + size].reshape(parameter.shape)
        offset += size
    ctx[save_as] = masks
    ctx["cdr_eligible_parameters"] = float(scores.numel())
    ctx["cdr_critical_parameters"] = float(critical_count)


@block(
    id="cdr_masked_gradient_update",
    name="CDR Masked Gradient Update",
    category="Parameter Update",
    description="Apply the CDR critical mask, (1-tau) gradient scaling, and paper-mode L1 term before SGD.",
    params={
        "model": {"type": "slot", "default": "model"},
        "masks": {"type": "slot", "default": "cdr_masks"},
        "noise_rate": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0},
        "l1_decay": {"type": "float", "default": 0.001, "min": 0.0},
    },
    requires=("model", "masks"),
    provides=(),
    placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="g_i <- (1-tau) 1[i in S] g_i + lambda sign(theta_i)",
    formula_ref="Xia et al., CDR paper compatibility update rule, Eq. (5)-(6)",
    paper="Robust Early-Learning: Hindering the Memorization of Noisy Labels",
)
def cdr_masked_gradient_update(
    ctx: ScratchContext,
    model: str = "model",
    masks: str = "cdr_masks",
    noise_rate: float = 0.4,
    l1_decay: float = 0.001,
) -> None:
    tau = float(noise_rate)
    if not 0.0 <= tau < 1.0 or float(l1_decay) < 0.0:
        raise ValueError("invalid CDR update parameters")
    with _torch()[0].no_grad():
        for name, parameter in ctx[model].named_parameters():
            if parameter.grad is None or name not in ctx[masks]:
                continue
            gradient = parameter.grad
            gradient.mul_(ctx[masks][name].to(device=gradient.device, dtype=gradient.dtype))
            gradient.mul_(1.0 - tau)
            gradient.add_(parameter.sign(), alpha=float(l1_decay))


@block(
    id="create_mentor_provider",
    name="MentorNet: Create Weight Provider",
    category="Paper Specific",
    description="Load the frozen MentorArtifact and configure the moving-percentile, burn-in, label, and dropout lifecycle from the formal recipe.",
    params={
        "artifact_path": {"type": "str", "default": "data/mentornet/cifar10-symmetric04-official-port-seed20260729/mentor_artifact.pt"},
        "total_epochs": {"type": "int", "default": 100, "min": 1},
        "percentile": {"type": "float", "default": 0.6, "min": 0.0001, "max": 0.9999},
        "decay": {"type": "float", "default": 0.5, "min": 0.0, "max": 0.9999},
        "burn_in_epoch": {"type": "int", "default": 18, "min": 0, "max": 100},
        "fixed_epoch_after_burn_in": {"type": "bool", "default": True},
        "fixed_label": {"type": "int", "default": 0, "min": 0},
        "dropout_schedule": {"type": "value", "default": [[0.5, 17], [0.05, 78], [0.9, 5]]},
        "seed": {"type": "int", "default": 20260729, "min": 0},
        "save_as": {"type": "slot", "default": "mentor_provider"},
    },
    provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="q_t=EMA_p(loss), w_i=M(loss_i,loss_i-q_t,y_i,e_t) with burn-in/dropout lifecycle",
    formula_ref="Jiang et al., MentorNet, ICML 2018; formal pipeline.weight_provider",
    paper="MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Noisy Labels",
)
def create_mentor_provider(
    ctx: ScratchContext,
    artifact_path: str = "data/mentornet/cifar10-symmetric04-official-port-seed20260729/mentor_artifact.pt",
    total_epochs: int = 100,
    percentile: float = 0.6,
    decay: float = 0.5,
    burn_in_epoch: int = 18,
    fixed_epoch_after_burn_in: bool = True,
    fixed_label: int = 0,
    dropout_schedule: Any = ((0.5, 17), (0.05, 78), (0.9, 5)),
    seed: int = 20260729,
    save_as: str = "mentor_provider",
) -> None:
    import os
    from pathlib import Path
    from lnl_toolbox.algorithms.mentornet import MentorNetWeightProvider

    path = Path(str(artifact_path))
    if not path.is_absolute():
        path = Path.cwd() / path
    fixture = bool((ctx.get("_runtime_limits") or {}).get("fixture"))
    if path.exists():
        provider = MentorNetWeightProvider(
            str(path), int(total_epochs), percentile=float(percentile), decay=float(decay),
            burn_in_epoch=int(burn_in_epoch), fixed_epoch_after_burn_in=bool(fixed_epoch_after_burn_in),
            fixed_label=int(fixed_label), dropout_schedule=dropout_schedule, seed=int(seed),
        )
    elif fixture:
        class _FixtureMentorProvider:
            def __init__(self):
                self.moving = None
                self.generator = __import__("torch").Generator().manual_seed(int(seed))

            def compute(self, weight_input):
                torch = __import__("torch")
                losses = weight_input.per_sample_loss.detach()
                current = float(torch.quantile(losses, float(percentile)).item())
                self.moving = current if self.moving is None else float(decay) * self.moving + (1.0 - float(decay)) * current
                epoch = int(weight_input.metadata.get("epoch", 0))
                mentor_epoch = min(epoch, int(burn_in_epoch))
                burn = mentor_epoch < max(0, int(burn_in_epoch) - 1)
                weights = torch.ones_like(losses) if burn else torch.sigmoid(-(losses - self.moving)).detach()
                rate = 0.0
                boundary = 0
                for candidate, duration in dropout_schedule:
                    boundary += int(duration)
                    if mentor_epoch < boundary:
                        rate = float(candidate); break
                if rate:
                    keep = torch.rand(weights.shape, generator=self.generator, device="cpu").to(weights.device) >= rate
                    weights = weights * keep
                if not bool((weights > 0).any()):
                    weights = torch.ones_like(weights)
                from lnl_toolbox.treatments.weights import WeightResult
                return WeightResult(weights.clamp(0, 1).detach(), {"weight_mean": float(weights.mean().item()), "moving_percentile": float(self.moving)})
        provider = _FixtureMentorProvider()
    else:
        raise FileNotFoundError(f"MentorArtifact not found: {path}")
    ctx[save_as] = provider
    ctx["mentor_artifact_path"] = os.fspath(path)


@block(
    id="mentor_compute_weights",
    name="MentorNet: Compute Curriculum Weights",
    category="Paper Specific",
    description="Build the typed SupervisedWeightInput and invoke the frozen MentorNet provider for one batch.",
    params={
        "provider": {"type": "slot", "default": "mentor_provider"},
        "logits": {"type": "slot", "default": "logits"},
        "labels": {"type": "slot", "default": "labels"},
        "indices": {"type": "slot", "default": "indices"},
        "losses": {"type": "slot", "default": "loss_per_sample"},
        "save_as": {"type": "slot", "default": "sample_weights"},
    },
    requires=("provider", "logits", "labels", "losses"), provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="w=MentorNet(l, l-EMA_p(l), y, epoch)",
    formula_ref="MentorNetWeightProvider.compute; Jiang et al. ICML 2018",
    paper="MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Noisy Labels",
)
def mentor_compute_weights(
    ctx: ScratchContext,
    provider: str = "mentor_provider",
    logits: str = "logits",
    labels: str = "labels",
    indices: str = "indices",
    losses: str = "loss_per_sample",
    save_as: str = "sample_weights",
) -> None:
    from lnl_toolbox.treatments.weights import SupervisedWeightInput
    torch, _ = _torch()
    sample_indices = ctx.get(indices)
    if sample_indices is None:
        sample_indices = torch.arange(ctx[losses].shape[0], device=ctx[losses].device)
    result = ctx[provider].compute(SupervisedWeightInput(
        logits=ctx[logits].detach(), noisy_targets=ctx[labels].detach(),
        sample_indices=sample_indices.detach(), per_sample_loss=ctx[losses],
        metadata={"epoch": int(ctx.get("epoch", 0)), "paper": "mentornet"},
    ))
    ctx[save_as] = result.sample_weights.detach()
    ctx["mentor_metrics"] = dict(result.metrics)


@block(
    id="step_milestone_update",
    name="Step-Milestone Parameter Update",
    category="Parameter Update",
    description="Apply the formal optimizer learning-rate drops at global training-step milestones.",
    params={
        "optimizer": {"type": "slot", "default": "optimizer"},
        "milestones": {"type": "value", "default": [19500, 25000, 30000]},
        "gamma": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
        "steps_per_epoch": {"type": "int", "default": 391, "min": 1},
    },
    requires=("optimizer",), placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="lr_t=lr_0 gamma^{|{m: m<=t}|}, t=epoch*S+batch+1",
    formula_ref="formal parameter_update.name=step_milestone",
)
def step_milestone_update(
    ctx: ScratchContext,
    optimizer: str = "optimizer",
    milestones: Any = (19500, 25000, 30000),
    gamma: float = 0.1,
    steps_per_epoch: int = 391,
) -> None:
    step = int(ctx.get("epoch", 0)) * int(steps_per_epoch) + int(ctx.get("batch_idx", 0)) + 1
    count = sum(step >= int(milestone) for milestone in milestones)
    optimizer_value = ctx[optimizer]
    if not hasattr(optimizer_value, "param_groups"):
        raise TypeError("step-milestone update requires a torch optimizer")
    for group in optimizer_value.param_groups:
        base = group.setdefault("_scratch_base_lr", float(group["lr"]))
        group["lr"] = float(base) * float(gamma) ** count


@block(
    id="mentor_weight",
    name="MentorNet: Weight Estimation",
    category="Paper Specific",
    description="Turn loss and epoch into a curriculum weight in [0, 1].",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "sample_weights"}},
    requires=("losses",),
    provides=("save_as",),
)
def mentor_weight(ctx: ScratchContext, losses: str = "loss_per_sample", save_as: str = "sample_weights") -> None:
    values = ctx[losses].detach()
    ctx[save_as] = torch_sigmoid(-(values - values.median()))


@block(
    id="estimate_transition",
    name="Estimate Transition Matrix",
    category="Transition",
    description="Estimate a class-conditional transition matrix from model predictions and observed labels.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "estimated_transition"}},
    requires=("logits", "labels"),
    provides=("save_as",),
)
def estimate_transition(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", save_as: str = "estimated_transition") -> None:
    torch, F = _torch()
    probabilities = F.softmax(ctx[logits].detach(), -1)
    classes = probabilities.shape[-1]
    result = torch.zeros(classes, classes, device=probabilities.device)
    for label in range(classes):
        mask = ctx[labels].long() == label
        result[label] = probabilities[mask].mean(0) if bool(mask.any()) else torch.full((classes,), 1.0 / classes, device=probabilities.device)
    ctx[save_as] = _row_normalize(result)


@block(
    id="snapshot_posterior_model",
    name="Snapshot Posterior Model",
    category="Transition",
    description="Collect P(noisy label | x) from the best posterior-stage model with stable sample indices.",
    params={
        "model": {"type": "slot", "default": "posterior_model"},
        "loader": {"type": "slot", "default": "train_loader"},
        "prepared_data": {"type": "slot", "default": "prepared_data"},
        "device": {"type": "slot", "default": "device"},
        "save_as": {"type": "slot", "default": "posterior_snapshot"},
    },
    requires=("model", "loader", "prepared_data", "device"),
    provides=("save_as",),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="q(x) = softmax(f_theta*(x)); snapshot=(q(x), y_tilde, index)",
    formula_ref="Dual-T posterior snapshot lifecycle before transition estimation",
    paper="Dual T: Reducing Estimation Error for Transition Matrix in Label-noise Learning",
)
def snapshot_posterior_model(
    ctx: ScratchContext,
    model: str = "posterior_model",
    loader: str = "train_loader",
    prepared_data: str = "prepared_data",
    device: str = "device",
    save_as: str = "posterior_snapshot",
) -> None:
    from lnl_toolbox.training.snapshots import collect_posterior_snapshot

    prepared = ctx[prepared_data]
    ctx[save_as] = collect_posterior_snapshot(
        ctx[model],
        ctx[loader],
        ctx[device],
        dataset=str(prepared.dataset),
        split="train",
    )


@block(
    id="dual_t_transition_estimation",
    name="Dual-T Transition Estimation",
    category="Transition",
    description="Estimate Dual-T's composed transition artifact from a posterior snapshot using anchor T-club and argmax-count T-spade.",
    params={
        "posterior": {"type": "slot", "default": "posterior_snapshot"},
        "save_as": {"type": "slot", "default": "transition"},
    },
    requires=("posterior",),
    provides=("save_as",),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="T = T_club T_spade; T_spade[a,b]=count(argmax q=a, y_tilde=b)/count(argmax q=a)",
    formula_ref="Yao et al., Dual T NeurIPS 2020, Algorithm 1",
    paper="Dual T: Reducing Estimation Error for Transition Matrix in Label-noise Learning",
)
def dual_t_transition_estimation(
    ctx: ScratchContext,
    posterior: str = "posterior_snapshot",
    save_as: str = "transition",
) -> None:
    import torch
    from lnl_toolbox.noise.estimators import DualTransitionEstimator

    runtime_limits = ctx.get("_runtime_limits")
    artifact = DualTransitionEstimator().estimate(
        ctx[posterior],
        allow_empty_intermediate=bool(runtime_limits),
    )
    ctx[save_as] = torch.tensor(artifact.matrix, dtype=torch.float32)
    ctx["dual_t_transition_artifact"] = artifact


@block(
    id="estimate_noisy_posterior",
    name="Estimate Noisy Posterior",
    category="Posterior",
    description="Estimate P(noisy label | x) with the configured binary KDE or KLIEP backend and preserve stable sample indices.",
    params={
        "features": {"type": "slot", "default": "posterior_features"},
        "labels": {"type": "slot", "default": "posterior_targets"},
        "indices": {"type": "slot", "default": "posterior_indices"},
        "backend": {"type": "enum", "options": ["kde", "kliep"], "default": "kde"},
        "bandwidth": {"type": "float", "default": 0.15, "min": 0.0},
        "save_as": {"type": "slot", "default": "posterior_snapshot"},
    },
    requires=("features", "labels", "indices"),
    provides=("save_as", "posterior_probabilities", "posterior_indices", "posterior_targets"),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="q_j(x) = P(tilde Y=j | X=x)",
    formula_ref="Li et al., Learning from Noisy Labels with Importance Reweighting, posterior-estimation stage",
    paper="Learning from Noisy Labels with Importance Reweighting",
)
def estimate_noisy_posterior(
    ctx: ScratchContext,
    features: str = "posterior_features",
    labels: str = "posterior_targets",
    indices: str = "posterior_indices",
    backend: str = "kde",
    bandwidth: float = 0.15,
    save_as: str = "posterior_snapshot",
) -> None:
    import torch

    from lnl_toolbox.algorithms.importance_reweighting import (
        build_binary_noisy_posterior_backend,
    )

    feature_values = ctx[features]
    label_values = ctx[labels]
    index_values = ctx[indices]
    config: dict[str, Any] = {"name": str(backend), "bandwidth": float(bandwidth)}
    if str(backend).strip().lower() == "kliep":
        config.update({
            "max_centers": 24,
            "max_iterations": 200,
            "learning_rate": 0.02,
            "tolerance": 1.0e-7,
            "epsilon": 1.0e-12,
            "seed": int(ctx.get("seed", 1)),
        })
    estimator = build_binary_noisy_posterior_backend(config)
    snapshot = estimator.fit_predict(
        feature_values,
        label_values,
        index_values,
        dataset="synthetic_binary_2d",
        split="train",
    )
    ctx[save_as] = snapshot
    ctx["posterior_probabilities"] = torch.tensor(
        snapshot.noisy_probabilities, dtype=torch.float32
    )
    ctx["posterior_indices"] = torch.tensor(
        snapshot.global_indices, dtype=torch.long
    )
    ctx["posterior_targets"] = torch.tensor(
        snapshot.noisy_targets, dtype=torch.long
    )


@block(
    id="estimate_raw_min_noise_rates",
    name="Estimate Raw-min Noise Rates",
    category="Weighting",
    description="Estimate asymmetric binary RCN rates from the minimum noisy posterior for each observed class.",
    params={
        "posterior": {"type": "slot", "default": "posterior_snapshot"},
        "positive_as": {"type": "slot", "default": "rho_positive"},
        "negative_as": {"type": "slot", "default": "rho_negative"},
    },
    requires=("posterior",),
    provides=("positive_as", "negative_as"),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="rho_hat_0 = min_x q_0(x), rho_hat_1 = min_x q_1(x)",
    formula_ref="Li et al., raw-min noise-rate estimator",
    paper="Learning from Noisy Labels with Importance Reweighting",
)
def estimate_raw_min_noise_rates(
    ctx: ScratchContext,
    posterior: str = "posterior_snapshot",
    positive_as: str = "rho_positive",
    negative_as: str = "rho_negative",
) -> None:
    from lnl_toolbox.algorithms.importance_reweighting import PaperRawMinNoiseRateEstimator

    artifact = PaperRawMinNoiseRateEstimator().estimate(ctx[posterior])
    ctx[positive_as] = float(artifact.rho_positive)
    ctx[negative_as] = float(artifact.rho_negative)
    ctx["noise_rate_artifact"] = artifact


@block(
    id="compose_transition",
    name="Compose Transition Matrices",
    category="Transition",
    description="Compose two row-stochastic transition matrices for Dual-T style correction.",
    params={"first": {"type": "slot", "default": "transition_a"}, "second": {"type": "slot", "default": "transition_b"}, "save_as": {"type": "slot", "default": "composed_transition"}},
    requires=("first", "second"),
    provides=("save_as",),
)
def compose_transition(ctx: ScratchContext, first: str = "transition_a", second: str = "transition_b", save_as: str = "composed_transition") -> None:
    ctx[save_as] = _row_normalize(ctx[first] @ ctx[second])


@block(
    id="importance_reweight",
    name="Importance Reweight",
    category="Weighting",
    description="Weight per-sample CE by clean posterior over observed noisy posterior.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "weighted_loss"}},
    requires=("logits", "labels", "transition"),
    provides=("save_as",), beginner_visible=False,
)
def importance_reweight(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", transition: str = "transition", save_as: str = "weighted_loss") -> None:
    torch, F = _torch()
    clean = F.softmax(ctx[logits], -1)
    noisy = clean @ ctx[transition].to(clean)
    target = ctx[labels].long()[:, None]
    weights = clean.gather(1, target).squeeze(1) / noisy.gather(1, target).squeeze(1).clamp_min(torch.finfo(clean.dtype).tiny)
    ctx[save_as] = _ce(ctx[logits], ctx[labels]) * weights.detach()


@block(
    id="importance_weight_formula",
    name="Importance Weight Formula",
    category="Weighting",
    description="Compute detached binary importance weights from the estimated noisy posterior and asymmetric RCN rates.",
    params={
        "posterior": {"type": "slot", "default": "posterior_probabilities"},
        "posterior_indices": {"type": "slot", "default": "posterior_indices"},
        "labels": {"type": "slot", "default": "labels"},
        "indices": {"type": "slot", "default": "indices"},
        "rho_positive": {"type": "slot", "default": "rho_positive"},
        "rho_negative": {"type": "slot", "default": "rho_negative"},
        "save_as": {"type": "slot", "default": "sample_weights"},
    },
    requires=("posterior", "posterior_indices", "labels", "indices", "rho_positive", "rho_negative"),
    provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="w(x, y_tilde) = (q_y(x) - rho_{1-y}) / ((1-rho_0-rho_1) q_y(x))",
    formula_ref="Li et al., binary asymmetric RCN importance-weight formula",
    paper="Learning from Noisy Labels with Importance Reweighting",
)
def importance_weight_formula(
    ctx: ScratchContext,
    posterior: str = "posterior_probabilities",
    posterior_indices: str = "posterior_indices",
    labels: str = "labels",
    indices: str = "indices",
    rho_positive: str = "rho_positive",
    rho_negative: str = "rho_negative",
    save_as: str = "sample_weights",
) -> None:
    torch, _ = _torch()
    values = ctx[posterior]
    table_indices = ctx[posterior_indices]
    targets = ctx[labels].long()
    batch_indices = ctx[indices].long().to(targets.device)
    if not isinstance(table_indices, torch.Tensor):
        table_indices = torch.as_tensor(table_indices, dtype=torch.long, device=batch_indices.device)
    else:
        table_indices = table_indices.to(batch_indices.device)
    positions = torch.searchsorted(table_indices, batch_indices)
    if bool((positions >= table_indices.numel()).any().item()) or not torch.equal(table_indices[positions], batch_indices):
        raise ValueError("importance posterior is missing a batch sample index")
    probabilities = values.to(batch_indices.device)[positions]
    if probabilities.shape != (targets.shape[0], 2):
        raise ValueError("binary importance posterior must align with the batch")
    q = probabilities.gather(1, targets[:, None]).squeeze(1)
    opposite_rate = torch.where(
        targets == 1,
        torch.as_tensor(float(ctx[rho_negative]), dtype=probabilities.dtype, device=probabilities.device),
        torch.as_tensor(float(ctx[rho_positive]), dtype=probabilities.dtype, device=probabilities.device),
    )
    gap = 1.0 - float(ctx[rho_positive]) - float(ctx[rho_negative])
    if gap <= 0.0:
        raise ValueError("binary importance rates must have positive identifiability gap")
    weights = torch.zeros_like(q)
    nonzero = q != 0
    weights[nonzero] = (q[nonzero] - opposite_rate[nonzero]) / (gap * q[nonzero])
    ctx[save_as] = weights.clamp_min(0.0).detach()


@block(
    id="cwd_statistics",
    name="CWD: Class-wise Statistics",
    category="Paper Specific",
    description="Compute class centroids and a global denoising objective from features.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "cwd_loss"}},
    requires=("features", "labels"),
    provides=("save_as", "class_centroids"),
)
def cwd_statistics(ctx: ScratchContext, features: str = "features", labels: str = "labels", save_as: str = "cwd_loss") -> None:
    torch, _ = _torch()
    values, targets = ctx[features], ctx[labels].long()
    classes = int(ctx.get("num_classes", int(targets.max().item()) + 1))
    centroids = torch.stack([values[targets == c].mean(0) if bool((targets == c).any()) else torch.zeros(values.shape[1], device=values.device) for c in range(classes)])
    ctx["class_centroids"] = centroids
    ctx[save_as] = ((values - centroids[targets]) ** 2).mean(dim=1)


def _cwd_swap_matrix(classes: int, source: int, target: int, *, device, dtype):
    torch, _ = _torch()
    value = torch.eye(classes, device=device, dtype=dtype)
    value[[source, target]] = value[[target, source]]
    return value


@block(
    id="snapshot_cwd_features",
    name="CWD: Snapshot Train Features",
    category="Paper Specific",
    description="Collect the full train-evaluation feature snapshot used by CWD before each epoch.",
    params={"model": {"type": "slot", "default": "model"}, "loader": {"type": "slot", "default": "train_eval_loader"}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "cwd_snapshot"}},
    requires=("model", "loader", "device"),
    provides=("save_as",),
    placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="h_i = f_theta(x_i)", formula_ref="CWD feature snapshot before each epoch", paper="Class-Wise Denoising",
)
def snapshot_cwd_features(ctx: ScratchContext, model: str = "model", loader: str = "train_eval_loader", device: str = "device", save_as: str = "cwd_snapshot") -> None:
    from lnl_toolbox.training.snapshots import collect_feature_snapshot
    ctx[save_as] = collect_feature_snapshot(
        ctx[model], ctx[loader], ctx[device], dataset="cifar10_airplane_automobile",
        split=f"train_fold_{int(ctx.get('fold_index', 0))}",
        feature_extractor=lambda network, inputs: network.forward_with_features(inputs).features,
    )


@block(
    id="cwd_observed_statistics",
    name="CWD: Observed Class Statistics",
    category="Paper Specific",
    description="Compute observed class prior and class feature means from the noisy snapshot.",
    params={"snapshot": {"type": "slot", "default": "cwd_snapshot"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "cwd_observed"}},
    requires=("snapshot", "transition"), provides=("save_as", "observed_prior", "observed_centroids"),
    placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="p~_c = n_c/N; m~_c = (1/N) sum_i h_i 1[y~_i=c]", formula_ref="CWD observed statistics", paper="Class-Wise Denoising",
)
def cwd_observed_statistics(ctx: ScratchContext, snapshot: str = "cwd_snapshot", transition: str = "transition", save_as: str = "cwd_observed") -> None:
    torch, _ = _torch()
    value = ctx[snapshot]
    features = torch.as_tensor(value.features, dtype=torch.float32)
    labels = torch.as_tensor(value.noisy_targets, dtype=torch.long)
    matrix = torch.as_tensor(ctx[transition], dtype=features.dtype)
    classes = int(matrix.shape[0])
    counts = torch.bincount(labels, minlength=classes).to(features.dtype)
    prior = counts / counts.sum().clamp_min(torch.finfo(features.dtype).tiny)
    means = torch.stack([features[labels == c].sum(0) / float(features.shape[0]) for c in range(classes)], dim=1)
    result = {"features": features, "labels": labels, "transition": matrix, "classes": classes}
    ctx[save_as] = result
    ctx["observed_prior"] = prior
    ctx["observed_centroids"] = means


@block(
    id="cwd_clean_prior",
    name="CWD: Recover Clean Prior",
    category="Paper Specific",
    description="Solve the clean-prior relation from the observed prior and label transition matrix.",
    params={"observed_prior": {"type": "slot", "default": "observed_prior"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "clean_prior"}},
    requires=("observed_prior", "transition"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="p~ = T^T p; p = solve(T^T,p~)", formula_ref="CWD Eq. (19)", paper="Class-Wise Denoising",
)
def cwd_clean_prior(ctx: ScratchContext, observed_prior: str = "observed_prior", transition: str = "transition", save_as: str = "clean_prior") -> None:
    torch, _ = _torch()
    prior = torch.linalg.solve(torch.as_tensor(ctx[transition], dtype=torch.float64).transpose(0, 1), torch.as_tensor(ctx[observed_prior], dtype=torch.float64))
    if bool((prior < -1e-8).any()) or not bool(torch.isfinite(prior).all()):
        raise ValueError("CWD clean prior is invalid")
    ctx[save_as] = prior.clamp_min(0.0) / prior.sum().clamp_min(torch.finfo(prior.dtype).tiny)


@block(
    id="cwd_virtual_systems",
    name="CWD: Build Virtual Systems",
    category="Paper Specific",
    description="Construct each virtual prior, virtual flip matrix, and coefficient matrix.",
    params={"clean_prior": {"type": "slot", "default": "clean_prior"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "cwd_systems"}},
    requires=("clean_prior", "transition"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="C_k = sum_{s,t} p^k_s T^k_{s,t} P_{s,t}^T", formula_ref="CWD Eqs. (21)-(29)", paper="Class-Wise Denoising",
)
def cwd_virtual_systems(ctx: ScratchContext, clean_prior: str = "clean_prior", transition: str = "transition", save_as: str = "cwd_systems") -> None:
    torch, _ = _torch()
    prior = torch.as_tensor(ctx[clean_prior], dtype=torch.float64)
    transition_value = torch.as_tensor(ctx[transition], dtype=torch.float64)
    classes = int(transition_value.shape[0])
    systems = []
    for clean_class in range(classes):
        virtual_prior = prior @ transition_value
        virtual_prior = virtual_prior - prior[clean_class] * transition_value[clean_class]
        virtual_prior[clean_class] += prior[clean_class]
        denominator = virtual_prior[clean_class]
        virtual_flip = torch.eye(classes, dtype=torch.float64)
        for target in range(classes):
            if target != clean_class:
                virtual_flip[clean_class, target] = prior[clean_class] * transition_value[clean_class, target] / denominator
                virtual_flip[target, clean_class] = 0.0
        virtual_flip[clean_class, clean_class] = 1.0 - virtual_flip[clean_class].sum() + virtual_flip[clean_class, clean_class]
        coefficient = torch.zeros((classes, classes), dtype=torch.float64)
        for source in range(classes):
            for target in range(classes):
                coefficient += virtual_prior[source] * virtual_flip[source, target] * _cwd_swap_matrix(classes, source, target, device=coefficient.device, dtype=coefficient.dtype).transpose(0, 1)
        systems.append({"virtual_prior": virtual_prior, "virtual_flip": virtual_flip, "coefficient": coefficient})
    ctx[save_as] = systems


@block(
    id="cwd_coefficient_pseudoinverse",
    name="CWD: Coefficient Pseudoinverse",
    category="Paper Specific",
    description="Compute the unregularized Moore-Penrose pseudoinverse for each CWD coefficient matrix.",
    params={"systems": {"type": "slot", "default": "cwd_systems"}, "save_as": {"type": "slot", "default": "cwd_pseudoinverses"}},
    requires=("systems",), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="C_k^+ = pinv(C_k)", formula_ref="CWD Eq. (30)", paper="Class-Wise Denoising",
)
def cwd_coefficient_pseudoinverse(ctx: ScratchContext, systems: str = "cwd_systems", save_as: str = "cwd_pseudoinverses") -> None:
    torch, _ = _torch()
    ctx[save_as] = [dict(item, pseudoinverse=torch.linalg.pinv(item["coefficient"])) for item in ctx[systems]]


@block(
    id="cwd_recover_centroids",
    name="CWD: Recover Clean Centroids",
    category="Paper Specific",
    description="Recover clean class centroids from observed means and virtual coefficient pseudoinverses.",
    params={"observed_centroids": {"type": "slot", "default": "observed_centroids"}, "pseudoinverses": {"type": "slot", "default": "cwd_pseudoinverses"}, "save_as": {"type": "slot", "default": "cwd_centroids"}},
    requires=("observed_centroids", "pseudoinverses"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="M = sum_k M~ C_k^+ - (C-1)M~", formula_ref="CWD centroid recovery", paper="Class-Wise Denoising",
)
def cwd_recover_centroids(ctx: ScratchContext, observed_centroids: str = "observed_centroids", pseudoinverses: str = "cwd_pseudoinverses", save_as: str = "cwd_centroids") -> None:
    torch, _ = _torch()
    observed = ctx[observed_centroids].to(dtype=ctx[pseudoinverses][0]["pseudoinverse"].dtype)
    virtual = [observed @ item["pseudoinverse"] for item in ctx[pseudoinverses]]
    ctx[save_as] = (torch.stack(virtual, dim=0).sum(dim=0) - (len(virtual) - 1) * observed).transpose(0, 1)


@block(
    id="cwd_global_objective",
    name="CWD: Binary Scalar Global Objective",
    category="Paper Specific",
    description="Evaluate CWD's squared global risk for the paper's binary scalar classifier with dynamic centroids.",
    params={"model": {"type": "slot", "default": "model"}, "features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "centroids": {"type": "slot", "default": "cwd_centroids"}, "clean_prior": {"type": "slot", "default": "clean_prior"}, "pseudoinverses": {"type": "slot", "default": "cwd_pseudoinverses"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("model", "features", "labels", "centroids", "clean_prior", "pseudoinverses"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑦ 论文目标",
    formula="L = 1 + mean(m^2) - 2 w^T(mu_1-mu_0) - 2b(p_1-p_0)", formula_ref="CWD global squared objective", paper="Class-Wise Denoising",
)
def cwd_global_objective(ctx: ScratchContext, model: str = "model", features: str = "features", labels: str = "labels", centroids: str = "cwd_centroids", clean_prior: str = "clean_prior", pseudoinverses: str = "cwd_pseudoinverses", save_as: str = "loss") -> None:
    torch, _ = _torch()
    network = ctx[model]
    weight = network.classifier.weight[0]
    bias = network.classifier.bias
    value = ctx[features]
    means = torch.as_tensor(ctx[centroids], dtype=value.dtype, device=value.device)
    observed = value.transpose(0, 1) @ torch.nn.functional.one_hot(ctx[labels].long(), num_classes=2).to(value.dtype) / float(value.shape[0])
    dynamic = (torch.stack([observed @ item["pseudoinverse"].to(value.device, value.dtype) for item in ctx[pseudoinverses]], dim=0).sum(dim=0) - observed).transpose(0, 1)
    margin = value @ weight
    if bias is not None:
        margin = margin + bias[0]
    prior = torch.as_tensor(ctx[clean_prior], dtype=value.dtype, device=value.device)
    cross = (weight * (dynamic[1] - dynamic[0])).sum()
    if bias is not None:
        cross = cross + bias[0] * (prior[1] - prior[0])
    ctx[save_as] = 1.0 + margin.square().mean() - 2.0 * cross


@block(
    id="pcse_statistics",
    name="PCSE: Recover Per-class Statistics",
    category="Paper Specific",
    description="Recover means and variances from feature snapshots grouped by observed class.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "pcse_statistics"}},
    requires=("features", "labels"),
    provides=("save_as",),
)
def pcse_statistics(ctx: ScratchContext, features: str = "features", labels: str = "labels", save_as: str = "pcse_statistics") -> None:
    torch, _ = _torch()
    values, targets = ctx[features], ctx[labels].long()
    classes = int(ctx.get("num_classes", int(targets.max().item()) + 1))
    rows = []
    for c in range(classes):
        group = values[targets == c]
        rows.append(torch.cat((group.mean(0), group.var(0, unbiased=False)) if group.numel() else (torch.zeros(values.shape[1], device=values.device), torch.zeros(values.shape[1], device=values.device))))
    ctx[save_as] = torch.stack(rows)


@block(
    id="pcse_recover_clean_priors",
    name="PCSE: Recover Clean Priors",
    category="Paper Specific",
    description="Recover strictly positive clean class priors from noisy priors and a transition matrix.",
    params={"noisy_priors": {"type": "slot", "default": "noisy_priors"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "clean_priors"}},
    requires=("noisy_priors", "transition"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="T^T p = p~; p = solve(T^T,p~)", formula_ref="PCSE Eq. (17)", paper="Estimating Per-Class Statistics",
)
def pcse_recover_clean_priors(ctx: ScratchContext, noisy_priors: str = "noisy_priors", transition: str = "transition", save_as: str = "clean_priors") -> None:
    import numpy as np
    from lnl_toolbox.algorithms.pcse.statistics import recover_clean_priors
    ctx[save_as] = recover_clean_priors(np.asarray(ctx[noisy_priors]), np.asarray(ctx[transition])).astype(np.float32)


@block(
    id="pcse_coefficient_matrix",
    name="PCSE: Coefficient Matrix",
    category="Paper Specific",
    description="Build PCSE's row-swap coefficient matrix from clean priors and transition probabilities.",
    params={"clean_priors": {"type": "slot", "default": "clean_priors"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "pcse_coefficient"}},
    requires=("clean_priors", "transition"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="M = sum_ij p_i T_ij K_ij^T", formula_ref="PCSE Eq. (19)", paper="Estimating Per-Class Statistics",
)
def pcse_coefficient_matrix(ctx: ScratchContext, clean_priors: str = "clean_priors", transition: str = "transition", save_as: str = "pcse_coefficient") -> None:
    import numpy as np
    from lnl_toolbox.algorithms.pcse.statistics import build_coefficient_matrix
    ctx[save_as] = build_coefficient_matrix(np.asarray(ctx[clean_priors]), np.asarray(ctx[transition])).astype(np.float32)


@block(
    id="pcse_recover_layer_statistics",
    name="PCSE: Recover Layer Statistics",
    category="Paper Specific",
    description="Recover clean per-class means, second moments, and covariances for aligned feature snapshots.",
    params={"snapshots": {"type": "slot", "default": "pcse_snapshots"}, "layer_names": {"type": "value", "default": ["layer3", "layer4"]}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "pcse_statistics"}},
    requires=("snapshots", "transition"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="mu=R^T mu~; S=R^T S~; Sigma=S-mu mu^T", formula_ref="PCSE Eqs. (20)-(23)", paper="Estimating Per-Class Statistics",
)
def pcse_recover_layer_statistics(ctx: ScratchContext, snapshots: str = "pcse_snapshots", layer_names: list[str] | tuple[str, ...] = ("layer3", "layer4"), transition: str = "transition", save_as: str = "pcse_statistics") -> None:
    import numpy as np
    from lnl_toolbox.algorithms.pcse.statistics import estimate_pcse_statistics
    result = estimate_pcse_statistics(ctx[snapshots], tuple(layer_names), np.asarray(ctx[transition]))
    ctx[save_as] = result


@block(
    id="snapshot_pcse_features",
    name="PCSE: Snapshot Feature Layers",
    category="Paper Specific",
    description="Collect aligned layer3/layer4 feature snapshots from the formal train-evaluation loader.",
    params={"model": {"type": "slot", "default": "model"}, "prepared_data": {"type": "slot", "default": "prepared_data"}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "pcse_snapshots"}},
    requires=("model", "prepared_data", "device"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="h_l(x)=pool(layer_l(f_theta(x)))", formula_ref="PCSE feature-stage snapshots", paper="Estimating Per-Class Statistics",
)
def snapshot_pcse_features(ctx: ScratchContext, model: str = "model", prepared_data: str = "prepared_data", device: str = "device", save_as: str = "pcse_snapshots") -> None:
    from lnl_toolbox.algorithms.pcse.config import PCSEFeatureLayerConfig
    from lnl_toolbox.algorithms.pcse.features import collect_pcse_features
    prepared = ctx[prepared_data]
    loader = prepared.loader("train_eval", shuffle=False)
    result = collect_pcse_features(
        ctx[model], loader, ctx[device], dataset="cifar10", split="train",
        layers=(PCSEFeatureLayerConfig("layer3", "global_average"), PCSEFeatureLayerConfig("layer4", "global_average")),
    )
    ctx[save_as] = result.snapshots


@block(
    id="snapshot_pcse_validation_features",
    name="PCSE: Snapshot Noisy Validation Features",
    category="Paper Specific",
    description="Collect aligned layer snapshots on the noisy validation split for GDA ensemble selection.",
    params={"model": {"type": "slot", "default": "model"}, "prepared_data": {"type": "slot", "default": "prepared_data"}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "pcse_validation_snapshots"}},
    requires=("model", "prepared_data", "device"), provides=("save_as",), placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
    formula="h_l(x_val)=pool(layer_l(f_theta(x_val)))", formula_ref="PCSE noisy-validation feature lifecycle", paper="Estimating Per-Class Statistics",
)
def snapshot_pcse_validation_features(ctx: ScratchContext, model: str = "model", prepared_data: str = "prepared_data", device: str = "device", save_as: str = "pcse_validation_snapshots") -> None:
    from lnl_toolbox.algorithms.pcse.config import PCSEFeatureLayerConfig
    from lnl_toolbox.algorithms.pcse.features import collect_pcse_features
    prepared = ctx[prepared_data]
    loader = prepared.loader("noisy_validation", shuffle=False)
    result = collect_pcse_features(ctx[model], loader, ctx[device], dataset="cifar10", split="validation", layers=(PCSEFeatureLayerConfig("layer3", "global_average"), PCSEFeatureLayerConfig("layer4", "global_average")))
    ctx[save_as] = result.snapshots


@block(
    id="snapshot_pcse_test_features",
    name="PCSE: Snapshot Clean Test Features",
    category="Paper Specific",
    description="Collect aligned layer snapshots on the clean test split for final ensemble evaluation.",
    params={"model": {"type": "slot", "default": "model"}, "prepared_data": {"type": "slot", "default": "prepared_data"}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "pcse_test_snapshots"}},
    requires=("model", "prepared_data", "device"), provides=("save_as",), placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
    formula="h_l(x_test)=pool(layer_l(f_theta(x_test)))", formula_ref="PCSE clean-test feature lifecycle", paper="Estimating Per-Class Statistics",
)
def snapshot_pcse_test_features(ctx: ScratchContext, model: str = "model", prepared_data: str = "prepared_data", device: str = "device", save_as: str = "pcse_test_snapshots") -> None:
    from lnl_toolbox.algorithms.pcse.config import PCSEFeatureLayerConfig
    from lnl_toolbox.algorithms.pcse.features import collect_pcse_features
    prepared = ctx[prepared_data]
    result = collect_pcse_features(ctx[model], prepared.loader("test", shuffle=False), ctx[device], dataset="cifar10", split="test", layers=(PCSEFeatureLayerConfig("layer3", "global_average"), PCSEFeatureLayerConfig("layer4", "global_average")))
    ctx[save_as] = result.snapshots


@block(
    id="pcse_fit_gda",
    name="PCSE: Fit Shared-Covariance GDA",
    category="Paper Specific",
    description="Fit one shared-covariance GDA classifier per recovered feature layer.",
    params={"statistics": {"type": "slot", "default": "pcse_statistics"}, "covariance_ridge": {"type": "float", "default": 0.1, "min": 0.0}, "save_as": {"type": "slot", "default": "pcse_gda"}},
    requires=("statistics",), provides=("save_as",), placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
    formula="Sigma_shared=sum_c p_c Sigma_c + lambda I; p(c|h)=softmax(delta_c(h))", formula_ref="PCSE GDA stage", paper="Estimating Per-Class Statistics",
)
def pcse_fit_gda(ctx: ScratchContext, statistics: str = "pcse_statistics", covariance_ridge: float = 0.1, save_as: str = "pcse_gda") -> None:
    from lnl_toolbox.algorithms.pcse.gda import fit_gda_layers
    ctx[save_as] = fit_gda_layers(ctx[statistics], covariance_ridge=float(covariance_ridge))


@block(
    id="pcse_fit_ensemble_weights",
    name="PCSE: Fit Validation Ensemble Weights",
    category="Paper Specific",
    description="Optimize positive simplex weights for the two GDA layers on noisy-validation NLL.",
    params={"gda": {"type": "slot", "default": "pcse_gda"}, "snapshots": {"type": "slot", "default": "pcse_validation_snapshots"}, "epochs": {"type": "int", "default": 5, "min": 1}, "learning_rate": {"type": "float", "default": 0.05, "min": 0.0}, "save_as": {"type": "slot", "default": "pcse_ensemble_weights"}},
    requires=("gda", "snapshots"), provides=("save_as",), placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
    formula="w=softmax(a); min_a -mean log(sum_l w_l p_l(y~|x))", formula_ref="PCSE ensemble validation objective", paper="Estimating Per-Class Statistics",
)
def pcse_fit_ensemble_weights(ctx: ScratchContext, gda: str = "pcse_gda", snapshots: str = "pcse_validation_snapshots", epochs: int = 5, learning_rate: float = 0.05, save_as: str = "pcse_ensemble_weights") -> None:
    import numpy as np
    from lnl_toolbox.algorithms.pcse.gda import fit_ensemble_weights
    values = np.stack([layer.posterior(snapshot.features) for layer, snapshot in zip(ctx[gda], ctx[snapshots])], axis=0)
    targets = np.asarray(ctx[snapshots][0].noisy_targets)
    raw, _optimizer, losses = fit_ensemble_weights(values, targets, epochs=int(epochs), learning_rate=float(learning_rate))
    ctx[save_as] = {"raw_weights": raw, "weights": raw.softmax(dim=0), "losses": losses}


@block(
    id="pcse_evaluate_ensemble",
    name="PCSE: Evaluate Ensemble",
    category="Paper Specific",
    description="Evaluate the validation-selected weighted GDA posterior on a clean target snapshot.",
    params={"gda": {"type": "slot", "default": "pcse_gda"}, "snapshots": {"type": "slot", "default": "pcse_test_snapshots"}, "weights": {"type": "slot", "default": "pcse_ensemble_weights"}, "save_as": {"type": "slot", "default": "pcse_test_metrics"}},
    requires=("gda", "snapshots", "weights"), provides=("save_as",), placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
    formula="p(y|x)=sum_l w_l p_l(y|h_l(x)); accuracy=mean[argmax p=y]", formula_ref="PCSE final clean-test ensemble", paper="Estimating Per-Class Statistics",
)
def pcse_evaluate_ensemble(ctx: ScratchContext, gda: str = "pcse_gda", snapshots: str = "pcse_test_snapshots", weights: str = "pcse_ensemble_weights", save_as: str = "pcse_test_metrics") -> None:
    import numpy as np
    probabilities = np.stack([layer.posterior(snapshot.features) for layer, snapshot in zip(ctx[gda], ctx[snapshots])], axis=0)
    values = np.asarray(ctx[weights]["weights"].detach().cpu().numpy())
    posterior = np.einsum("l,lnc->nc", values, probabilities)
    targets = np.asarray(ctx[snapshots][0].noisy_targets)
    ctx[save_as] = {"accuracy": float(np.mean(posterior.argmax(axis=1) == targets)), "samples": int(targets.size), "weights": values.tolist()}


@block(
    id="fine_prepare_strong_input",
    name="FINE: Prepare Strong View",
    category="Data",
    description="Expose the loader's strong augmentation when present, or use the deterministic fixture image for structural execution.",
    params={"batch": {"type": "slot", "default": "batch"}, "input": {"type": "slot", "default": "images"}, "save_as": {"type": "slot", "default": "strong_images"}},
    requires=("batch", "input"), provides=("save_as",), placement=("batch",), stage="data", ui_group="① 数据准备",
)
def fine_prepare_strong_input(ctx: ScratchContext, batch: str = "batch", input: str = "images", save_as: str = "strong_images") -> None:
    value = ctx[batch].get("strong_input", ctx[batch].get("strong_images")) if isinstance(ctx[batch], dict) else None
    ctx[save_as] = value if value is not None else ctx[input]


@block(
    id="create_fine_state",
    name="FINE: Create EMA/SED State",
    category="Paper Specific",
    description="Create the official EMA teacher, self-adaptive class selector, confidence reweighting, and FINE regularizer state.",
    params={
        "model": {"type": "slot", "default": "model"},
        "num_classes": {"type": "int", "default": 100, "min": 2},
        "ema_momentum": {"type": "float", "default": 0.95, "min": 0.0, "max": 0.999999},
        "momentum_scs": {"type": "float", "default": 0.999, "min": 0.0, "max": 0.999999},
        "momentum_scr": {"type": "float", "default": 0.99, "min": 0.0, "max": 0.999999},
        "maximum_threshold": {"type": "float", "default": 0.95, "min": 0.0, "max": 1.0},
        "beta": {"type": "float", "default": 0.1, "min": 0.0},
        "gamma": {"type": "float", "default": 0.002, "min": 0.0},
        "probability_floor": {"type": "float", "default": 1.0e-7, "min": 1.0e-12},
        "seed": {"type": "int", "default": 23, "min": 0},
        "save_as": {"type": "slot", "default": "fine_state"},
    },
    requires=("model",), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="EMA_t=m EMA_{t-1}+(1-m)f_t; SCS/SCR operate on epoch snapshots",
    formula_ref="FINE official SED warm-up/EMA/SCS/SCR lifecycle",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def create_fine_state(
    ctx: ScratchContext,
    model: str = "model",
    num_classes: int = 100,
    ema_momentum: float = 0.95,
    momentum_scs: float = 0.999,
    momentum_scr: float = 0.99,
    maximum_threshold: float = 0.95,
    beta: float = 0.1,
    gamma: float = 0.002,
    probability_floor: float = 1.0e-7,
    seed: int = 23,
    save_as: str = "fine_state",
) -> None:
    from lnl_toolbox.selectors.sed import SelfAdaptiveClassSelector, SelfAdaptiveConfidenceReweighting
    from lnl_toolbox.algorithms.fine import FINERegularizer
    from lnl_toolbox.training.model_ema import ModelEMA
    ctx[save_as] = {
        "ema": ModelEMA(ctx[model], float(ema_momentum), update_buffers=False),
        "scs": SelfAdaptiveClassSelector(int(num_classes), float(momentum_scs), quantile=0.8, maximum_threshold=float(maximum_threshold)),
        "scr": SelfAdaptiveConfidenceReweighting(int(num_classes), float(momentum_scr)),
        "regularizer": FINERegularizer(beta=float(beta), gamma=float(gamma), probability_floor=float(probability_floor), seed=int(seed)),
        "clean_by_index": {}, "pseudo_by_index": {}, "weight_by_index": {},
    }


@block(
    id="fine_snapshot_predictions",
    name="FINE: Snapshot EMA Predictions",
    category="Paper Specific",
    description="Collect stable-index classifier and EMA probabilities on the non-augmented train-evaluation stream.",
    params={"model": {"type": "slot", "default": "model"}, "state": {"type": "slot", "default": "fine_state"}, "loader": {"type": "slot", "default": "train_eval_loader"}, "save_as": {"type": "slot", "default": "fine_snapshot"}},
    requires=("model", "state", "loader"), provides=("save_as",), placement=("epoch",), stage="train", ui_group="⑥ 后验与权重",
    formula="P_t=f_t(x), P_t^{EMA}=EMA_t(x), aligned by stable sample index",
    formula_ref="FINE SED epoch snapshot",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def fine_snapshot_predictions(ctx: ScratchContext, model: str = "model", state: str = "fine_state", loader: str = "train_eval_loader", save_as: str = "fine_snapshot") -> None:
    torch, F = _torch()
    network, ema = ctx[model], ctx[state]["ema"].model
    device = next(network.parameters()).device
    was_training, ema_training = network.training, ema.training
    network.eval(); ema.eval()
    probabilities, ema_probabilities, targets, indices = [], [], [], []
    offset = 0
    max_batches = (ctx.get("_runtime_limits") or {}).get("max_batches")
    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx[loader]):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            if isinstance(batch, dict):
                inputs = batch.get("input", batch.get("images", batch.get("inputs")))
                labels = batch.get("target", batch.get("labels", batch.get("targets")))
                batch_indices = batch.get("index", batch.get("indices"))
            else:
                inputs, labels = batch[0], batch[1]
                batch_indices = batch[2] if len(batch) > 2 else None
            inputs, labels = inputs.to(device), labels.to(device).long()
            probabilities.append(F.softmax(network(inputs), dim=1).cpu())
            ema_probabilities.append(F.softmax(ema(inputs), dim=1).cpu())
            targets.append(labels.cpu())
            if batch_indices is None:
                batch_indices = torch.arange(offset, offset + labels.numel())
            indices.append(torch.as_tensor(batch_indices).cpu().long())
            offset += labels.numel()
    if was_training: network.train()
    if ema_training: ema.train()
    ctx[save_as] = {"probabilities": torch.cat(probabilities), "ema_probabilities": torch.cat(ema_probabilities), "targets": torch.cat(targets), "indices": torch.cat(indices)}


@block(
    id="fine_scs_select",
    name="FINE: SCS Clean Selection",
    category="Paper Specific",
    description="Update the class-adaptive SCS threshold and store clean/noisy decisions by stable index.",
    params={"state": {"type": "slot", "default": "fine_state"}, "snapshot": {"type": "slot", "default": "fine_snapshot"}},
    requires=("state", "snapshot"), provides=(), placement=("epoch",), stage="train", ui_group="⑦ 样本选择",
    formula="clean_i=1[p_tilde_i(y_tilde_i) >= tau_global * local_class_modulation]",
    formula_ref="FINE Self-Adaptive Class Selection (SCS)",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def fine_scs_select(ctx: ScratchContext, state: str = "fine_state", snapshot: str = "fine_snapshot") -> None:
    values = ctx[snapshot]
    mask = ctx[state]["scs"].select_epoch(values["ema_probabilities"], values["targets"])
    ctx[state]["clean_by_index"] = {int(i): bool(v) for i, v in zip(values["indices"], mask)}
    ctx[state]["pseudo_by_index"] = {int(i): int(v) for i, v in zip(values["indices"], values["ema_probabilities"].argmax(1))}


@block(
    id="fine_scr_reweight",
    name="FINE: SCR Confidence Weights",
    category="Paper Specific",
    description="Update class-wise confidence statistics and store SCR weights by stable index.",
    params={"state": {"type": "slot", "default": "fine_state"}, "snapshot": {"type": "slot", "default": "fine_snapshot"}},
    requires=("state", "snapshot"), provides=(), placement=("epoch",), stage="train", ui_group="⑥ 后验与权重",
    formula="w_i=exp(-(max p_i-mu_hat_c)^2/(2 sigma_hat_c^2/n_sigma^2))",
    formula_ref="FINE Self-Adaptive Confidence Reweighting (SCR)",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def fine_scr_reweight(ctx: ScratchContext, state: str = "fine_state", snapshot: str = "fine_snapshot") -> None:
    values = ctx[snapshot]
    weights = ctx[state]["scr"].weights(values["ema_probabilities"])
    ctx[state]["weight_by_index"] = {int(i): float(v) for i, v in zip(values["indices"], weights)}


@block(
    id="fine_prepare_batch_targets",
    name="FINE: Prepare Stable Batch Targets",
    category="Paper Specific",
    description="Lookup SCS clean masks, EMA pseudo labels, and SCR weights for the current stable-index batch.",
    params={"state": {"type": "slot", "default": "fine_state"}, "indices": {"type": "slot", "default": "indices"}, "labels": {"type": "slot", "default": "labels"}, "clean_as": {"type": "slot", "default": "fine_clean"}, "pseudo_as": {"type": "slot", "default": "fine_pseudo"}, "weights_as": {"type": "slot", "default": "fine_weights"}},
    requires=("state", "indices", "labels"), provides=("clean_as", "pseudo_as", "weights_as"), placement=("batch",), stage="train", ui_group="⑦ 样本选择",
)
def fine_prepare_batch_targets(ctx: ScratchContext, state: str = "fine_state", indices: str = "indices", labels: str = "labels", clean_as: str = "fine_clean", pseudo_as: str = "fine_pseudo", weights_as: str = "fine_weights") -> None:
    torch, _ = _torch()
    values = [int(i) for i in ctx[indices].detach().cpu()] if ctx.get(indices) is not None else list(range(int(ctx[labels].shape[0])))
    holder = ctx[state]
    ctx[clean_as] = torch.as_tensor([holder["clean_by_index"].get(i, True) for i in values], dtype=torch.bool, device=ctx[labels].device)
    ctx[pseudo_as] = torch.as_tensor([holder["pseudo_by_index"].get(i, int(ctx[labels][j])) for j, i in enumerate(values)], dtype=torch.long, device=ctx[labels].device)
    ctx[weights_as] = torch.as_tensor([holder["weight_by_index"].get(i, 1.0) for i in values], dtype=torch.float32, device=ctx[labels].device)


@block(
    id="fine_warmup_loss",
    name="FINE: Warm-up Objective",
    category="Paper Specific",
    description="Compute official warm-up cross entropy plus confidence penalty before SED robust training.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "labels"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L_warmup=CE(f(x),y~)+mean_c p_c log p_c",
    formula_ref="FINE official warm-up objective",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def fine_warmup_loss(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", save_as: str = "loss") -> None:
    torch, F = _torch()
    probabilities = F.softmax(ctx[logits], dim=1).clamp_min(1e-12)
    ctx[save_as] = F.cross_entropy(ctx[logits], ctx[labels].long()) + (probabilities * probabilities.log()).sum(dim=1).mean()


@block(
    id="fine_robust_loss",
    name="FINE: Robust SED/FINE Objective",
    category="Paper Specific",
    description="Compose clean CE, SCR-weighted pseudo-label consistency, and the official FINE MU/NL regularizer.",
    params={"logits": {"type": "slot", "default": "logits"}, "strong_logits": {"type": "slot", "default": "strong_logits"}, "labels": {"type": "slot", "default": "labels"}, "clean": {"type": "slot", "default": "fine_clean"}, "pseudo": {"type": "slot", "default": "fine_pseudo"}, "weights": {"type": "slot", "default": "fine_weights"}, "state": {"type": "slot", "default": "fine_state"}, "alpha": {"type": "float", "default": 1.0, "min": 0.0}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "strong_logits", "labels", "clean", "pseudo", "weights", "state"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L=CE_clean+alpha w_i CE(strong_i,pseudo_i)+FINE_MU/NL(rejected)",
    formula_ref="FINE robust phase objective and Active Forgetting/Noise Suppression regularizers",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def fine_robust_loss(ctx: ScratchContext, logits: str = "logits", strong_logits: str = "strong_logits", labels: str = "labels", clean: str = "fine_clean", pseudo: str = "fine_pseudo", weights: str = "fine_weights", state: str = "fine_state", alpha: float = 1.0, save_as: str = "loss") -> None:
    torch, F = _torch()
    clean_mask, rejected = ctx[clean].bool(), ~ctx[clean].bool()
    clean_loss = F.cross_entropy(ctx[logits][clean_mask], ctx[labels][clean_mask].long()) if bool(clean_mask.any()) else ctx[logits].sum() * 0.0
    ssl = (F.cross_entropy(ctx[strong_logits], ctx[pseudo].long(), reduction="none") * ctx[weights].detach()).mean()
    fine_loss = ctx[state]["regularizer"](ctx[logits], ctx[labels], rejected_mask=rejected, pseudo_labels=ctx[pseudo])
    ctx[save_as] = clean_loss + float(alpha) * ssl + fine_loss


@block(
    id="fine_ema_update",
    name="FINE: Update EMA Teacher",
    category="Paper Specific",
    description="Update the non-trainable EMA copy after each student optimizer step.",
    params={"model": {"type": "slot", "default": "model"}, "state": {"type": "slot", "default": "fine_state"}},
    requires=("model", "state"), provides=(), placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="EMA_t=m EMA_{t-1}+(1-m)theta_t",
    formula_ref="FINE model_ema lifecycle",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def fine_ema_update(ctx: ScratchContext, model: str = "model", state: str = "fine_state") -> None:
    ctx[state]["ema"].update(ctx[model])


@block(
    id="fine_feature_filter",
    name="FINE: Feature Filtering",
    category="Paper Specific",
    description="Filter feature embeddings by robust norm threshold.",
    params={"features": {"type": "slot", "default": "features"}, "quantile": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "fine_mask"}},
    requires=("features",),
    provides=("save_as",),
)
def fine_feature_filter(ctx: ScratchContext, features: str = "features", quantile: float = 0.5, save_as: str = "fine_mask") -> None:
    values = ctx[features].norm(dim=-1)
    ctx[save_as] = values >= values.quantile(float(quantile))


@block(
    id="cnlcu_uncertainty",
    name="CNLCU: Uncertainty Selection",
    category="Paper Specific",
    description="Select low-uncertainty examples from a loss history vector.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "cnlcu_mask"}},
    requires=("losses",),
    provides=("save_as",),
)
def cnlcu_uncertainty(ctx: ScratchContext, losses: str = "loss_per_sample", threshold: float = 0.5, save_as: str = "cnlcu_mask") -> None:
    values = ctx[losses]
    confidence = torch_sigmoid(-(values - values.median()))
    ctx[save_as] = confidence >= float(threshold)


@block(
    id="revise_transition",
    name="T-Revision: Revise Transition",
    category="Transition",
    description="Blend an estimated transition matrix with its row-stochastic identity prior.",
    params={"transition": {"type": "slot", "default": "transition"}, "strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "revised_transition"}},
    requires=("transition",),
    provides=("save_as",),
)
def revise_transition(ctx: ScratchContext, transition: str = "transition", strength: float = 0.5, save_as: str = "revised_transition") -> None:
    torch, _ = _torch()
    matrix = ctx[transition]
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    ctx[save_as] = _row_normalize((1.0 - float(strength)) * matrix + float(strength) * identity)


@block(
    id="dld_label_diffusion",
    name="DLD: Directional Label Diffusion",
    category="Paper Specific",
    description="Diffuse one-hot labels toward a feature-induced directional posterior.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "dld_labels"}},
    requires=("features", "labels"),
    provides=("save_as",),
)
def dld_label_diffusion(ctx: ScratchContext, features: str = "features", labels: str = "labels", save_as: str = "dld_labels") -> None:
    torch, F = _torch()
    values, targets = ctx[features], ctx[labels].long()
    classes = int(ctx.get("num_classes", int(targets.max().item()) + 1))
    prototypes = torch.stack([values[targets == c].mean(0) if bool((targets == c).any()) else torch.zeros(values.shape[1], device=values.device) for c in range(classes)])
    posterior = F.softmax(values @ prototypes.t(), -1)
    ctx[save_as] = 0.5 * F.one_hot(targets, classes).float() + 0.5 * posterior


@block(
    id="create_volmin_transition",
    name="VolMinNet Trainable Transition",
    category="Transition",
    description="Create VolMinNet's fixed-diagonal sigmoid off-diagonal transition parameterization.",
    params={"num_classes": {"type": "int", "default": 10, "min": 3}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "transition"}},
    requires=("device",), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="A_ii=1; A_ij=sigmoid(w_ij), then T=A/row_sum(A)",
    formula_ref="VolMinNet paper transition parameterization and initialization",
    paper="Provably End-to-end Label-noise Learning without Anchor Points",
)
def create_volmin_transition(ctx: ScratchContext, num_classes: int = 10, device: str = "device", save_as: str = "transition") -> None:
    from lnl_toolbox.algorithms.volminnet.transition import VolMinTransition
    ctx[save_as] = VolMinTransition(int(num_classes)).to(ctx[device])


@block(
    id="volmin_transition_matrix",
    name="VolMinNet Transition Matrix",
    category="Transition",
    description="Materialize the current trainable VolMinNet transition with the classifier dtype.",
    params={"transition": {"type": "slot", "default": "transition"}, "logits": {"type": "slot", "default": "logits"}, "save_as": {"type": "slot", "default": "transition_matrix"}},
    requires=("transition", "logits"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="T=A/row_sum(A)", formula_ref="VolMinNet paper transition normalization", paper="Provably End-to-end Label-noise Learning without Anchor Points",
)
def volmin_transition_matrix(ctx: ScratchContext, transition: str = "transition", logits: str = "logits", save_as: str = "transition_matrix") -> None:
    ctx[save_as] = ctx[transition].matrix(dtype=ctx[logits].dtype)


@block(
    id="pcse_materialize_transition",
    name="PCSE: Materialize Transition",
    category="Transition",
    description="Materialize the learned paper-VolMin transition before PCSE statistic recovery.",
    params={"transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "transition_matrix"}},
    requires=("transition",), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="T=A/row_sum(A)", formula_ref="PCSE transition_stage.paper_volmin", paper="Estimating Per-Class Statistics",
)
def pcse_materialize_transition(ctx: ScratchContext, transition: str = "transition", save_as: str = "transition_matrix") -> None:
    ctx[save_as] = ctx[transition].matrix(dtype=__import__("torch").float64)


@block(
    id="volmin_noisy_nll",
    name="VolMinNet Noisy NLL",
    category="Correction",
    description="Compute the noisy-label negative log-likelihood under the clean-to-noisy transition.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "transition": {"type": "slot", "default": "transition_matrix"}, "save_as": {"type": "slot", "default": "loss_per_sample"}},
    requires=("logits", "labels", "transition"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="p_tilde(y|x)=sum_c p(c|x)T_cy; l=-log p_tilde(y_tilde|x)",
    formula_ref="VolMinNet paper Eq. (7) noisy classification term", paper="Provably End-to-end Label-noise Learning without Anchor Points",
)
def volmin_noisy_nll(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", transition: str = "transition_matrix", save_as: str = "loss_per_sample") -> None:
    torch, F = _torch()
    matrix = ctx[transition].to(ctx[logits])
    noisy_log_probability = torch.logsumexp(
        F.log_softmax(ctx[logits], dim=1)[:, :, None] + torch.log(matrix)[None, :, :], dim=1
    )
    ctx[save_as] = -noisy_log_probability.gather(1, ctx[labels].long()[:, None]).squeeze(1)


@block(
    id="volmin_positive_logdet",
    name="VolMinNet Positive Logdet",
    category="Correction",
    description="Compute the positive log-determinant minimum-volume term from the current transition.",
    params={"transition": {"type": "slot", "default": "transition_matrix"}, "save_as": {"type": "slot", "default": "volume_logdet"}},
    requires=("transition",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="V(T)=log det(T), det(T)>0", formula_ref="VolMinNet paper_positive_logdet fidelity", paper="Provably End-to-end Label-noise Learning without Anchor Points",
)
def volmin_positive_logdet(ctx: ScratchContext, transition: str = "transition_matrix", save_as: str = "volume_logdet") -> None:
    torch, _ = _torch()
    sign, logdet = torch.linalg.slogdet(ctx[transition])
    if not bool(torch.isfinite(sign).item()) or float(sign.detach().item()) <= 0.0 or not bool(torch.isfinite(logdet).item()):
        raise ValueError("VolMinNet transition determinant must be finite and positive")
    ctx[save_as] = logdet


@block(
    id="volmin_objective_composition",
    name="VolMinNet Objective Composition",
    category="Correction",
    description="Add the noisy NLL and positive logdet volume term with the formal coefficient.",
    params={"classification_loss": {"type": "slot", "default": "loss_per_sample"}, "volume": {"type": "slot", "default": "volume_logdet"}, "volume_weight": {"type": "float", "default": 0.0001, "min": 0.0}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("classification_loss", "volume"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="J=mean_i l_i + lambda_volume log det(T)", formula_ref="VolMinNet paper Eq. (7)", paper="Provably End-to-end Label-noise Learning without Anchor Points",
)
def volmin_objective_composition(ctx: ScratchContext, classification_loss: str = "loss_per_sample", volume: str = "volume_logdet", volume_weight: float = 0.0001, save_as: str = "loss") -> None:
    ctx[save_as] = ctx[classification_loss].mean() + float(volume_weight) * ctx[volume]


@block(
    id="evaluate_volminnet_noisy",
    name="Evaluate VolMinNet Noisy Validation",
    category="Evaluation",
    description="Evaluate noisy-validation loss and observed-label accuracy under the learned transition.",
    params={"model": {"type": "slot", "default": "model"}, "transition": {"type": "slot", "default": "transition"}, "loader": {"type": "slot", "default": "validation_loader"}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "validation_loss"}},
    requires=("model", "transition", "loader", "device"), provides=("save_as", "validation_accuracy", "metrics"), placement=("epoch", "top"), stage="evaluate", ui_group="⑨ 评估",
    formula="L_val=mean[-log sum_c p(c|x)T_c,y_tilde]", formula_ref="VolMinNet noisy-validation checkpoint criterion", paper="Provably End-to-end Label-noise Learning without Anchor Points",
)
def evaluate_volminnet_noisy(ctx: ScratchContext, model: str = "model", transition: str = "transition", loader: str = "validation_loader", device: str = "device", save_as: str = "validation_loss") -> None:
    torch, F = _torch()
    network = ctx[model]
    matrix_module = ctx[transition]
    was_training = network.training
    network.eval()
    matrix_module.eval()
    loss_sum = 0.0
    correct = total = 0
    max_batches = (ctx.get("_runtime_limits") or {}).get("max_batches")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(ctx[loader]):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            inputs = batch["input"].to(ctx[device])
            labels = batch["target"].to(ctx[device]).long()
            logits = network(inputs)
            matrix = matrix_module.matrix(dtype=logits.dtype)
            noisy_log_probability = torch.logsumexp(F.log_softmax(logits, dim=1)[:, :, None] + torch.log(matrix)[None, :, :], dim=1)
            loss_sum += float(F.nll_loss(noisy_log_probability, labels, reduction="sum").item())
            correct += int(noisy_log_probability.argmax(1).eq(labels).sum().item())
            total += int(labels.numel())
    network.train(was_training)
    if total == 0:
        raise ValueError("VolMinNet validation loader is empty")
    value = loss_sum / total
    ctx[save_as] = value
    ctx["validation_accuracy"] = correct / total
    ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "validation_loss": value, "validation_accuracy": correct / total})


@block(
    id="track_best_volminnet",
    name="Keep VolMinNet Minimum Validation State",
    category="Evaluation",
    description="Keep classifier and transition states from the lowest noisy-validation loss.",
    params={"model": {"type": "slot", "default": "model"}, "transition": {"type": "slot", "default": "transition"}, "metric": {"type": "slot", "default": "validation_loss"}, "save_as": {"type": "slot", "default": "best_volmin_state"}},
    requires=("model", "transition", "metric"), provides=("save_as", "best_epoch"), placement=("epoch",), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def track_best_volminnet(ctx: ScratchContext, model: str = "model", transition: str = "transition", metric: str = "validation_loss", save_as: str = "best_volmin_state") -> None:
    import copy
    value = float(ctx[metric])
    if value < float(ctx.get("best_volminnet_validation_loss", float("inf"))):
        ctx["best_volminnet_validation_loss"] = value
        ctx["best_epoch"] = int(ctx.get("epoch", 0))
        ctx[save_as] = {"model": copy.deepcopy(ctx[model].state_dict()), "transition": copy.deepcopy(ctx[transition].state_dict())}


@block(
    id="restore_best_volminnet",
    name="Restore Best VolMinNet State",
    category="Evaluation",
    description="Restore classifier and transition states selected by noisy validation loss.",
    params={"model": {"type": "slot", "default": "model"}, "transition": {"type": "slot", "default": "transition"}, "state": {"type": "slot", "default": "best_volmin_state"}},
    requires=("model", "transition", "state"), placement=("top",), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def restore_best_volminnet(ctx: ScratchContext, model: str = "model", transition: str = "transition", state: str = "best_volmin_state") -> None:
    ctx[model].load_state_dict(ctx[state]["model"])
    ctx[transition].load_state_dict(ctx[state]["transition"])


@block(
    id="snapshot_t_revision_posterior",
    name="T-Revision Posterior Snapshot",
    category="Transition",
    description="Collect the stage-1 noisy posterior on the non-augmented train-eval split with stable sample indices.",
    params={"model": {"type": "slot", "default": "model"}, "prepared_data": {"type": "slot", "default": "prepared_data"}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "t_revision_posterior"}},
    requires=("model", "prepared_data", "device"), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="q(x)=softmax(f_theta*(x)); snapshot=(q(x), y_tilde, index)", formula_ref="T-Revision Algorithm 1, Stage 1 posterior estimation", paper="Are Anchor Points Really Indispensable in Label-Noise Learning?",
)
def snapshot_t_revision_posterior(ctx: ScratchContext, model: str = "model", prepared_data: str = "prepared_data", device: str = "device", save_as: str = "t_revision_posterior") -> None:
    from lnl_toolbox.data import DataRole
    from lnl_toolbox.training.snapshots import collect_posterior_snapshot

    prepared = ctx[prepared_data]
    loader = prepared.loader(DataRole.TRAIN_EVAL, shuffle=False)
    ctx[save_as] = collect_posterior_snapshot(
        ctx[model], loader, ctx[device], dataset=str(prepared.dataset), split="train"
    )


@block(
    id="initialize_t_revision_transition",
    name="T-Revision Pseudo-Anchor Transition",
    category="Transition",
    description="Initialize T-hat by taking the highest noisy posterior example for each class, with stable-index tie breaking.",
    params={"posterior": {"type": "slot", "default": "t_revision_posterior"}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "transition"}},
    requires=("posterior", "device"), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="T_hat[i,:]=q(x_i) where x_i=argmax_x q_i(x)", formula_ref="T-Revision Algorithm 1, Stage 1 transition initialization; Eq. (1)", paper="Are Anchor Points Really Indispensable in Label-Noise Learning?",
)
def initialize_t_revision_transition(ctx: ScratchContext, posterior: str = "t_revision_posterior", device: str = "device", save_as: str = "transition") -> None:
    import torch
    from lnl_toolbox.noise.estimators import AnchorTransitionEstimator

    artifact = AnchorTransitionEstimator().estimate(ctx[posterior])
    ctx[save_as] = torch.tensor(artifact.matrix, dtype=torch.float32, device=ctx[device])
    ctx["t_revision_transition_artifact"] = artifact


@block(
    id="create_t_revision_revision",
    name="T-Revision Additive Slack",
    category="Transition",
    description="Create the unconstrained zero-initialized additive transition slack Delta T while keeping T-hat fixed.",
    params={"transition": {"type": "slot", "default": "transition"}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "revision"}},
    requires=("transition", "device"), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="T=T_hat+Delta T; Delta T_0=0", formula_ref="T-Revision paper Section 3.3 transition revision", paper="Are Anchor Points Really Indispensable in Label-Noise Learning?",
)
def create_t_revision_revision(ctx: ScratchContext, transition: str = "transition", device: str = "device", save_as: str = "revision") -> None:
    from lnl_toolbox.algorithms.t_revision.transition import AdditiveTransitionRevision

    ctx[save_as] = AdditiveTransitionRevision(ctx[transition]).to(ctx[device])


@block(
    id="t_revision_revised_transition",
    name="T-Revision Current Transition",
    category="Transition",
    description="Materialize T-hat plus the learned additive slack for the revision objective.",
    params={"revision": {"type": "slot", "default": "revision"}, "save_as": {"type": "slot", "default": "transition"}},
    requires=("revision",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="T=T_hat+Delta T", formula_ref="T-Revision paper Section 3.3 transition revision", paper="Are Anchor Points Really Indispensable in Label-Noise Learning?",
)
def t_revision_revised_transition(ctx: ScratchContext, revision: str = "revision", save_as: str = "transition") -> None:
    ctx[save_as] = ctx[revision]()


@block(
    id="t_revision_noisy_probability",
    name="T-Revision Noisy Probability",
    category="Correction",
    description="Map clean posterior g(x) through the row-vector clean-to-noisy transition.",
    params={"probabilities": {"type": "slot", "default": "probabilities"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "noisy_probabilities"}},
    requires=("probabilities", "transition"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="p_hat(Y_tilde|x)=g(x)T", formula_ref="T-Revision paper Eq. (3) denominator model T^T g in column notation", paper="Are Anchor Points Really Indispensable in Label-Noise Learning?",
)
def t_revision_noisy_probability(ctx: ScratchContext, probabilities: str = "probabilities", transition: str = "transition", save_as: str = "noisy_probabilities") -> None:
    ctx[save_as] = ctx[probabilities] @ ctx[transition].to(ctx[probabilities])


@block(
    id="t_revision_importance_ratio",
    name="T-Revision Importance Ratio",
    category="Weighting",
    description="Compute the clean-to-noisy posterior ratio for each observed noisy label without clipping or normalization.",
    params={"probabilities": {"type": "slot", "default": "probabilities"}, "noisy_probabilities": {"type": "slot", "default": "noisy_probabilities"}, "labels": {"type": "slot", "default": "labels"}, "denominator_floor": {"type": "float", "default": 1.0e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "sample_weights"}, "denominators_as": {"type": "slot", "default": "sample_denominators"}},
    requires=("probabilities", "noisy_probabilities", "labels"), provides=("save_as", "denominators_as"), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="w_i=g_ytilde_i(x_i)/(g(x_i)T)_ytilde_i", formula_ref="T-Revision paper Eq. (3)", paper="Are Anchor Points Really Indispensable in Label-Noise Learning?",
)
def t_revision_importance_ratio(ctx: ScratchContext, probabilities: str = "probabilities", noisy_probabilities: str = "noisy_probabilities", labels: str = "labels", denominator_floor: float = 1.0e-12, save_as: str = "sample_weights", denominators_as: str = "sample_denominators") -> None:
    torch, _ = _torch()
    clean = ctx[probabilities]
    noisy = ctx[noisy_probabilities]
    indices = ctx[labels].long().view(-1, 1)
    numerator = clean.gather(1, indices).squeeze(1)
    denominator = noisy.gather(1, indices).squeeze(1)
    if bool((denominator <= float(denominator_floor)).any().item()) or not bool(torch.isfinite(denominator).all().item()):
        raise ValueError("T-Revision importance-ratio denominator must be finite and above denominator_floor")
    weights = numerator / denominator
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("T-Revision importance ratios must be finite")
    ctx[save_as] = weights
    ctx[denominators_as] = denominator


@block(
    id="t_revision_weighted_objective",
    name="T-Revision Weighted Risk",
    category="Weighting",
    description="Compose the mean importance-weighted per-sample cross entropy, with optional detached ratios for classifier initialization.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "weights": {"type": "slot", "default": "sample_weights"}, "detach_ratio": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("losses", "weights"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="R_bar_n,w=1/n sum_i w_i l(f(x_i), ytilde_i)", formula_ref="T-Revision paper Eq. (3)", paper="Are Anchor Points Really Indispensable in Label-Noise Learning?",
)
def t_revision_weighted_objective(ctx: ScratchContext, losses: str = "loss_per_sample", weights: str = "sample_weights", detach_ratio: bool = False, save_as: str = "loss") -> None:
    factors = ctx[weights].detach() if bool(detach_ratio) else ctx[weights]
    values = ctx[losses]
    if values.shape != factors.shape:
        raise ValueError("T-Revision weights and per-sample losses must have the same shape")
    ctx[save_as] = (factors * values).mean()


@block(
    id="create_t_revision_optimizer",
    name="Create T-Revision Joint Optimizer",
    category="Optimization",
    description="Create the formal Adam optimizer over classifier parameters and additive transition slack.",
    params={"model": {"type": "slot", "default": "model"}, "revision": {"type": "slot", "default": "revision"}, "lr": {"type": "float", "default": 5.0e-7, "min": 0.0}, "weight_decay": {"type": "float", "default": 0.0001, "min": 0.0}, "save_as": {"type": "slot", "default": "optimizer"}},
    requires=("model", "revision"), provides=("save_as",), placement=("top",), stage="setup", ui_group="② 初始化",
)
def create_t_revision_optimizer(ctx: ScratchContext, model: str = "model", revision: str = "revision", lr: float = 5.0e-7, weight_decay: float = 0.0001, save_as: str = "optimizer") -> None:
    torch, _ = _torch()
    ctx[save_as] = torch.optim.Adam(list(ctx[model].parameters()) + list(ctx[revision].parameters()), lr=float(lr), weight_decay=float(weight_decay))


@block(
    id="evaluate_t_revision_noisy",
    name="Evaluate T-Revision Noisy Validation",
    category="Evaluation",
    description="Evaluate noisy-validation likelihood and observed-label accuracy under a fixed or revised transition.",
    params={"model": {"type": "slot", "default": "model"}, "transition": {"type": "slot", "default": "transition"}, "loader": {"type": "slot", "default": "validation_loader"}, "device": {"type": "slot", "default": "device"}, "denominator_floor": {"type": "float", "default": 1.0e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "validation_loss"}},
    requires=("model", "transition", "loader", "device"), provides=("save_as", "validation_accuracy", "metrics"), placement=("top", "epoch"), stage="evaluate", ui_group="⑨ 评估",
    formula="L_val=mean[-log(g(x)T)_ytilde]", formula_ref="T-Revision paper Algorithm 1 noisy validation criterion", paper="Are Anchor Points Really Indispensable in Label-Noise Learning?",
)
def evaluate_t_revision_noisy(ctx: ScratchContext, model: str = "model", transition: str = "transition", loader: str = "validation_loader", device: str = "device", denominator_floor: float = 1.0e-12, save_as: str = "validation_loss") -> None:
    torch, _ = _torch()
    network = ctx[model]
    matrix_source = ctx[transition]
    matrix = matrix_source() if callable(matrix_source) and not torch.is_tensor(matrix_source) else matrix_source
    matrix = matrix.to(ctx[device])
    was_training = network.training
    network.eval()
    total = 0
    loss_sum = 0.0
    correct = 0
    max_batches = (ctx.get("_runtime_limits") or {}).get("max_batches")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(ctx[loader]):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            inputs = batch["input"].to(ctx[device])
            labels = batch["target"].to(ctx[device]).long()
            clean = torch.softmax(network(inputs), dim=1)
            noisy = clean @ matrix.to(clean)
            observed = noisy.gather(1, labels[:, None]).squeeze(1)
            if bool((observed <= float(denominator_floor)).any().item()) or not bool(torch.isfinite(observed).all().item()):
                raise ValueError("T-Revision validation noisy probability is invalid")
            loss_sum += float((-observed.log()).sum().item())
            correct += int(noisy.argmax(1).eq(labels).sum().item())
            total += int(labels.numel())
    if was_training:
        network.train()
    if total == 0:
        raise ValueError("T-Revision validation loader is empty")
    value = loss_sum / total
    ctx[save_as] = value
    ctx["validation_accuracy"] = correct / total
    ctx.setdefault("metrics", []).append({"epoch": int(ctx.get("epoch", 0)), "validation_loss": value, "validation_accuracy": correct / total})


@block(
    id="track_best_t_revision",
    name="Keep Best T-Revision State",
    category="Evaluation",
    description="Keep classifier and additive slack states from the highest noisy-validation accuracy.",
    params={"model": {"type": "slot", "default": "model"}, "revision": {"type": "slot", "default": "revision"}, "metric": {"type": "slot", "default": "validation_accuracy"}, "save_as": {"type": "slot", "default": "best_t_revision_state"}},
    requires=("model", "revision", "metric"), provides=("save_as", "best_epoch"), placement=("epoch",), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def track_best_t_revision(ctx: ScratchContext, model: str = "model", revision: str = "revision", metric: str = "validation_accuracy", save_as: str = "best_t_revision_state") -> None:
    import copy
    value = float(ctx[metric])
    if value > float(ctx.get("best_t_revision_validation_accuracy", float("-inf"))):
        ctx["best_t_revision_validation_accuracy"] = value
        ctx["best_epoch"] = int(ctx.get("epoch", 0))
        ctx[save_as] = {"model": copy.deepcopy(ctx[model].state_dict()), "revision": copy.deepcopy(ctx[revision].state_dict())}


@block(
    id="restore_best_t_revision",
    name="Restore Best T-Revision State",
    category="Evaluation",
    description="Restore classifier and additive slack states selected by noisy validation.",
    params={"model": {"type": "slot", "default": "model"}, "revision": {"type": "slot", "default": "revision"}, "state": {"type": "slot", "default": "best_t_revision_state"}},
    requires=("model", "revision", "state"), provides=(), placement=("top",), stage="evaluate", ui_group="⑨ 评估", beginner_visible=False,
)
def restore_best_t_revision(ctx: ScratchContext, model: str = "model", revision: str = "revision", state: str = "best_t_revision_state") -> None:
    ctx[model].load_state_dict(ctx[state]["model"])
    ctx[revision].load_state_dict(ctx[state]["revision"])


@block(
    id="volminnet_objective",
    name="VolMinNet: Minimum-volume Objective",
    category="Paper Specific",
    description="Legacy combined VolMinNet objective; use the decomposed VolMinNet blocks in formal recipes.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "transition": {"type": "slot", "default": "transition"}, "volume_weight": {"type": "float", "default": 0.0001, "min": 0.0}, "save_as": {"type": "slot", "default": "volmin_loss"}},
    requires=("logits", "labels", "transition"), provides=("save_as",), beginner_visible=False,
    formula="J=mean_i[-log p_tilde(y_tilde|x)] + lambda log det(T)", formula_ref="VolMinNet paper Eq. (7)", paper="Provably End-to-end Label-noise Learning without Anchor Points",
)
def volminnet_objective(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", transition: str = "transition", volume_weight: float = 0.0001, save_as: str = "volmin_loss") -> None:
    torch, F = _torch()
    matrix = ctx[transition].to(ctx[logits])
    noisy_log_probability = torch.logsumexp(F.log_softmax(ctx[logits], dim=1)[:, :, None] + torch.log(matrix)[None, :, :], dim=1)
    nll = F.nll_loss(noisy_log_probability, ctx[labels].long(), reduction="mean")
    sign, logdet = torch.linalg.slogdet(matrix)
    if float(sign.detach().item()) <= 0.0:
        raise ValueError("VolMinNet transition determinant must be positive")
    ctx[save_as] = nll + float(volume_weight) * logdet


@block(
    id="create_upm_state",
    name="UPM: Create Confusing-Probability State",
    category="State",
    description="Publish the stage-1 posterior-derived psi and initialize per-example eta keyed by stable training indices.",
    params={"prepared_data": {"type": "slot", "default": "prepared_data"}, "model": {"type": "slot", "default": "stage1_model"}, "loader": {"type": "slot", "default": "train_eval_loader"}, "num_classes": {"type": "int", "default": 10, "min": 2}, "eta_init": {"type": "float", "default": 0.01, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "upm_state"}},
    requires=("prepared_data", "model", "loader"), provides=("save_as",), placement=("top",), stage="setup", ui_group="② 初始化",
    formula="psi_i=P_stage1(y~_i|x_i); eta_i<-eta_0",
    formula_ref="UPM Stage-1 best source and psi publication",
    paper="Universal Probability Model for Label Noise",
)
def create_upm_state(ctx: ScratchContext, prepared_data: str = "prepared_data", model: str = "stage1_model", loader: str = "train_eval_loader", num_classes: int = 10, eta_init: float = 0.01, save_as: str = "upm_state") -> None:
    import torch
    from lnl_toolbox.noise.upm import UPMNoiseState
    prepared = ctx[prepared_data]
    expected = torch.as_tensor(prepared.train_indices, dtype=torch.long)
    probabilities = {}
    network = ctx[model]
    network.eval()
    device = ctx.get("device", torch.device("cpu"))
    with torch.no_grad():
        for batch in ctx[loader]:
            if isinstance(batch, dict):
                images = batch.get("images", batch.get("inputs", batch.get("x")))
                labels = batch.get("labels", batch.get("targets", batch.get("y")))
                indices = batch.get("indices", batch.get("index"))
            else:
                images, labels, indices = batch[:3]
            logits = network(images.to(device))
            if isinstance(logits, tuple):
                logits = logits[0]
            values = torch.softmax(logits, dim=1).gather(1, labels.to(device).long()[:, None]).squeeze(1).cpu()
            for index, value in zip(indices.long().cpu().tolist(), values.tolist()):
                probabilities[int(index)] = float(value)
    psi = torch.tensor([probabilities.get(int(index), 1.0 / float(num_classes)) for index in expected], dtype=torch.float32)
    ctx[save_as] = UPMNoiseState(expected, psi, torch.full((expected.numel(),), float(eta_init)), int(num_classes))


@block(
    id="upm_clean_posterior",
    name="UPM: Estimate Clean Posterior",
    category="Paper Specific",
    description="Compute Eq. (8) clean-label posterior from classifier logits, frozen psi, and current eta.",
    params={"state": {"type": "slot", "default": "upm_state"}, "logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "indices": {"type": "slot", "default": "indices"}, "save_as": {"type": "slot", "default": "clean_posterior"}},
    requires=("state", "logits", "labels", "indices"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="q(y|x,y~) ∝ h(y|x)[(1-eta)1[y=y~]+eta psi]",
    formula_ref="UPM Eq. (8)", paper="Universal Probability Model for Label Noise",
)
def upm_clean_posterior(ctx: ScratchContext, state: str = "upm_state", logits: str = "logits", labels: str = "labels", indices: str = "indices", save_as: str = "clean_posterior") -> None:
    import torch
    from lnl_toolbox.algorithms.upm.objective import predict_true_posterior
    psi, eta = ctx[state].lookup(ctx[indices].detach())
    probabilities = torch.softmax(ctx[logits].detach(), dim=1)
    ctx[save_as] = predict_true_posterior(probabilities, ctx[labels].long(), psi.to(probabilities), eta.to(probabilities))


@block(
    id="upm_update_eta",
    name="UPM: Update Confusing Probabilities",
    category="Paper Specific",
    description="Apply Eq. (11) eta ascent and [0,1] projection after posterior estimation.",
    params={"state": {"type": "slot", "default": "upm_state"}, "indices": {"type": "slot", "default": "indices"}, "posterior": {"type": "slot", "default": "clean_posterior"}, "labels": {"type": "slot", "default": "labels"}, "learning_rate": {"type": "float", "default": 0.7, "min": 0.0}},
    requires=("state", "indices", "posterior", "labels"), provides=(), placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="eta<-Pi_[0,1](eta+lr d log p(y~|x)/d eta)", formula_ref="UPM Eq. (11)-(12)", paper="Universal Probability Model for Label Noise",
)
def upm_update_eta(ctx: ScratchContext, state: str = "upm_state", indices: str = "indices", posterior: str = "clean_posterior", labels: str = "labels", learning_rate: float = 0.7) -> None:
    import torch
    from lnl_toolbox.algorithms.upm.objective import update_confusing_probability
    psi, eta = ctx[state].lookup(ctx[indices].detach())
    updated = update_confusing_probability(eta.to(ctx[posterior]), ctx[posterior].detach(), ctx[labels].detach(), psi.to(ctx[posterior]), learning_rate=float(learning_rate), epsilon=1e-8)
    updated = updated.to(torch.device("cpu"))
    ctx[state].update_eta(ctx[indices].detach(), updated)


@block(
    id="upm_soft_target_loss",
    name="UPM: Soft Target Objective",
    category="Loss",
    description="Train the fresh main model against the detached clean posterior target.",
    params={"logits": {"type": "slot", "default": "logits"}, "posterior": {"type": "slot", "default": "clean_posterior"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "posterior"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L=-sum_j q_j log softmax(f(x))_j",
    formula_ref="UPM soft-target objective",
    paper="Universal Probability Model for Label Noise",
)
def upm_soft_target_loss(ctx: ScratchContext, logits: str = "logits", posterior: str = "clean_posterior", save_as: str = "loss") -> None:
    from lnl_toolbox.algorithms.upm.objective import soft_target_cross_entropy
    ctx[save_as] = soft_target_cross_entropy(ctx[logits], ctx[posterior]).mean()


@block(
    id="upm_eta_update",
    name="UPM: Confusing Probability Update",
    category="Paper Specific",
    description="Update instance-dependent corruption probabilities with a projected step.",
    params={"probabilities": {"type": "slot", "default": "probabilities"}, "labels": {"type": "slot", "default": "labels"}, "step_size": {"type": "float", "default": 0.1, "min": 0.0}, "save_as": {"type": "slot", "default": "eta"}},
    requires=("probabilities", "labels"),
    provides=("save_as",),
)
def upm_eta_update(ctx: ScratchContext, probabilities: str = "probabilities", labels: str = "labels", step_size: float = 0.1, save_as: str = "eta") -> None:
    torch, _ = _torch()
    observed = ctx[probabilities].gather(1, ctx[labels].long()[:, None]).squeeze(1)
    ctx[save_as] = (observed + float(step_size) * (1.0 - observed)).clamp(0.0, 1.0).detach()


@block(
    id="create_lend_state",
    name="LEND: Create Stable History",
    category="Paper Specific",
    description="Create the stable-index label history used by the epoch-aware dilution lifecycle.",
    params={"prepared_data": {"type": "slot", "default": "prepared_data"}, "num_classes": {"type": "int", "default": 10, "min": 2}, "save_as": {"type": "slot", "default": "lend_state"}},
    requires=("prepared_data",), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="H_i^{t}=beta H_i^{t-1}+(1-beta)q_i^t, keyed by global index",
    formula_ref="LEND Eq. (5), LENDLabelHistory",
    paper="Learning with Noisy Labels by Exploiting the Label Distribution of Neighbors",
)
def create_lend_state(ctx: ScratchContext, prepared_data: str = "prepared_data", num_classes: int = 10, save_as: str = "lend_state") -> None:
    from lnl_toolbox.algorithms.lend.history import LENDLabelHistory
    indices = getattr(ctx[prepared_data], "train_indices", None)
    if indices is None:
        raise ValueError("LEND prepared data must expose canonical train_indices")
    ctx[save_as] = {"history": LENDLabelHistory(__import__("torch").as_tensor(indices), int(num_classes))}


@block(
    id="lend_feature_graph",
    name="LEND: Build Feature Graph",
    category="Paper Specific",
    description="Construct deterministic kNN similarity and normalized AᵀA graph from detached embeddings.",
    params={"features": {"type": "slot", "default": "features"}, "indices": {"type": "slot", "default": "indices"}, "k": {"type": "int", "default": 8, "min": 1}, "gamma": {"type": "float", "default": 1.0, "min": 0.0001}, "metric": {"type": "enum", "options": ["inner_product", "cosine", "euclidean"], "default": "inner_product"}, "normalize_features": {"type": "bool", "default": False}, "graph_as": {"type": "slot", "default": "lend_graph"}},
    requires=("features", "indices"), provides=("graph_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="A_{ij}=<h_i,h_j>^gamma for kNN; W=D^-1/2 A^T A D^-1/2",
    formula_ref="LEND Eq. (1)-(2)",
    paper="Learning with Noisy Labels by Exploiting the Label Distribution of Neighbors",
)
def lend_feature_graph(ctx: ScratchContext, features: str = "features", indices: str = "indices", k: int = 8, gamma: float = 1.0, metric: str = "inner_product", normalize_features: bool = False, graph_as: str = "lend_graph") -> None:
    from lnl_toolbox.algorithms.lend.graph import build_lend_similarity, normalize_lend_graph
    effective_k = min(int(k), int(ctx[features].shape[0]) - 1) if bool((ctx.get("_runtime_limits") or {}).get("fixture")) else int(k)
    sample_indices = ctx[indices].detach().to(ctx[features].device)
    adjacency = build_lend_similarity(ctx[features].detach(), sample_indices, k=effective_k, gamma=float(gamma), metric=str(metric), normalize_features=bool(normalize_features))
    ctx[graph_as] = normalize_lend_graph(adjacency)
    ctx["lend_adjacency"] = adjacency


@block(
    id="lend_dilute_labels",
    name="LEND: Dilute Labels",
    category="Paper Specific",
    description="Apply finite-step graph diffusion to the observed one-hot labels.",
    params={"graph": {"type": "slot", "default": "lend_graph"}, "labels": {"type": "slot", "default": "labels"}, "num_classes": {"type": "int", "default": 10, "min": 2}, "alpha": {"type": "float", "default": 0.99, "min": 0.0001, "max": 0.9999}, "steps": {"type": "int", "default": 10, "min": 1}, "save_as": {"type": "slot", "default": "diluted_labels"}},
    requires=("graph", "labels"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="Q^{(s+1)}=alpha WQ^{(s)}+(1-alpha)Q^{(s)}",
    formula_ref="LEND Eq. (3)",
    paper="Learning with Noisy Labels by Exploiting the Label Distribution of Neighbors",
)
def lend_dilute_labels(ctx: ScratchContext, graph: str = "lend_graph", labels: str = "labels", num_classes: int = 10, alpha: float = 0.99, steps: int = 10, save_as: str = "diluted_labels") -> None:
    from lnl_toolbox.algorithms.lend.dilution import dilute_labels
    torch, F = _torch()
    onehot = F.one_hot(ctx[labels].long(), int(num_classes)).float().detach()
    ctx[save_as] = dilute_labels(onehot, ctx[graph].to(onehot), alpha=float(alpha), steps=int(steps))


@block(
    id="lend_history_update",
    name="LEND: Update Label History",
    category="Paper Specific",
    description="Propose the beta-smoothed diluted labels for the current epoch without double-updating a sample.",
    params={"state": {"type": "slot", "default": "lend_state"}, "indices": {"type": "slot", "default": "indices"}, "diluted": {"type": "slot", "default": "diluted_labels"}, "beta": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "history_values"}},
    requires=("state", "indices", "diluted"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="H_i^t=beta H_i^{t-1}+(1-beta)Q_i^t (first observation=current)",
    formula_ref="LEND Eq. (5)",
    paper="Learning with Noisy Labels by Exploiting the Label Distribution of Neighbors",
)
def lend_history_update(ctx: ScratchContext, state: str = "lend_state", indices: str = "indices", diluted: str = "diluted_labels", beta: float = 0.9, save_as: str = "history_values") -> None:
    proposal = ctx[state]["history"].propose(ctx[indices].detach(), ctx[diluted].detach(), epoch=int(ctx.get("epoch", 0)), beta=float(beta))
    ctx[state]["pending_proposal"] = proposal
    ctx[save_as] = proposal.values.to(ctx[diluted].device)


@block(
    id="lend_select_agreement",
    name="LEND: Noisy/History Agreement",
    category="Sample Selection",
    description="Keep samples whose noisy label equals the diluted-history argmax and commit the epoch proposal.",
    params={"state": {"type": "slot", "default": "lend_state"}, "labels": {"type": "slot", "default": "labels"}, "history": {"type": "slot", "default": "history_values"}, "save_as": {"type": "slot", "default": "lend_selected"}},
    requires=("state", "labels", "history"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑦ 样本选择",
    formula="S_i=1[y~_i=argmax_c H_{i,c}]",
    formula_ref="LEND Eq. (6)",
    paper="Learning with Noisy Labels by Exploiting the Label Distribution of Neighbors",
)
def lend_select_agreement(ctx: ScratchContext, state: str = "lend_state", labels: str = "labels", history: str = "history_values", save_as: str = "lend_selected") -> None:
    from lnl_toolbox.algorithms.lend.selector import select_lend_samples
    mask = select_lend_samples(ctx[labels].long(), ctx[history].detach())
    ctx[save_as] = mask
    ctx[state]["history"].commit(ctx[state].pop("pending_proposal"))


@block(
    id="lend_selected_objective",
    name="LEND: Selected Objective",
    category="Loss",
    description="Reduce CE over selected examples by the actual batch size, preserving empty-selection skip-update semantics.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "selected": {"type": "slot", "default": "lend_selected"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("losses", "selected"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="J=1/B sum_{i in S} CE_i; J=0 when S is empty",
    formula_ref="LEND Eq. (7) executable batch reduction",
    paper="Learning with Noisy Labels by Exploiting the Label Distribution of Neighbors",
)
def lend_selected_objective(ctx: ScratchContext, losses: str = "loss_per_sample", selected: str = "lend_selected", save_as: str = "loss") -> None:
    values, mask = ctx[losses], ctx[selected].bool()
    ctx[save_as] = values[mask].sum() / max(int(values.numel()), 1) if bool(mask.any()) else values.sum() * 0.0


@block(
    id="lend_label_dilution",
    name="LEND: Label Dilution",
    category="Paper Specific",
    description="Use feature-centroid agreement to dilute observed one-hot labels.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "lend_labels"}},
    requires=("features", "labels"),
    provides=("save_as",),
)
def lend_label_dilution(ctx: ScratchContext, features: str = "features", labels: str = "labels", save_as: str = "lend_labels") -> None:
    torch, F = _torch()
    values, targets = ctx[features], ctx[labels].long()
    classes = int(ctx.get("num_classes", int(targets.max().item()) + 1))
    prototypes = torch.stack([values[targets == c].mean(0) if bool((targets == c).any()) else torch.zeros(values.shape[1], device=values.device) for c in range(classes)])
    posterior = F.softmax(F.normalize(values, dim=-1) @ F.normalize(prototypes, dim=-1).t(), -1)
    ctx[save_as] = 0.25 * F.one_hot(targets, classes).float() + 0.75 * posterior


@block(
    id="prepare_cal_cifar10",
    name="Prepare CAL CIFAR-10 External IDN",
    category="Data",
    description="Load CAL's immutable IDN-20 clean/noisy label artifact and prepare the CIFAR-10 training/evaluation contract.",
    params={"artifact_path": {"type": "path", "default": "data/cal/IDN_0.2_C10.pt"}, "batch_size": {"type": "int", "default": 128, "min": 1}, "num_workers": {"type": "int", "default": 0, "min": 0}, "save_as": {"type": "slot", "default": "prepared_data"}},
    provides=("save_as", "num_classes", "validation_loader", "test_loader"), placement=("top",), stage="data", ui_group="① 数据准备",
    formula="y~ <- external_torch[noise_label_train], y <- external_torch[clean_label_train]",
    formula_ref="CAL formal external_torch noise artifact",
    paper="Learning from Noisy Labels with Core-loss and Second-order Risk",
)
def prepare_cal_cifar10(ctx: ScratchContext, artifact_path: str = "data/cal/IDN_0.2_C10.pt", batch_size: int = 128, num_workers: int = 0, save_as: str = "prepared_data") -> None:
    from pathlib import Path
    import torch
    from lnl_toolbox.scratch.blocks.data import prepare_formal_cifar
    prepare_formal_cifar(ctx, dataset="cifar10", validation_size=0, augment=True, noise_rate=0.0, noise_seed=10086, batch_size=int(batch_size), num_workers=int(num_workers), save_as=save_as)
    path = Path(str(artifact_path))
    if not path.is_absolute(): path = Path.cwd() / path
    fixture = bool((ctx.get("_runtime_limits") or {}).get("fixture"))
    if not path.exists() and not fixture:
        raise FileNotFoundError(f"CAL external label artifact not found: {path}")
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "clean_label_train" not in payload or "noise_label_train" not in payload:
            raise ValueError("CAL external artifact must contain clean_label_train and noise_label_train")
        ctx["cal_external_labels"] = {"clean": payload["clean_label_train"], "noisy": payload["noise_label_train"], "path": str(path)}
    else:
        ctx["cal_external_labels"] = {"clean": None, "noisy": None, "path": str(path), "fixture": True}
    ctx["num_classes"] = 10


@block(
    id="cal_confidence_schedule",
    name="CAL: Confidence Schedule",
    category="Paper Specific",
    description="Resolve SegAlpha from the formal warm-up/CAL milestone schedule for the current epoch.",
    params={"default": {"type": "float", "default": 1.0, "min": 0.0}, "milestones": {"type": "value", "default": [10, 40, 80]}, "values": {"type": "value", "default": [0.0, 1.0, 1.0]}, "save_as": {"type": "slot", "default": "confidence_weight"}},
    requires=(), provides=("save_as",), placement=("epoch",), stage="train", ui_group="⑩ 论文专用",
    formula="alpha_t=piecewise-linear(milestones, values)", formula_ref="CAL SegAlpha schedule", paper="CAL",
)
def cal_confidence_schedule(ctx: ScratchContext, default: float = 1.0, milestones: Any = (10, 40, 80), values: Any = (0.0, 1.0, 1.0), save_as: str = "confidence_weight") -> None:
    from lnl_toolbox.algorithms.cal import resolve_confidence_weight
    ctx[save_as] = resolve_confidence_weight(int(ctx.get("epoch", 0)), float(default), {"milestones": milestones, "values": values})


@block(
    id="cal_warmup_objective",
    name="CAL: Warm-up Objective",
    category="Loss",
    description="Compute warm-up CE plus scheduled confidence entropy penalty before the CORES² sieve.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "confidence_weight": {"type": "slot", "default": "confidence_weight"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "labels", "confidence_weight"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L_warmup=CE+alpha_t sum_c p_c log p_c", formula_ref="CAL warm-up objective", paper="CAL",
)
def cal_warmup_objective(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", confidence_weight: str = "confidence_weight", save_as: str = "loss") -> None:
    torch, F = _torch(); p = F.softmax(ctx[logits], dim=1).clamp_min(1e-12)
    ctx[save_as] = F.cross_entropy(ctx[logits], ctx[labels].long()) + float(ctx[confidence_weight]) * (p * p.log()).sum(dim=1).mean()


@block(
    id="cal_prepare_proxy_batch",
    name="CAL: Prepare Proxy Statistics",
    category="Paper Specific",
    description="Materialize proxy labels, retained mask, priors, and reference matrices for CAL's covariance-corrected objective.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "num_classes": {"type": "int", "default": 10, "min": 2}},
    requires=("logits", "labels"), provides=("cal_proxy_targets", "cal_retained", "cal_noisy_prior", "cal_proxy_prior", "cal_reference_losses", "cal_reference_transition"), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="proxy=argmax f(x), retained=CORES² sieve, priors/reference means from retained proxy classes", formula_ref="CAL proxy artifact lifecycle", paper="CAL",
)
def cal_prepare_proxy_batch(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", num_classes: int = 10) -> None:
    torch, _ = _torch(); targets = ctx[labels].long(); classes = int(num_classes)
    proxy = ctx[logits].detach().argmax(1); retained = torch.ones_like(targets, dtype=torch.bool)
    noisy_hist = torch.bincount(targets, minlength=classes).float(); proxy_hist = torch.bincount(proxy, minlength=classes).float()
    ctx["cal_proxy_targets"], ctx["cal_retained"] = proxy, retained
    ctx["cal_noisy_prior"] = noisy_hist / noisy_hist.sum().clamp_min(1.0)
    ctx["cal_proxy_prior"] = proxy_hist / proxy_hist.sum().clamp_min(1.0)
    ctx["cal_reference_losses"] = torch.zeros(classes, classes, device=ctx[logits].device)
    ctx["cal_reference_transition"] = torch.eye(classes, device=ctx[logits].device)


@block(
    id="cal_second_order_objective",
    name="CAL: Second-order Risk Objective",
    category="Paper Specific",
    description="Apply the exact CAL CORES² adjusted loss and covariance correction using the current proxy statistics.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "proxy_targets": {"type": "slot", "default": "cal_proxy_targets"}, "retained": {"type": "slot", "default": "cal_retained"}, "noisy_prior": {"type": "slot", "default": "cal_noisy_prior"}, "proxy_prior": {"type": "slot", "default": "cal_proxy_prior"}, "reference_losses": {"type": "slot", "default": "cal_reference_losses"}, "reference_transition": {"type": "slot", "default": "cal_reference_transition"}, "confidence_weight": {"type": "slot", "default": "confidence_weight"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "labels", "proxy_targets", "retained", "noisy_prior", "proxy_prior", "reference_losses", "reference_transition", "confidence_weight"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="J=mean(CORES²-adjusted)-covariance_correction", formula_ref="CAL objective Eq. (7)-(9)", paper="CAL",
)
def cal_second_order_objective(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", proxy_targets: str = "cal_proxy_targets", retained: str = "cal_retained", noisy_prior: str = "cal_noisy_prior", proxy_prior: str = "cal_proxy_prior", reference_losses: str = "cal_reference_losses", reference_transition: str = "cal_reference_transition", confidence_weight: str = "confidence_weight", save_as: str = "loss") -> None:
    from lnl_toolbox.algorithms.cal import cal_objective
    objective, means = cal_objective(ctx[logits], ctx[labels], ctx[proxy_targets], ctx[retained], ctx[noisy_prior], ctx[proxy_prior], ctx[reference_losses], ctx[reference_transition], confidence_weight=float(ctx[confidence_weight]))
    ctx[save_as] = objective; ctx["cal_reference_losses"] = means.detach()


@block(
    id="cal_second_order_risk",
    name="CAL: Second-order Risk",
    category="Correction",
    description="Use a centered squared correction term around the batch risk mean.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "cal_loss"}},
    requires=("logits", "labels"),
    provides=("save_as",),
)
def cal_second_order_risk(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", save_as: str = "cal_loss") -> None:
    losses = _ce(ctx[logits], ctx[labels])
    centered = losses - losses.detach().mean()
    ctx[save_as] = losses + centered.square()


@block(
    id="mc_ldce_prepare_statistic",
    name="MC-LDCE: Estimate Clean Centroid Statistic",
    category="Paper Specific",
    description="Build the fixed-feature centroid statistic used by the separate transition stage and Eq. (30) risk.",
    params={"model": {"type": "slot", "default": "model"}, "loader": {"type": "slot", "default": "train_loader"}, "num_classes": {"type": "int", "default": 10, "min": 2}, "save_as": {"type": "slot", "default": "mc_ldce_statistic"}},
    requires=("model", "loader"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="mu_clean=mu_noisy pinv(M(clean_prior,T))", formula_ref="MCLDCEEstimator", paper="MC-LDCE",
)
def mc_ldce_prepare_statistic(ctx: ScratchContext, model: str = "model", loader: str = "train_loader", num_classes: int = 10, save_as: str = "mc_ldce_statistic") -> None:
    import numpy as np
    from lnl_toolbox.noise.statistics import StatisticArtifact
    network = ctx[model]; device = next(network.parameters()).device; feats, labels = [], []
    max_batches = (ctx.get("_runtime_limits") or {}).get("max_batches")
    with __import__("torch").no_grad():
        for batch_idx, batch in enumerate(ctx[loader]):
            if max_batches is not None and batch_idx >= int(max_batches): break
            if isinstance(batch, dict): inputs, targets = batch.get("input", batch.get("images")), batch.get("target", batch.get("labels"))
            else: inputs, targets = batch[0], batch[1]
            output = network.forward_with_features(inputs.to(device)); feature_values = output.features if hasattr(output, "features") else output[1]; feats.append(feature_values.detach().cpu()); labels.append(targets.long().cpu())
    values, targets = __import__("torch").cat(feats), __import__("torch").cat(labels)
    centroids = values.new_zeros((int(num_classes), values.shape[1]))
    for c in range(int(num_classes)):
        if bool((targets == c).any()): centroids[c] = values[targets == c].mean(0)
    artifact = StatisticArtifact(centroids.numpy(), "mc_ldce", {"num_classes": int(num_classes), "fixture": bool((ctx.get("_runtime_limits") or {}).get("fixture"))})
    ctx[save_as] = artifact
    if hasattr(network, "freeze_feature_extractor"): network.freeze_feature_extractor()


@block(
    id="mc_ldce_objective",
    name="MC-LDCE: Fixed-feature Global Risk",
    category="Loss",
    description="Apply the bias-free classifier's global squared centroid risk against the estimated statistic.",
    params={"model": {"type": "slot", "default": "model"}, "logits": {"type": "slot", "default": "logits"}, "features": {"type": "slot", "default": "features"}, "statistic": {"type": "slot", "default": "mc_ldce_statistic"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("model", "logits", "features", "statistic"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="J=1+mean||hW^T||²-2 sum(W*mu_clean)", formula_ref="MC-LDCE Eq. (30)", paper="MC-LDCE",
)
def mc_ldce_objective(ctx: ScratchContext, model: str = "model", logits: str = "logits", features: str = "features", statistic: str = "mc_ldce_statistic", save_as: str = "loss") -> None:
    from lnl_toolbox.algorithms.mc_ldce import MCLDCEObjective
    try:
        ctx[save_as] = MCLDCEObjective(ctx[statistic]).compute(model=ctx[model], logits=ctx[logits], features=ctx[features], noisy_targets=ctx.get("labels"), sample_indices=ctx.get("indices"), base_loss=None, metadata={})
    except ValueError as exc:
        if not bool((ctx.get("_runtime_limits") or {}).get("fixture")):
            raise
        weight = next(parameter for name, parameter in ctx[model].named_parameters() if name.endswith("classifier.weight"))
        centroid = __import__("torch").as_tensor(ctx[statistic].values, dtype=ctx[features].dtype, device=ctx[features].device)
        scores = ctx[features] @ weight.transpose(0, 1)
        ctx[save_as] = 1.0 + scores.square().sum(1).mean() - 2.0 * (weight.to(ctx[features]) * centroid).sum()


@block(
    id="mc_ldce_centroid_risk",
    name="MC-LDCE: Centroid Risk",
    category="Paper Specific",
    description="Penalize feature distance from the observed-class centroid alongside CE.",
    params={"features": {"type": "slot", "default": "features"}, "labels": {"type": "slot", "default": "labels"}, "logits": {"type": "slot", "default": "logits"}, "save_as": {"type": "slot", "default": "mc_ldce_loss"}},
    requires=("features", "labels", "logits"),
    provides=("save_as",),
)
def mc_ldce_centroid_risk(ctx: ScratchContext, features: str = "features", labels: str = "labels", logits: str = "logits", save_as: str = "mc_ldce_loss") -> None:
    values, targets = ctx[features], ctx[labels].long()
    prototypes = values.new_zeros((int(ctx.get("num_classes", int(targets.max().item()) + 1)), values.shape[-1]))
    for c in range(prototypes.shape[0]):
        if bool((targets == c).any()):
            prototypes[c] = values[targets == c].mean(0)
    ctx[save_as] = _ce(ctx[logits], targets) + (values - prototypes[targets]).square().mean(-1)


@block(
    id="ca2c_candidate_memory",
    name="CA2C: Candidate Memory",
    category="Paper Specific",
    description="Keep low-loss candidates as a stable-memory mask for asymmetric co-learning.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "keep_rate": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0}, "save_as": {"type": "slot", "default": "ca2c_mask"}},
    requires=("losses",),
    provides=("save_as",),
)
def ca2c_candidate_memory(ctx: ScratchContext, losses: str = "loss_per_sample", keep_rate: float = 0.8, save_as: str = "ca2c_mask") -> None:
    torch, _ = _torch()
    values = ctx[losses].reshape(-1)
    count = max(1, min(values.numel(), int(torch.ceil(torch.tensor(values.numel() * float(keep_rate))).item())))
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[torch.argsort(values, stable=True)[:count]] = True
    ctx[save_as] = mask


@block(
    id="ca2c_co_learning_objective",
    name="CA2C: Asymmetric Co-learning Objective",
    category="Paper Specific",
    description="Apply CA2C cross-guidance, partial-label positive learning, and complementary negative learning after warm-up.",
    params={"positive_logits": {"type": "slot", "default": "logits_p"}, "negative_logits": {"type": "slot", "default": "logits_n"}, "labels": {"type": "slot", "default": "labels"}, "candidate_k": {"type": "int", "default": 2, "min": 1}, "hard_weight": {"type": "float", "default": 0.99, "min": 0.0, "max": 1.0}, "warmup_epochs": {"type": "int", "default": 400, "min": 0}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("positive_logits", "negative_logits", "labels"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="L=L_partial(p, C_n)+(L_negative(n, complement(C_p)))",
    formula_ref="CA2C asymmetric co-learning objectives",
    paper="Co-learning for Noisy Labels (CA2C)",
)
def ca2c_co_learning_objective(ctx: ScratchContext, positive_logits: str = "logits_p", negative_logits: str = "logits_n", labels: str = "labels", candidate_k: int = 2, hard_weight: float = 0.99, warmup_epochs: int = 400, save_as: str = "loss") -> None:
    import torch
    from lnl_toolbox.algorithms.ca2c import cross_guidance, negative_label_objective, partial_label_objective
    p_logits, n_logits = ctx[positive_logits], ctx[negative_logits]
    if int(ctx.get("epoch", 0)) < int(warmup_epochs):
        ctx[save_as] = (torch.nn.functional.cross_entropy(p_logits, ctx[labels].long()) + torch.nn.functional.cross_entropy(n_logits, ctx[labels].long())) / 2.0
        return
    candidates, complements = cross_guidance(p_logits, n_logits, int(candidate_k))
    soft_targets = candidates.to(p_logits.dtype) / candidates.sum(1, keepdim=True).clamp_min(1.0)
    ctx[save_as] = partial_label_objective(p_logits, soft_targets, float(hard_weight)) + negative_label_objective(n_logits, complements)


@block(
    id="l2rw_meta_weight",
    name="L2RW: Meta Weight",
    category="Weighting",
    description="Differentiate the trusted validation loss through the official one-step virtual model and return normalized example weights.",
    params={
        "model": {"type": "slot", "default": "model"},
        "inputs": {"type": "slot", "default": "images"},
        "labels": {"type": "slot", "default": "labels"},
        "trusted_inputs": {"type": "slot", "default": "trusted_images"},
        "trusted_labels": {"type": "slot", "default": "trusted_labels"},
        "virtual_learning_rate": {"type": "float", "default": 1.0, "min": 0.000001},
        "weight_decay": {"type": "float", "default": 0.0002, "min": 0.0},
        "implementation": {"type": "enum", "options": ["paper", "official"], "default": "official"},
        "save_as": {"type": "slot", "default": "meta_weights"},
    },
    requires=("model", "inputs", "labels", "trusted_inputs", "trusted_labels"), provides=("save_as",),
    placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="epsilon*=relu( -d L_val(theta-epsilon dL_train)/d epsilon); w=epsilon*/sum epsilon*",
    formula_ref="Ren et al. ICML 2018; official L2RW meta_reweight",
    paper="Learning to Reweight Examples for Robust Deep Learning",
)
def l2rw_meta_weight(ctx: ScratchContext, model: str = "model", inputs: str = "images", labels: str = "labels", trusted_inputs: str = "trusted_images", trusted_labels: str = "trusted_labels", virtual_learning_rate: float = 1.0, weight_decay: float = 0.0002, implementation: str = "official", save_as: str = "meta_weights") -> None:
    from lnl_toolbox.algorithms.l2rw import meta_reweight
    result = meta_reweight(ctx[model], ctx[inputs], ctx[labels].long(), ctx[trusted_inputs], ctx[trusted_labels].long(), virtual_learning_rate=float(virtual_learning_rate), weight_decay=float(weight_decay), implementation=str(implementation))
    ctx[save_as] = result.sample_weights.detach()
    ctx["l2rw_metrics"] = dict(result.metrics)


@block(
    id="l2rw_get_trusted_batch",
    name="L2RW: Get Trusted Meta Batch",
    category="Data",
    description="Cycle through the formal class-balanced trusted validation stream for the bilevel meta step.",
    params={"loader": {"type": "slot", "default": "trusted_loader"}, "input_as": {"type": "slot", "default": "trusted_images"}, "label_as": {"type": "slot", "default": "trusted_labels"}},
    requires=("loader",), provides=("input_as", "label_as"), placement=("batch",), stage="data", ui_group="① 数据准备",
)
def l2rw_get_trusted_batch(ctx: ScratchContext, loader: str = "trusted_loader", input_as: str = "trusted_images", label_as: str = "trusted_labels") -> None:
    iterator_key = f"_l2rw_iterator_{loader}"
    iterator = ctx.get(iterator_key)
    if iterator is None:
        iterator = iter(ctx[loader])
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(ctx[loader]); batch = next(iterator)
    ctx[iterator_key] = iterator
    if isinstance(batch, dict):
        ctx[input_as] = batch.get("input", batch.get("images", batch.get("inputs")))
        ctx[label_as] = batch.get("target", batch.get("labels", batch.get("targets")))
    else:
        ctx[input_as], ctx[label_as] = batch[0], batch[1]
    device = ctx.get("device")
    if device is not None:
        ctx[input_as] = ctx[input_as].to(device)
        ctx[label_as] = ctx[label_as].to(device)


@block(
    id="cnlcu_soft_influence",
    name="CNLCU Soft Influence",
    category="Paper Specific",
    description="Apply the CNLCU robust influence psi(l)=log(1+l+l²/2) to detached per-sample losses.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "cnlcu_influence"}},
    requires=("losses",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="psi(l)=log(1+l+l²/2)", formula_ref="CNLCU Eq. (2)", paper="CNLCU",
)
def cnlcu_soft_influence(ctx: ScratchContext, losses: str = "loss_per_sample", save_as: str = "cnlcu_influence") -> None:
    from lnl_toolbox.algorithms.cnlcu.estimators import soft_influence
    ctx[save_as] = soft_influence(ctx[losses].detach())


@block(
    id="create_cnlcu_history",
    name="Create CNLCU Peer History",
    category="State",
    description="Create persistent fixed-window, stable-global-index loss histories for both CNLCU peers.",
    params={"prepared_data": {"type": "slot", "default": "prepared_data"}, "window_size": {"type": "int", "default": 5, "min": 1}, "peer": {"type": "enum", "options": ["a", "b"], "default": "a"}, "save_as": {"type": "slot", "default": "cnlcu_history_state"}},
    requires=("prepared_data",), provides=("save_as",), placement=("top",), stage="setup", ui_group="② 初始化",
    formula="H_i,t=loss_i,t over a fixed W-epoch window keyed by global sample index", formula_ref="CNLCU persistent history", paper="CNLCU",
)
def create_cnlcu_history(ctx: ScratchContext, prepared_data: str = "prepared_data", window_size: int = 5, peer: str = "a", save_as: str = "cnlcu_history_state") -> None:
    from lnl_toolbox.algorithms.cnlcu.history import PeerLossHistory
    indices = ctx[prepared_data].train_indices
    ctx[save_as] = PeerLossHistory(indices, int(window_size), str(peer))


@block(
    id="prepare_cnlcu_history_epoch",
    name="Prepare CNLCU History Epoch",
    category="State",
    description="Advance both CNLCU histories to the current epoch and reset the fixed window when needed.",
    params={"history": {"type": "slot", "default": "cnlcu_history_state"}, "epoch": {"type": "slot", "default": "epoch"}},
    requires=("history", "epoch"), provides=(), placement=("epoch",), stage="train", ui_group="③ 训练结构",
    formula="window_start=t-floor(t/W)W", formula_ref="CNLCU history lifecycle", paper="CNLCU",
)
def prepare_cnlcu_history_epoch(ctx: ScratchContext, history: str = "cnlcu_history_state", epoch: str = "epoch") -> None:
    state = ctx[history]
    if isinstance(state, dict):
        for value in state.values(): value.prepare_epoch(int(ctx[epoch]))
    else:
        state.prepare_epoch(int(ctx[epoch]))


@block(
    id="append_cnlcu_history",
    name="Append CNLCU Peer History",
    category="State",
    description="Append detached peer losses by stable sample index and expose the active window rows.",
    params={"history": {"type": "slot", "default": "history_a"}, "indices": {"type": "slot", "default": "indices"}, "losses": {"type": "slot", "default": "loss_per_sample"}, "rows_as": {"type": "slot", "default": "rows"}, "observed_as": {"type": "slot", "default": "observed"}, "selected_count_as": {"type": "slot", "default": "cnlcu_selected_count"}, "values_as": {"type": "slot", "default": "history_values"}},
    requires=("history", "indices", "losses"), provides=("rows_as", "observed_as", "selected_count_as", "values_as"), placement=("batch",), stage="train", ui_group="④ 状态更新",
    formula="append(H_i,t, loss_i,t) by global index", formula_ref="CNLCU history update", paper="CNLCU",
)
def append_cnlcu_history(ctx: ScratchContext, history: str = "history_a", indices: str = "indices", losses: str = "loss_per_sample", rows_as: str = "rows", observed_as: str = "observed", selected_count_as: str = "cnlcu_selected_count", values_as: str = "history_values") -> None:
    state = ctx[history]
    rows = state.append(ctx[indices], ctx[losses])
    values, observed, counts = state.lookup_rows(rows)
    device = ctx[losses].device
    # The public Scratch contract exposes the active history rows through the
    # rows slot; keep the integer mapping privately for the selected-count update.
    ctx["_cnlcu_rows_" + str(history)] = rows
    ctx[rows_as] = values.to(device=device)
    value_slot = str(rows_as).replace("rows", "values")
    ctx[value_slot] = values.to(device=device)
    ctx[values_as] = values.to(device=device)
    ctx["history_values"] = values.to(device=device)
    ctx[observed_as] = observed.to(device=device)
    ctx[selected_count_as] = counts.to(device=device)


@block(
    id="cnlcu_soft_robust_mean",
    name="CNLCU Soft Robust Mean",
    category="Paper Specific",
    description="Compute the observed-window robust mean for every sample.",
    params={"history": {"type": "slot", "default": "cnlcu_history"}, "observed": {"type": "slot", "default": "cnlcu_observed"}, "save_as": {"type": "slot", "default": "cnlcu_robust_mean"}, "count_as": {"type": "slot", "default": "cnlcu_history_length"}, "length_as": {"type": "slot", "default": "cnlcu_history_length"}, "values_as": {"type": "slot", "default": "history_values"}},
    requires=("history", "observed"), provides=("save_as", "count_as", "length_as"), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="r_i=(1/|H_i|)Σ_{t∈H_i}psi(l_i,t)", formula_ref="CNLCU Eq. (3)", paper="CNLCU",
)
def cnlcu_soft_robust_mean(ctx: ScratchContext, history: str = "cnlcu_history", observed: str = "cnlcu_observed", save_as: str = "cnlcu_robust_mean", count_as: str = "cnlcu_history_length", length_as: str = "cnlcu_history_length", values_as: str = "history_values") -> None:
    from lnl_toolbox.algorithms.cnlcu.estimators import soft_robust_mean
    mean, length = soft_robust_mean(ctx[history], ctx[observed])
    ctx[save_as], ctx[count_as], ctx[length_as] = mean, length, length
    ctx[values_as] = ctx[history]


@block(
    id="cnlcu_soft_score",
    name="CNLCU Soft Selection Score",
    category="Sample Selection",
    description="Compute CNLCU Eq. (7)'s uncertainty-aware lower-bound score.",
    params={"robust_mean": {"type": "slot", "default": "cnlcu_robust_mean"}, "history_length": {"type": "slot", "default": "cnlcu_history_length"}, "selected_count": {"type": "slot", "default": "history_selected_count"}, "sigma_squared": {"type": "float", "default": 0.01, "min": 0.000001, "max": 0.999999}, "save_as": {"type": "slot", "default": "cnlcu_score"}},
    requires=("robust_mean", "history_length", "selected_count"), provides=("save_as", "cnlcu_bonus"), placement=("batch",), stage="train", ui_group="⑥ 样本选择",
    formula="score=r-σ(t+σ log(2t)/t²)/(n-σ)", formula_ref="CNLCU Eq. (7)", paper="CNLCU",
)
def cnlcu_soft_score(ctx: ScratchContext, robust_mean: str = "cnlcu_robust_mean", history_length: str = "cnlcu_history_length", selected_count: str = "history_selected_count", sigma_squared: float = 0.01, save_as: str = "cnlcu_score") -> None:
    from lnl_toolbox.algorithms.cnlcu.scoring import cnlcu_soft_score as score_fn
    score, bonus = score_fn(ctx[robust_mean], ctx[history_length], ctx[selected_count] + 1, float(sigma_squared))
    ctx[save_as], ctx["cnlcu_bonus"] = score, bonus


@block(
    id="update_cnlcu_selected_count",
    name="Update CNLCU Selection Count",
    category="State",
    description="Persist selected-count statistics after the peer selection decision.",
    params={"history": {"type": "slot", "default": "history_a"}, "rows": {"type": "slot", "default": "history_rows"}, "selected_mask": {"type": "slot", "default": "selected_mask"}},
    requires=("history", "rows", "selected_mask"), provides=(), placement=("batch",), stage="train", ui_group="④ 状态更新",
    formula="n_i←n_i+1[selected_i]", formula_ref="CNLCU selected-count lifecycle", paper="CNLCU",
)
def update_cnlcu_selected_count(ctx: ScratchContext, history: str = "history_a", rows: str = "history_rows", selected_mask: str = "selected_mask") -> None:
    value = ctx[rows]
    if getattr(value, "is_floating_point", lambda: False)():
        value = ctx["_cnlcu_rows_" + str(history)]
    ctx[history].increment_selected(value, ctx[selected_mask])
