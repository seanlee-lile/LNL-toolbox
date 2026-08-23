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
    id="l2rw_meta_weight",
    name="L2RW: Meta Weight",
    category="Weighting",
    description="Produce detached normalized example weights from a virtual meta objective.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "meta_weights"}},
    requires=("losses",),
    provides=("save_as",),
)
def l2rw_meta_weight(ctx: ScratchContext, losses: str = "loss_per_sample", save_as: str = "meta_weights") -> None:
    weights = (-ctx[losses].detach()).softmax(dim=0)
    ctx[save_as] = weights * weights.numel()
