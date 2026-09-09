"""Independent paper-level noisy-label operations.

These blocks intentionally operate on Context slots instead of calling the
legacy algorithm package.  They are small, semantic operations that can be
composed in a recipe and are useful for shape/value smoke checks.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...context import ScratchContext
from ...registry import block


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional train extra
        raise RuntimeError("Paper blocks require PyTorch; install the `train` extra.") from exc
    return torch, F


@block(
    id="pdl_fit_part_representation",
    name="PDL Part Representation",
    category="Transition",
    description="Fit PDL's multiplicative-update nonnegative part representation once on train plus noisy-validation features.",
    params={
        "features": {"type": "slot", "default": "pdl_representation_features"},
        "num_parts": {"type": "int", "required": True, "min": 1},
        "iterations": {"type": "int", "required": True, "min": 1},
        "error_tolerance": {"type": "float", "default": 1.0e-5, "min": 0.0},
        "representation_seed": {"type": "int", "required": True, "min": 0},
        "seed_policy": {"type": "enum", "required": True, "options": ["stochastic", "deterministic"]},
        "parts_as": {"type": "slot", "default": "pdl_parts"},
        "coefficients_as": {"type": "slot", "default": "pdl_coefficients"},
        "indices_as": {"type": "slot", "default": "pdl_representation_indices"},
    },
    requires=("features",),
    provides=("parts_as", "coefficients_as", "indices_as"),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="H,W = argmin_{H,W>=0} ||X-WH||^2; normalize W rows", formula_kind="special",
    formula_ref="PDL official train_m multiplicative updates",
    paper="Part-dependent Label Noise",
)
def pdl_fit_part_representation(
    ctx: ScratchContext,
    features: str = "pdl_representation_features",
    num_parts: int | None = None,
    iterations: int | None = None,
    error_tolerance: float = 1.0e-5,
    representation_seed: int | None = None,
    seed_policy: str | None = None,
    parts_as: str = "pdl_parts",
    coefficients_as: str = "pdl_coefficients",
    indices_as: str = "pdl_representation_indices",
) -> None:
    from ...native_stats import fit_part_representation
    if num_parts is None or iterations is None or representation_seed is None or seed_policy is None:
        raise ValueError("PDL part representation requires num_parts, iterations, representation_seed, and seed_policy")
    snapshot = ctx[features]
    if bool((ctx.get("_runtime_limits") or {}).get("fixture")):
        # The bounded fixture uses a four-dimensional representation; cap the
        # formal part count only for this synthetic run so factorisation is
        # well-defined without changing the formal recipe.
        num_parts = min(int(num_parts), min(int(snapshot.features.shape[0]), int(snapshot.features.shape[1])))
    seed = None if seed_policy == "stochastic" else int(representation_seed)
    parts, coefficients = fit_part_representation(
        snapshot.features, int(num_parts), seed=seed,
        iterations=int(iterations), error_tolerance=float(error_tolerance),
    )
    ctx[parts_as] = parts
    ctx[coefficients_as] = coefficients
    ctx[indices_as] = snapshot.global_indices.copy()


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
        "basis_epochs": {"type": "int", "required": True, "min": 1},
        "basis_learning_rate": {"type": "float", "required": True, "min": 0.0},
        "basis_loss_threshold": {"type": "float", "required": True, "min": 0.0},
        "representation_seed": {"type": "int", "required": True, "min": 0},
        "train_as": {"type": "slot", "default": "pdl_train_basis"},
        "validation_as": {"type": "slot", "default": "pdl_validation_basis"},
    },
    requires=("coefficients", "representation_indices", "train_posterior", "validation_posterior", "train_anchors", "validation_anchors"),
    provides=("train_as", "validation_as"),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="M_c=argmin_M sum_r ||W_{a_c(r)} M - q_{a_c(r)}||^2; normalize |M| rows", formula_kind="special",
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
    basis_epochs: int | None = None,
    basis_learning_rate: float | None = None,
    basis_loss_threshold: float | None = None,
    representation_seed: int | None = None,
    train_as: str = "pdl_train_basis",
    validation_as: str = "pdl_validation_basis",
) -> None:
    import numpy as np
    if basis_epochs is None or basis_learning_rate is None or basis_loss_threshold is None or representation_seed is None:
        raise ValueError("PDL basis fitting requires basis_epochs, basis_learning_rate, basis_loss_threshold, and representation_seed")
    from ...native_stats import fit_pdl_basis_matrices_pair
    coeff = ctx[coefficients]
    rep_indices = np.asarray(ctx[representation_indices], dtype=np.int64)
    train = ctx[train_posterior]
    validation = ctx[validation_posterior]
    train_global = np.asarray(train.global_indices, dtype=np.int64)
    validation_global = np.asarray(validation.global_indices, dtype=np.int64)
    train_positions = np.searchsorted(rep_indices, train_global)
    validation_positions = np.searchsorted(rep_indices, validation_global)
    if np.any(train_positions >= rep_indices.size) or np.any(validation_positions >= rep_indices.size):
        raise KeyError("PDL representation does not cover train/validation snapshots")
    # The common anchor-selection block publishes stable sample identities,
    # not local array positions.  Resolve those identities explicitly at the
    # paper primitive boundary before selecting rows from each snapshot.
    train_anchor_ids = np.asarray(ctx[train_anchors], dtype=np.int64)
    validation_anchor_ids = np.asarray(ctx[validation_anchors], dtype=np.int64)
    # Bounded fixtures cap the NMF part count to the tiny feature width.  Use
    # the same number of anchor candidates for that fixture-only path; the
    # formal recipe supplies exactly ``num_parts`` candidates per class.
    if bool((ctx.get("_runtime_limits") or {}).get("fixture")):
        fixture_parts = int(coeff.shape[1])
        train_anchor_ids = train_anchor_ids[:, :fixture_parts]
        validation_anchor_ids = validation_anchor_ids[:, :fixture_parts]
    train_anchor_positions = np.searchsorted(train_global, train_anchor_ids)
    validation_anchor_positions = np.searchsorted(validation_global, validation_anchor_ids)
    if np.any(train_anchor_positions >= train_global.size) or not np.array_equal(train_global[train_anchor_positions], train_anchor_ids):
        raise KeyError("PDL train anchors are not aligned to train snapshot indices")
    if np.any(validation_anchor_positions >= validation_global.size) or not np.array_equal(validation_global[validation_anchor_positions], validation_anchor_ids):
        raise KeyError("PDL validation anchors are not aligned to validation snapshot indices")
    limits = ctx.get("_runtime_limits") or {}
    effective_epochs = int(basis_epochs)
    if limits:
        effective_epochs = min(effective_epochs, max(1, int(limits.get("max_epochs", 1))))
    train_basis, validation_basis = fit_pdl_basis_matrices_pair(
        coeff[train_positions][train_anchor_positions], train.noisy_probabilities[train_anchor_positions],
        coeff[validation_positions][validation_anchor_positions], validation.noisy_probabilities[validation_anchor_positions],
        epochs=effective_epochs, learning_rate=float(basis_learning_rate),
        loss_threshold=float(basis_loss_threshold), seed=int(representation_seed),
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
        "num_parts": {"type": "int", "required": True, "min": 1},
        "representation_seed": {"type": "int", "required": True, "min": 0},
        "train_as": {"type": "slot", "default": "pdl_transition"},
        "validation_as": {"type": "slot", "default": "pdl_validation_transition"},
        "revision_validation_as": {"type": "slot", "default": "pdl_revision_validation_transition"},
    },
    requires=("parts", "coefficients", "representation_indices", "train_features", "train_posterior", "validation_features", "validation_posterior", "train_basis", "validation_basis"),
    provides=("train_as", "validation_as", "revision_validation_as"),
    placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="T(x)=sum_r beta_r(x) M_r", formula_kind="special",
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
    num_parts: int | None = None,
    representation_seed: int | None = None,
    train_as: str = "pdl_transition",
    validation_as: str = "pdl_validation_transition",
    revision_validation_as: str = "pdl_revision_validation_transition",
) -> None:
    from ...native_stats import PartTransitionEstimator
    if num_parts is None or representation_seed is None:
        raise ValueError("PDL transition estimation requires num_parts and representation_seed")
    classes = int(ctx[train_posterior].num_classes)
    # Bounded fixtures may cap the representation rank; the formal recipe's
    # ``num_parts`` and the realized coefficient width are identical outside
    # that fixture-only path.
    effective_parts = int(getattr(ctx[parts], "shape", (0, 0))[1])
    estimator = PartTransitionEstimator(effective_parts, classes, representation_seed=int(representation_seed))
    train_artifact = estimator.estimate_from_shared_representation(
        ctx[train_features], ctx[train_posterior],
        representation_parts=ctx[parts], representation_coefficients=ctx[coefficients],
        representation_indices=ctx[representation_indices], part_matrices=ctx[train_basis],
    )
    validation_artifact = estimator.estimate_from_shared_representation(
        ctx[validation_features], ctx[validation_posterior],
        representation_parts=ctx[parts], representation_coefficients=ctx[coefficients],
        representation_indices=ctx[representation_indices], part_matrices=ctx[validation_basis],
    )
    ctx[train_as] = train_artifact
    ctx[validation_as] = validation_artifact
    ctx[revision_validation_as] = validation_artifact.with_part_matrices(
        train_artifact.part_matrices, role="revision_validation",
        source_artifact_hash=train_artifact.artifact_hash,
    )


@block(
    id="create_mentor_provider",
    name="MentorNet: Create Weight Provider",
    category="Paper Specific",
    description="Load the frozen MentorArtifact and configure the moving-percentile, burn-in, label, and dropout lifecycle from the formal recipe.",
    params={
        "artifact_path": {"type": "str", "required": True},
        "total_epochs": {"type": "int", "required": True, "min": 1},
        "percentile": {"type": "float", "required": True, "min": 0.0001, "max": 0.9999},
        "decay": {"type": "float", "required": True, "min": 0.0, "max": 0.9999},
        "burn_in_epoch": {"type": "int", "required": True, "min": 0},
        "fixed_epoch_after_burn_in": {"type": "bool", "required": True},
        "fixed_label": {"type": "int", "required": True, "min": 0},
        "dropout_schedule": {"type": "value", "required": True},
        "seed": {"type": "int", "required": True, "min": 0},
        "save_as": {"type": "slot", "default": "mentor_provider"},
        "threshold_as": {"type": "slot", "default": "mentor_threshold"},
    },
    provides=("save_as", "threshold_as"), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="q_t=EMA_p(loss), w_i=M(loss_i,loss_i-q_t,y_i,e_t) with burn-in/dropout lifecycle", formula_kind="special",
    formula_ref="Jiang et al., MentorNet, ICML 2018; formal pipeline.weight_provider",
    paper="MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Noisy Labels",
)
def create_mentor_provider(
    ctx: ScratchContext,
    artifact_path: str | None = None,
    total_epochs: int | None = None,
    percentile: float | None = None,
    decay: float | None = None,
    burn_in_epoch: int | None = None,
    fixed_epoch_after_burn_in: bool | None = None,
    fixed_label: int | None = None,
    dropout_schedule: Any = None,
    seed: int | None = None,
    save_as: str = "mentor_provider",
    threshold_as: str = "mentor_threshold",
) -> None:
    import os
    from pathlib import Path
    from ...native_stats import MentorNetWeightProvider

    if not artifact_path or total_epochs is None or percentile is None or decay is None or burn_in_epoch is None or fixed_epoch_after_burn_in is None or fixed_label is None or dropout_schedule is None or seed is None:
        raise ValueError("MentorNet provider requires artifact_path and all curriculum parameters")
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
                from ...native_stats import WeightResult
                return WeightResult(weights.clamp(0, 1).detach(), {"weight_mean": float(weights.mean().item()), "moving_percentile": float(self.moving)})
        provider = _FixtureMentorProvider()
    else:
        raise FileNotFoundError(f"MentorArtifact not found: {path}")
    ctx[save_as] = provider
    ctx[threshold_as] = None
    ctx["mentor_artifact_path"] = os.fspath(path)


@block(id="mentor_build_features", name="MentorNet: Build Mentor Features", category="Weighting", description="Expose loss, loss deviation, label and curriculum epoch features consumed by the frozen mentor.", params={"provider":{"type":"slot","default":"mentor_provider"},"losses":{"type":"slot","default":"loss_per_sample"},"labels":{"type":"slot","default":"labels"},"threshold":{"type":"slot","default":"mentor_threshold"},"save_as":{"type":"slot","default":"mentor_features"}}, requires=("provider","losses","labels","threshold"), provides=("save_as",), placement=("batch",), formula="v_i=(loss_i,loss_i-q_t,y_i,e_t)", formula_kind="special", formula_ref="MentorNet feature construction", paper="MentorNet")
def mentor_build_features(ctx: ScratchContext, provider: str="mentor_provider", losses: str="loss_per_sample", labels: str="labels", threshold: str="mentor_threshold", save_as: str="mentor_features") -> None:
    torch,_=_torch(); holder=ctx[provider]; epoch=int(ctx.get("epoch",0)); mentor_epoch=min(epoch,int(holder.burn_in_epoch)) if holder.burn_in_epoch is not None and holder.fixed_epoch_after_burn_in else epoch
    mentor_labels=torch.full_like(ctx[labels].long(),int(holder.fixed_label)) if holder.fixed_label is not None else ctx[labels].long()
    ctx[save_as]={"losses":ctx[losses].detach(),"deviation":ctx[losses].detach()-float(ctx[threshold]),"labels":mentor_labels,"epochs":torch.full_like(ctx[losses].detach(),float(mentor_epoch)),"mentor_epoch":mentor_epoch}


@block(id="mentor_predict_sample_weights", name="MentorNet: Predict Sample Weights", category="Weighting", description="Apply burn-in, frozen mentor prediction, and configured dropout to exposed mentor features.", params={"provider":{"type":"slot","default":"mentor_provider"},"features":{"type":"slot","default":"mentor_features"},"save_as":{"type":"slot","default":"sample_weights"}}, requires=("provider","features"), provides=("save_as",), placement=("batch",), formula="w_i=M(v_i) with burn-in and dropout", formula_kind="special", formula_ref="MentorNet curriculum prediction", paper="MentorNet")
def mentor_predict_sample_weights(ctx: ScratchContext, provider: str="mentor_provider", features: str="mentor_features", save_as: str="sample_weights") -> None:
    torch,_=_torch(); holder=ctx[provider]; values=ctx[features]; mentor_epoch=int(values["mentor_epoch"]); burn=holder.burn_in_epoch is not None and mentor_epoch < max(0,int(holder.burn_in_epoch)-1)
    if burn: weights=torch.ones_like(values["losses"])
    else:
        model = getattr(holder, "model", None)
        if model is None:
            weights = torch.sigmoid(-values["deviation"])
        else:
            model.to(values["losses"].device)
            with torch.no_grad(): weights=model(values["losses"],values["deviation"],values["labels"],values["epochs"])
    rate=float(holder._dropout_rate(mentor_epoch)) if hasattr(holder, "_dropout_rate") else 0.0
    if rate: weights=weights*(torch.rand(weights.shape,generator=holder.generator,device="cpu").to(weights.device)>=rate)
    if not bool((weights>0).any()): weights=torch.ones_like(weights)
    ctx[save_as]=weights.clamp(0,1).detach()


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
    formula="T = T_club T_spade; T_spade[a,b]=count(argmax q=a, y_tilde=b)/count(argmax q=a)", formula_kind="special",
    formula_ref="Yao et al., Dual T NeurIPS 2020, Algorithm 1",
    paper="Dual T: Reducing Estimation Error for Transition Matrix in Label-noise Learning",
)
def dual_t_transition_estimation(
    ctx: ScratchContext,
    posterior: str = "posterior_snapshot",
    save_as: str = "transition",
) -> None:
    import torch
    from ...native_stats import DualTransitionEstimator

    runtime_limits = ctx.get("_runtime_limits")
    artifact = DualTransitionEstimator().estimate(
        ctx[posterior],
        allow_empty_intermediate=bool(runtime_limits),
    )
    ctx[save_as] = torch.tensor(artifact.matrix, dtype=torch.float32)
    ctx["dual_t_transition_artifact"] = artifact


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
    formula="w(x, y_tilde) = (q_y(x) - rho_{1-y}) / ((1-rho_0-rho_1) q_y(x))", formula_kind="composite",
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


def _cwd_swap_matrix(classes: int, source: int, target: int, *, device, dtype):
    torch, _ = _torch()
    value = torch.eye(classes, device=device, dtype=dtype)
    value[[source, target]] = value[[target, source]]
    return value


@block(
    id="cwd_observed_statistics",
    name="CWD: Observed Class Statistics",
    category="Paper Specific",
    description="Compute observed class prior and class feature means from the noisy snapshot.",
    params={"snapshot": {"type": "slot", "default": "cwd_snapshot"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "cwd_observed"}},
    requires=("snapshot", "transition"), provides=("save_as", "observed_prior", "observed_centroids"),
    placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="p~_c = n_c/N; m~_c = (1/N) sum_i h_i 1[y~_i=c]", formula_ref="CWD observed statistics", formula_kind="special", paper="Class-Wise Denoising",
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
    id="cwd_virtual_systems",
    name="CWD: Build Virtual Systems",
    category="Paper Specific",
    description="Construct each virtual prior, virtual flip matrix, and coefficient matrix.",
    params={"clean_prior": {"type": "slot", "default": "clean_prior"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "cwd_systems"}},
    requires=("clean_prior", "transition"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="C_k = sum_{s,t} p^k_s T^k_{s,t} P_{s,t}^T", formula_ref="CWD Eqs. (21)-(29)", formula_kind="special", paper="Class-Wise Denoising",
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
    formula="C_k^+ = pinv(C_k)", formula_ref="CWD Eq. (30)", formula_kind="special", paper="Class-Wise Denoising",
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
    formula="M = sum_k M~ C_k^+ - (C-1)M~", formula_ref="CWD centroid recovery", formula_kind="special", paper="Class-Wise Denoising",
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
    formula="L = 1 + mean(m^2) - 2 w^T(mu_1-mu_0) - 2b(p_1-p_0)", formula_ref="CWD global squared objective", formula_kind="composite", paper="Class-Wise Denoising",
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
    id="pcse_recover_layer_statistics",
    name="PCSE: Recover Layer Statistics",
    category="Paper Specific",
    description="Recover clean per-class means, second moments, and covariances for aligned feature snapshots.",
    params={"snapshots": {"type": "slot", "default": "pcse_snapshots"}, "layer_names": {"type": "value", "required": True}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "pcse_statistics"}},
    requires=("snapshots", "transition"), provides=("save_as",), placement=("top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="mu=R^T mu~; S=R^T S~; Sigma=S-mu mu^T", formula_ref="PCSE Eqs. (20)-(23)", formula_kind="special", paper="Estimating Per-Class Statistics",
)
def pcse_recover_layer_statistics(ctx: ScratchContext, snapshots: str = "pcse_snapshots", layer_names: list[str] | tuple[str, ...] | None = None, transition: str = "transition", save_as: str = "pcse_statistics") -> None:
    import numpy as np
    from ...native_stats import estimate_pcse_statistics
    if not layer_names:
        snapshot_value = ctx[snapshots]
        if isinstance(snapshot_value, Mapping):
            layer_names = tuple(str(name) for name in snapshot_value)
        else:
            layer_names = tuple(str(name) for name in getattr(snapshot_value, "layer_names", ()))
        if not layer_names:
            raise ValueError("PCSE layer_names must be supplied by the Recipe or snapshot metadata")
    matrix = ctx[transition]
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    elif hasattr(matrix, "matrix"):
        matrix = matrix.matrix()
        if hasattr(matrix, "detach"):
            matrix = matrix.detach().cpu().numpy()
    result = estimate_pcse_statistics(ctx[snapshots], tuple(layer_names), np.asarray(matrix))
    ctx[save_as] = result


@block(
    id="create_fine_state",
    name="FINE: Create EMA/SED State",
    category="Paper Specific",
    description="Create the official EMA teacher, self-adaptive class selector, confidence reweighting, and FINE regularizer state.",
    params={
        "model": {"type": "slot", "default": "model"},
        "ema_model": {"type": "value", "default": None},
        "prepared_data": {"type": "value", "default": None},
        "num_classes": {"type": "int", "min": 2},
        "ema_momentum": {"type": "float", "required": True, "min": 0.0, "max": 0.999999},
        "momentum_scs": {"type": "float", "required": True, "min": 0.0, "max": 0.999999},
        "momentum_scr": {"type": "float", "required": True, "min": 0.0, "max": 0.999999},
        "quantile": {"type": "float", "required": True, "min": 0.0001, "max": 0.9999},
        "maximum_threshold": {"type": "float", "required": True, "min": 0.0, "max": 1.0},
        "beta": {"type": "float", "required": True, "min": 0.0},
        "gamma": {"type": "float", "required": True, "min": 0.0},
        "probability_floor": {"type": "float", "default": 1.0e-7, "min": 1.0e-12},
        "seed": {"type": "int", "required": True, "min": 0},
        "save_as": {"type": "slot", "default": "fine_state"},
    },
    requires=("model",), provides=("save_as", "fine_clean_state", "fine_pseudo_state", "fine_weight_state"), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="EMA_t=m EMA_{t-1}+(1-m)f_t; SCS/SCR operate on epoch snapshots", formula_kind="special",
    formula_ref="FINE official SED warm-up/EMA/SCS/SCR lifecycle",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def create_fine_state(
    ctx: ScratchContext,
    model: str = "model",
    ema_model: Any = None,
    prepared_data: Any = None,
    num_classes: int | None = None,
    ema_momentum: float | None = None,
    momentum_scs: float | None = None,
    momentum_scr: float | None = None,
    quantile: float | None = None,
    maximum_threshold: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    probability_floor: float = 1.0e-7,
    seed: int | None = None,
    save_as: str = "fine_state",
) -> None:
    torch, _ = _torch()
    from ...native_stats import SelfAdaptiveClassSelector, SelfAdaptiveConfidenceReweighting, FINERegularizer, ModelEMA
    prepared = ctx[prepared_data] if isinstance(prepared_data, str) and prepared_data in ctx else prepared_data
    if num_classes is None:
        num_classes = getattr(prepared, "num_classes", None) if prepared is not None else None
        if num_classes is None:
            num_classes = ctx.get("num_classes")
    if num_classes is None or ema_momentum is None or momentum_scs is None or momentum_scr is None or quantile is None or maximum_threshold is None or beta is None or gamma is None or seed is None:
        raise ValueError("FINE state requires class count and all SCS/SCR/EMA parameters")
    ema = ctx[ema_model] if isinstance(ema_model, str) and ema_model in ctx else ModelEMA(ctx[model], float(ema_momentum), update_buffers=False)
    ctx[save_as] = {
        "ema": ema,
        "scs": SelfAdaptiveClassSelector(int(num_classes), float(momentum_scs), quantile=float(quantile), maximum_threshold=float(maximum_threshold)),
        "scr": SelfAdaptiveConfidenceReweighting(int(num_classes), float(momentum_scr)),
        "regularizer": FINERegularizer(beta=float(beta), gamma=float(gamma), probability_floor=float(probability_floor), seed=int(seed)),
        # Per-sample decisions are published as generic indexed tables by the
        # epoch selection blocks below.  Keeping the table contract here
        # prevents a paper-specific batch lookup object from becoming part of
        # the public State language.
    }
    train_indices = getattr(prepared, "train_indices", None) if prepared is not None else None
    rows = torch.as_tensor(train_indices, dtype=torch.long).reshape(-1).cpu() if train_indices is not None else torch.empty(0, dtype=torch.long)
    size = int(rows.max().item()) + 1 if rows.numel() else 0
    ctx["fine_clean_state"] = {"values": torch.ones(size, dtype=torch.bool), "seen": torch.zeros(size, dtype=torch.bool), "last_epoch": torch.full((size,), -1, dtype=torch.long)}
    ctx["fine_pseudo_state"] = {"values": torch.zeros(size, dtype=torch.long), "seen": torch.zeros(size, dtype=torch.bool), "last_epoch": torch.full((size,), -1, dtype=torch.long)}
    ctx["fine_weight_state"] = {"values": torch.ones(size, dtype=torch.float32), "seen": torch.zeros(size, dtype=torch.bool), "last_epoch": torch.full((size,), -1, dtype=torch.long)}


def _ensure_fine_indexed_table(holder: dict[str, Any], name: str, indices: Any, *, dtype: Any, initial: Any) -> dict[str, Any]:
    """Return a generic ``create_indexed_state``-compatible table.

    FINE's SCS/SCR algorithms compute their values in an epoch snapshot, but
    the training batch consumes them by stable sample index.  This helper only
    allocates the public indexed-table shape; it does not implement selection
    or weighting semantics.
    """
    torch, _ = _torch()
    rows = torch.as_tensor(indices, dtype=torch.long).reshape(-1).cpu()
    size = int(rows.max().item()) + 1 if rows.numel() else 0
    table = holder.get(name)
    if table is None or int(table["values"].shape[0]) < size:
        fill = bool(initial) if dtype == torch.bool else initial
        table = {
            # A scalar indexed value is represented as [N], so the canonical
            # indexed_read operation publishes the batch shape directly.
            "values": torch.full((size,), fill, dtype=dtype),
            "seen": torch.zeros(size, dtype=torch.bool),
            "last_epoch": torch.full((size,), -1, dtype=torch.long),
        }
        holder[name] = table
    return table


@block(
    id="fine_snapshot_predictions",
    name="FINE: Snapshot EMA Predictions",
    category="Paper Specific",
    description="Collect stable-index classifier and EMA probabilities on the non-augmented train-evaluation stream.",
    params={"model": {"type": "slot", "default": "model"}, "state": {"type": "slot", "default": "fine_state"}, "loader": {"type": "slot", "default": "train_eval_loader"}, "save_as": {"type": "slot", "default": "fine_snapshot"}},
    requires=("model", "state", "loader"), provides=("save_as",), placement=("epoch",), stage="train", ui_group="⑥ 后验与权重",
    formula="P_t=f_t(x), P_t^{EMA}=EMA_t(x), aligned by stable sample index", formula_kind="special",
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
    requires=("state", "snapshot"), provides=("fine_clean_state", "fine_pseudo_state"), placement=("epoch",), stage="train", ui_group="⑦ 样本选择",
    formula="clean_i=1[p_tilde_i(y_tilde_i) >= tau_global * local_class_modulation]", formula_kind="special",
    formula_ref="FINE Self-Adaptive Class Selection (SCS)",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def fine_scs_select(ctx: ScratchContext, state: str = "fine_state", snapshot: str = "fine_snapshot") -> None:
    torch, _ = _torch()
    values = ctx[snapshot]
    mask = ctx[state]["scs"].select_epoch(values["ema_probabilities"], values["targets"])
    rows = torch.as_tensor(values["indices"], dtype=torch.long).reshape(-1).cpu()
    clean_table = _ensure_fine_indexed_table(ctx[state], "clean_state", rows, dtype=torch.bool, initial=True)
    pseudo_table = _ensure_fine_indexed_table(ctx[state], "pseudo_state", rows, dtype=torch.long, initial=0)
    clean_table["values"][rows] = torch.as_tensor(mask, dtype=torch.bool).reshape(-1).cpu()
    clean_table["seen"][rows] = True
    clean_table["last_epoch"][rows] = int(ctx.get("epoch", -1))
    pseudo = torch.as_tensor(values["ema_probabilities"]).argmax(1).to(torch.long).reshape(-1).cpu()
    pseudo_table["values"][rows] = pseudo
    pseudo_table["seen"][rows] = True
    pseudo_table["last_epoch"][rows] = int(ctx.get("epoch", -1))
    ctx["fine_clean_state"] = clean_table
    ctx["fine_pseudo_state"] = pseudo_table


@block(
    id="fine_scr_reweight",
    name="FINE: SCR Confidence Weights",
    category="Paper Specific",
    description="Update class-wise confidence statistics and store SCR weights by stable index.",
    params={"state": {"type": "slot", "default": "fine_state"}, "snapshot": {"type": "slot", "default": "fine_snapshot"}},
    requires=("state", "snapshot"), provides=("fine_weight_state",), placement=("epoch",), stage="train", ui_group="⑥ 后验与权重",
    formula="w_i=exp(-(max p_i-mu_hat_c)^2/(2 sigma_hat_c^2/n_sigma^2))", formula_kind="special",
    formula_ref="FINE Self-Adaptive Confidence Reweighting (SCR)",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def fine_scr_reweight(ctx: ScratchContext, state: str = "fine_state", snapshot: str = "fine_snapshot") -> None:
    torch, _ = _torch()
    values = ctx[snapshot]
    weights = ctx[state]["scr"].weights(values["ema_probabilities"])
    rows = torch.as_tensor(values["indices"], dtype=torch.long).reshape(-1).cpu()
    weight_table = _ensure_fine_indexed_table(ctx[state], "weight_state", rows, dtype=torch.float32, initial=1.0)
    weight_table["values"][rows] = torch.as_tensor(weights, dtype=torch.float32).reshape(-1).cpu()
    weight_table["seen"][rows] = True
    weight_table["last_epoch"][rows] = int(ctx.get("epoch", -1))
    ctx["fine_weight_state"] = weight_table


@block(
    id="fine_warmup_loss",
    name="FINE: Warm-up Objective",
    category="Paper Specific",
    description="Compute the paper's warm-up cross entropy before SED robust training.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "labels"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L_warmup=CE(f(x),y~)", formula_kind="composite",
    formula_ref="FINE official warm-up objective",
    paper="FINE: Filtering Noise in the Feature Space for Robust Learning with Noisy Labels",
)
def fine_warmup_loss(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", save_as: str = "loss") -> None:
    _, F = _torch()
    # Algorithm 1, lines 2-7: warm-up is ordinary CE on the observed labels.
    # FINE's MU/NL terms are introduced only in the robust-training stage.
    ctx[save_as] = F.cross_entropy(ctx[logits], ctx[labels].long())


@block(
    id="sed_rejected_regularizer",
    name="SED Rejected-sample Regularizer",
    category="Loss",
    description="Apply FINE's rejected-sample machine-unlearning and complementary-label negative-learning terms.",
    params={"logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "clean": {"type": "slot", "default": "clean"}, "state": {"type": "slot", "default": "sed_state"}, "save_as": {"type": "slot", "default": "sed_regularizer"}},
    requires=("logits", "labels", "clean", "state"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="R=β·(1/C)log p_y~+γ·(−1/C)log(1−p_y~comp), y~comp∼Uniform(Y\\{y~})", formula_kind="composite", formula_ref="FINE Eq. (2)-(5)",
)
def sed_rejected_regularizer(ctx: ScratchContext, logits: str = "logits", labels: str = "labels", clean: str = "clean", state: str = "sed_state", save_as: str = "sed_regularizer") -> None:
    ctx[save_as] = ctx[state]["regularizer"](ctx[logits], ctx[labels], rejected_mask=~ctx[clean].bool())


@block(
    id="initialize_t_revision_transition",
    name="T-Revision Pseudo-Anchor Transition",
    category="Transition",
    description="Initialize T-hat by taking the highest noisy posterior example for each class, with stable-index tie breaking.",
    params={"posterior": {"type": "slot", "default": "t_revision_posterior"}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "transition"}},
    requires=("posterior", "device"), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="T_hat[i,:]=q(x_i) where x_i=argmax_x q_i(x)", formula_kind="special", formula_ref="T-Revision Algorithm 1, Stage 1 transition initialization; Eq. (1)", paper="Are Anchor Points Really Indispensable in Label-Noise Learning?",
)
def initialize_t_revision_transition(ctx: ScratchContext, posterior: str = "t_revision_posterior", device: str = "device", save_as: str = "transition") -> None:
    import torch
    from ...native_stats import AnchorTransitionEstimator

    artifact = AnchorTransitionEstimator().estimate(ctx[posterior])
    ctx[save_as] = torch.tensor(artifact.matrix, dtype=torch.float32, device=ctx[device])
    ctx["t_revision_transition_artifact"] = artifact


@block(
    id="t_revision_importance_ratio",
    name="T-Revision Importance Ratio",
    category="Weighting",
    description="Compute the clean-to-noisy posterior ratio for each observed noisy label without clipping or normalization.",
    params={"probabilities": {"type": "slot", "default": "probabilities"}, "noisy_probabilities": {"type": "slot", "default": "noisy_probabilities"}, "labels": {"type": "slot", "default": "labels"}, "denominator_floor": {"type": "float", "default": 1.0e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "sample_weights"}, "denominators_as": {"type": "slot", "default": "sample_denominators"}},
    requires=("probabilities", "noisy_probabilities", "labels"), provides=("save_as", "denominators_as"), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="w_i=g_ytilde_i(x_i)/(g(x_i)T)_ytilde_i", formula_kind="composite", formula_ref="T-Revision paper Eq. (3)", paper="Are Anchor Points Really Indispensable in Label-Noise Learning?",
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
    id="upm_update_eta",
    name="UPM: Update Confusing Probabilities",
    category="Paper Specific",
    description="Apply Eq. (11) eta ascent and [0,1] projection after posterior estimation.",
    params={"state": {"type": "slot", "default": "upm_eta_state"}, "psi_state": {"type": "slot", "default": "upm_psi_state"}, "indices": {"type": "slot", "default": "indices"}, "posterior": {"type": "slot", "default": "clean_posterior"}, "labels": {"type": "slot", "default": "labels"}, "learning_rate": {"type": "float", "required": True, "min": 0.0}},
    requires=("state", "psi_state", "indices", "posterior", "labels"), provides=(), placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="eta<-Pi_[0,1](eta+lr d log p(y~|x)/d eta)", formula_kind="special", formula_ref="UPM Eq. (11)-(12)", paper="Universal Probability Model for Label Noise",
)
def upm_update_eta(ctx: ScratchContext, state: str = "upm_eta_state", psi_state: str = "upm_psi_state", indices: str = "indices", posterior: str = "clean_posterior", labels: str = "labels", learning_rate: float | None = None) -> None:
    import torch
    from ...native_stats import update_confusing_probability
    if learning_rate is None:
        raise ValueError("UPM eta update requires an explicit learning_rate")
    rows = torch.as_tensor(ctx[indices], dtype=torch.long).reshape(-1).cpu()
    eta_table, psi_table = ctx[state], ctx[psi_state]
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= int(eta_table["values"].shape[0])):
        raise IndexError("UPM eta state does not cover requested indices")
    eta = eta_table["values"][rows].reshape(-1)
    psi = psi_table["values"][rows].reshape(-1)
    updated = update_confusing_probability(eta.to(ctx[posterior]), ctx[posterior].detach(), ctx[labels].detach(), psi.to(ctx[posterior]), learning_rate=float(learning_rate), epsilon=1e-8).to(torch.device("cpu"))
    eta_table["values"][rows] = updated.reshape(-1, 1)
    eta_table["seen"][rows] = True


@block(
    id="cal_materialize_proxy_artifact",
    name="CAL: Materialize Warm-up Proxy Artifact",
    category="Posterior",
    description="Freeze the warm-up posterior into the stable-index CORES² proxy artifact used by the second stage.",
    params={"model": {"type": "slot", "default": "warmup_model"}, "loader": {"type": "slot", "default": "train_eval_loader"}, "prepared_data": {"type": "slot", "default": "prepared_data"}, "noisy_prior": {"type": "slot", "default": "cal_noisy_prior"}, "confidence_weight": {"type": "slot", "default": "confidence_weight"}, "lower_threshold": {"type": "float", "required": True}, "upper_threshold": {"type": "float", "required": True}, "proxy_as": {"type": "slot", "default": "cal_proxy_artifact"}, "proxy_prior_as": {"type": "slot", "default": "cal_proxy_prior"}, "reference_transition_as": {"type": "slot", "default": "cal_reference_transition"}, "reference_losses_as": {"type": "slot", "default": "cal_reference_losses"}, "proxy_targets_state_as": {"type": "slot", "default": "cal_proxy_targets_state"}, "retained_state_as": {"type": "slot", "default": "cal_retained_state"}},
    requires=("model", "loader", "prepared_data", "noisy_prior", "confidence_weight"), provides=("proxy_as", "proxy_prior_as", "reference_transition_as", "reference_losses_as", "proxy_targets_state_as", "retained_state_as"), placement=("top",), stage="setup", ui_group="⑥ 后验与权重",
    formula="proxy=CORES2(argmax f_warmup(x), adjusted_loss, lower, upper)", formula_kind="special", formula_ref="CAL proxy artifact lifecycle", paper="Learning from Noisy Labels with Core-loss and Second-order Risk",
)
def cal_materialize_proxy_artifact(ctx: ScratchContext, model: str = "warmup_model", loader: str = "train_eval_loader", prepared_data: str = "prepared_data", noisy_prior: str = "cal_noisy_prior", confidence_weight: str = "confidence_weight", lower_threshold: float | None = None, upper_threshold: float | None = None, proxy_as: str = "cal_proxy_artifact", proxy_prior_as: str = "cal_proxy_prior", reference_transition_as: str = "cal_reference_transition", reference_losses_as: str = "cal_reference_losses", proxy_targets_state_as: str = "cal_proxy_targets_state", retained_state_as: str = "cal_retained_state") -> None:
    import numpy as np
    import torch
    from ...native_stats import cores2_adjusted_losses, CALProxyArtifact, build_cal_proxy_artifact, _reference_transition_means, collect_posterior_snapshot
    if lower_threshold is None or upper_threshold is None:
        raise ValueError("CAL proxy artifact requires explicit lower_threshold and upper_threshold")
    prepared = ctx[prepared_data]; classes = int(prepared.num_classes)
    train_role = getattr(prepared, "datasets", {}).get("train")
    train_samples = tuple(getattr(train_role, "samples", ())) if train_role is not None else ()
    if len(train_samples) != len(getattr(prepared, "train_indices", ())):
        raise ValueError("CAL proxy artifact requires a train role with observed targets")
    observed_targets = np.asarray([int(sample.observed_target) for sample in train_samples], dtype=np.int64)
    if bool((ctx.get("_runtime_limits") or {}).get("fixture")):
        indices = np.asarray(prepared.train_indices, dtype=np.int64)
        artifact = CALProxyArtifact(indices, observed_targets, np.zeros(indices.size, dtype=np.int8), "fixture", float(lower_threshold), float(upper_threshold))
    else:
        device = next(ctx[model].parameters()).device
        snapshot = collect_posterior_snapshot(ctx[model], ctx[loader], device, dataset=prepared.dataset, split="train")
        all_losses, all_indices = [], []
        was_training = ctx[model].training; ctx[model].eval()
        with torch.inference_mode():
            for batch in ctx[loader]:
                logits = ctx[model](batch["input"].to(device))
                all_losses.append(cores2_adjusted_losses(logits, batch["target"].to(device), ctx[noisy_prior].to(device), float(ctx[confidence_weight])).cpu().numpy())
                all_indices.append(batch["index"].cpu().numpy())
        ctx[model].train(was_training)
        losses, indices = np.concatenate(all_losses), np.concatenate(all_indices)
        order = np.argsort(indices, kind="stable")
        if not np.array_equal(indices[order], snapshot.global_indices):
            raise ValueError("CAL adjusted-loss indices do not match warm-up posterior snapshot")
        artifact = build_cal_proxy_artifact(snapshot, losses[order], lower_threshold=float(lower_threshold), upper_threshold=float(upper_threshold))
    retained = artifact.sample_status != 2
    proxy_prior = np.bincount(artifact.proxy_targets[retained], minlength=classes).astype(np.float32)
    if proxy_prior.sum() <= 0:
        raise ValueError("CAL proxy artifact retained no samples")
    proxy_prior_value = torch.as_tensor(proxy_prior / proxy_prior.sum(), dtype=torch.float32)
    reference_transition_value = _reference_transition_means(artifact, np.asarray(prepared.train_indices), observed_targets, classes)
    reference_losses_value = torch.zeros(classes, classes, dtype=torch.float32)
    # Publish the artifact and its stable-index tables as independent slots;
    # subsequent recipes consume them through the public indexed state blocks.
    indices = torch.as_tensor(artifact.global_indices, dtype=torch.long)
    size = int(indices.max().item()) + 1 if indices.numel() else 0
    def _table(values):
        values = torch.as_tensor(values)
        if values.ndim == 1:
            values = values[:, None]
        table = {"values": torch.zeros((size, values.shape[1]), dtype=values.dtype), "seen": torch.zeros(size, dtype=torch.bool), "last_epoch": torch.full((size,), -1, dtype=torch.long)}
        table["values"][indices] = values.detach().cpu(); table["seen"][indices] = True
        return table
    ctx[proxy_as] = artifact
    ctx[proxy_prior_as] = proxy_prior_value
    ctx[reference_transition_as] = torch.as_tensor(reference_transition_value, dtype=torch.float32)
    ctx[reference_losses_as] = reference_losses_value
    ctx[proxy_targets_state_as] = _table(artifact.proxy_targets)
    ctx[retained_state_as] = _table(retained)


@block(id="cal_cores2_adjusted_risk", name="CAL: CORES2 Adjusted Risk", category="Loss", description="Compute CAL Eq. (7) with the square-root noisy prior used by the paper's confidence regularizer.", params={"logits":{"type":"slot","default":"logits"},"labels":{"type":"slot","default":"labels"},"noisy_prior":{"type":"slot","default":"cal_noisy_prior"},"confidence_weight":{"type":"slot","default":"confidence_weight"},"save_as":{"type":"slot","default":"cal_adjusted_risk"}}, requires=("logits","labels","noisy_prior","confidence_weight"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式", formula="mean[-log p_y-alpha sum_c sqrt(pi_c)/sum_j sqrt(pi_j) log p_c]", formula_kind="composite", formula_ref="CAL Eq. (7)", paper="Learning from Noisy Labels with Core-loss and Second-order Risk")
def cal_cores2_adjusted_risk(ctx: ScratchContext, logits: str="logits", labels: str="labels", noisy_prior: str="cal_noisy_prior", confidence_weight: str="confidence_weight", save_as: str="cal_adjusted_risk") -> None:
    torch,F=_torch(); probability=F.softmax(ctx[logits],dim=1); observed=-torch.log(probability+1.0e-8).gather(1,ctx[labels].long()[:,None]).squeeze(1); all_losses=-torch.log(probability+1.0e-5); prior=ctx[noisy_prior].to(all_losses).clamp_min(0).sqrt(); prior=prior/prior.sum().clamp_min(torch.finfo(all_losses.dtype).tiny); ctx[save_as]=(observed-float(ctx[confidence_weight])*(all_losses*prior).sum(1)).mean()


@block(id="cal_covariance_correction", name="CAL: Covariance Correction", category="Loss", description="Compute the Eq. (8)-(9) retained-proxy covariance correction from detached reference matrices.", params={"logits":{"type":"slot","default":"logits"},"labels":{"type":"slot","default":"labels"},"proxy_targets":{"type":"slot","default":"cal_proxy_targets"},"retained":{"type":"slot","default":"cal_retained"},"proxy_prior":{"type":"slot","default":"cal_proxy_prior"},"reference_losses":{"type":"slot","default":"cal_reference_losses"},"reference_transition":{"type":"slot","default":"cal_reference_transition"},"save_as":{"type":"slot","default":"cal_covariance"}}, requires=("logits","labels","proxy_targets","retained","proxy_prior","reference_losses","reference_transition"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式", formula="sum_c pi_hat_c Cov(1[y~=j],ell_j | yhat=c)", formula_kind="special", formula_ref="CAL Eq. (8)-(9)", paper="Learning from Noisy Labels with Core-loss and Second-order Risk")
def cal_covariance_correction(ctx: ScratchContext, logits: str="logits", labels: str="labels", proxy_targets: str="cal_proxy_targets", retained: str="cal_retained", proxy_prior: str="cal_proxy_prior", reference_losses: str="cal_reference_losses", reference_transition: str="cal_reference_transition", save_as: str="cal_covariance") -> None:
    torch,F=_torch(); losses=-torch.log(F.softmax(ctx[logits],dim=1)+1.0e-5); classes=losses.shape[1]; correction=losses.sum()*0.0; prior=ctx[proxy_prior].to(losses); means=ctx[reference_losses].detach().to(losses); transition=ctx[reference_transition].detach().to(losses); proxy_values=ctx[proxy_targets].reshape(-1); retained_values=ctx[retained].bool().reshape(-1)
    for c in range(classes):
        mask=retained_values & proxy_values.eq(c)
        if not bool(mask.any()): continue
        selected=losses[mask]; observed=ctx[labels][mask].long()
        for j in range(classes): correction=correction+prior[c]*((observed.eq(j).to(losses.dtype)-transition[c,j])*(selected[:,j]-means[c,j])).mean()
    ctx[save_as]=correction


@block(
    id="mc_ldce_objective",
    name="MC-LDCE: Fixed-feature Global Risk",
    category="Loss",
    description="Apply the bias-free classifier's global squared centroid risk against the estimated statistic.",
    params={"model": {"type": "slot", "default": "model"}, "logits": {"type": "slot", "default": "logits"}, "features": {"type": "slot", "default": "features"}, "statistic": {"type": "slot", "default": "mc_ldce_statistic"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("model", "logits", "features", "statistic"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="J=1+mean||hW^T||²-2 sum(W*mu_clean)", formula_kind="composite", formula_ref="MC-LDCE Eq. (30)", paper="MC-LDCE",
)
def mc_ldce_objective(ctx: ScratchContext, model: str = "model", logits: str = "logits", features: str = "features", statistic: str = "mc_ldce_statistic", save_as: str = "loss") -> None:
    torch = __import__("torch")
    weight = next((parameter for name, parameter in ctx[model].named_parameters() if name.endswith("classifier.weight")), None)
    if weight is None:
        raise ValueError("MC-LDCE requires a classifier.weight parameter")
    centroid = torch.as_tensor(ctx[statistic].values, dtype=ctx[features].dtype, device=ctx[features].device)
    scores = ctx[features] @ weight.transpose(0, 1)
    ctx[save_as] = 1.0 + scores.square().sum(1).mean() - 2.0 * (weight.to(ctx[features]) * centroid).sum()


@block(
    id="mc_ldce_volmin_objective",
    name="MC-LDCE: Paper VolMin Objective",
    category="Loss",
    description="Apply the PaperVolMin noisy-label likelihood plus volume penalty to the independent transition estimator.",
    params={"logits": {"type": "slot", "default": "transition_logits"}, "labels": {"type": "slot", "default": "labels"}, "transition": {"type": "slot", "default": "mc_ldce_transition"}, "lambda_volume": {"type": "float", "default": 0.0001, "min": 0.0}, "determinant_tolerance": {"type": "float", "default": 1.0e-8, "min": 0.0}, "condition_limit": {"type": "float", "default": 1.0e8, "min": 1.0}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "labels", "transition"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L=-log((T^T p)_y)+lambda logdet(T)", formula_kind="composite", formula_ref="MC-LDCE PaperVolMin transition stage", paper="MC-LDCE",
)
def mc_ldce_volmin_objective(ctx: ScratchContext, logits: str = "transition_logits", labels: str = "labels", transition: str = "mc_ldce_transition", lambda_volume: float = 0.0001, determinant_tolerance: float = 1.0e-8, condition_limit: float = 1.0e8, save_as: str = "loss") -> None:
    from ...native_stats import paper_volmin_objective
    loss, diagnostics = paper_volmin_objective(ctx[logits].to(dtype=__import__("torch").float64), ctx[labels], ctx[transition].matrix(), lambda_volume=float(lambda_volume), determinant_tolerance=float(determinant_tolerance), condition_limit=float(condition_limit))
    ctx[save_as], ctx["mc_ldce_volmin_metrics"] = loss, diagnostics


@block(id="mc_ldce_recover_statistic", name="MC-LDCE: Recover Clean Centroids", category="Paper Specific", description="Recover the fixed clean class-centroid statistic from noisy feature centroids and the separately learned transition matrix.", params={"model":{"type":"slot","default":"model"},"loader":{"type":"slot","default":"train_eval_loader"},"transition":{"type":"slot","default":"mc_ldce_transition"},"num_classes":{"type":"int","default":10,"min":2},"save_as":{"type":"slot","default":"mc_ldce_statistic"}}, requires=("model","loader","transition"), provides=("save_as",), placement=("top",), stage="setup", ui_group="⑥ 后验与权重", formula="mu=mu_tilde pinv(sum_i pi_i sum_j T_ij swap(i,j)^T)", formula_kind="special", formula_ref="MC-LDCE centroid recovery", paper="MC-LDCE")
def mc_ldce_recover_statistic(ctx: ScratchContext, model: str="model", loader: str="train_eval_loader", transition: str="mc_ldce_transition", num_classes: int=10, save_as: str="mc_ldce_statistic") -> None:
    from types import SimpleNamespace
    torch,F=_torch(); network=ctx[model]; device=ctx.get("device",next(network.parameters()).device); feats=[]; targets=[]; network.eval()
    with torch.no_grad():
        for batch in ctx[loader]:
            if isinstance(batch,dict): x=batch.get("input",batch.get("images")); y=batch.get("target",batch.get("labels"))
            else: x,y=batch[:2]
            output=network.forward_with_features(x.to(device)); feature=output.features if hasattr(output,"features") else output[1]
            feats.append(feature.detach()); targets.append(y.to(device).long())
    features=torch.cat(feats); labels=torch.cat(targets); classes=int(num_classes); onehot=F.one_hot(labels,classes).to(features)
    noisy_centroid=features.T@onehot/float(labels.numel()); observed=onehot.mean(0); matrix=ctx[transition].matrix().detach().to(features)
    clean_prior=torch.linalg.lstsq(matrix.T,observed[:,None]).solution[:,0].clamp_min(0); clean_prior=clean_prior/clean_prior.sum().clamp_min(torch.finfo(features.dtype).eps)
    imputation=torch.zeros((classes,classes),device=features.device,dtype=features.dtype)
    eye=torch.eye(classes,device=features.device,dtype=features.dtype)
    for i in range(classes):
        for j in range(classes):
            swap=eye.clone(); swap[[i,j]]=swap[[j,i]]; imputation+=clean_prior[i]*matrix[i,j]*swap.T
    centroids=(noisy_centroid@torch.linalg.pinv(imputation)).T
    ctx[save_as]=SimpleNamespace(values=centroids.detach().cpu().numpy(),clean_prior=clean_prior.detach().cpu().numpy())


@block(
    id="cnlcu_soft_score",
    name="CNLCU Soft Selection Score",
    category="Sample Selection",
    description="Compute CNLCU Eq. (7)'s uncertainty-aware lower-bound score.",
    params={"robust_mean": {"type": "slot", "default": "cnlcu_robust_mean"}, "history_length": {"type": "slot", "default": "cnlcu_history_length"}, "selected_count": {"type": "slot", "default": "history_selected_count"}, "sigma_squared": {"type": "float", "default": 0.01, "min": 0.000001, "max": 0.999999}, "save_as": {"type": "slot", "default": "cnlcu_score"}},
    requires=("robust_mean", "history_length", "selected_count"), provides=("save_as", "cnlcu_bonus"), placement=("batch",), stage="train", ui_group="⑥ 样本选择",
    formula="score=r-σ²(t+σ² log(2t)/t²)/(n-σ²)", formula_kind="special", formula_ref="CNLCU Eq. (7)", paper="CNLCU",
)
def cnlcu_soft_score(ctx: ScratchContext, robust_mean: str = "cnlcu_robust_mean", history_length: str = "cnlcu_history_length", selected_count: str = "history_selected_count", sigma_squared: float = 0.01, save_as: str = "cnlcu_score") -> None:
    from ...native_stats import cnlcu_soft_score as score_fn
    robust = ctx[robust_mean]
    length = ctx[history_length]
    selected = ctx[selected_count]
    # Indexed state reads expose scalar tables as [N,1]; the CNLCU formula is
    # elementwise over samples, so remove that storage-only dimension here.
    if getattr(length, "ndim", 0) > 1 and int(length.shape[-1]) == 1:
        length = length.squeeze(-1)
    if getattr(selected, "ndim", 0) > 1 and int(selected.shape[-1]) == 1:
        selected = selected.squeeze(-1)
    score, bonus = score_fn(robust, length, selected + 1, float(sigma_squared))
    ctx[save_as], ctx["cnlcu_bonus"] = score, bonus
