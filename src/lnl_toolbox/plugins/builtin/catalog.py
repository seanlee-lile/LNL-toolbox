from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from lnl_toolbox.algorithms.coteaching import coteaching_exchange
from lnl_toolbox.losses.numpy_losses import cross_entropy, generalized_cross_entropy
from lnl_toolbox.noise import (
    AnchorTransitionEstimator,
    DualTransitionEstimator,
    generate_instance_dependent,
    generate_pdl_idn,
    generate_pairflip,
    generate_symmetric,
)
from lnl_toolbox.noise.estimators import KnownTransitionEstimator
from lnl_toolbox.noise.generators import generate_class_conditional
from lnl_toolbox.plugins import PluginCatalog


def create_builtin_catalog() -> PluginCatalog:
    """Register small examples used to validate extension points."""

    catalog = PluginCatalog()
    catalog.add("numpy_loss", "ce", cross_entropy, capabilities=("per_sample", "reference"))
    catalog.add(
        "numpy_loss",
        "gce",
        partial(generalized_cross_entropy, q=0.7),
        capabilities=("per_sample", "noise_robust", "reference"),
        metadata={"example": True},
    )

    try:
        from lnl_toolbox.losses.torch_losses import (
            ActivePassiveLoss,
            CrossEntropyLoss,
            GeneralizedCrossEntropyLoss,
            MeanAbsoluteErrorLoss,
            NormalizedCrossEntropyLoss,
            ReverseCrossEntropyLoss,
        )
    except ImportError:
        pass  # PyTorch is optional for the task-neutral package core.
    else:
        common = ("per_sample", "torch")
        catalog.add("loss", "ce", CrossEntropyLoss, capabilities=common)
        catalog.add(
            "loss", "gce", GeneralizedCrossEntropyLoss,
            capabilities=common + ("noise_robust",),
        )
        catalog.add(
            "loss", "nce", NormalizedCrossEntropyLoss,
            capabilities=common + ("normalized", "noise_robust"),
        )
        catalog.add(
            "loss", "mae", MeanAbsoluteErrorLoss,
            capabilities=common + ("passive", "noise_robust"),
        )
        catalog.add(
            "loss", "rce", ReverseCrossEntropyLoss,
            capabilities=common + ("passive", "noise_robust"),
        )
        catalog.add(
            "loss", "apl", ActivePassiveLoss,
            capabilities=common + ("composite", "noise_robust"),
        )
    try:
        from lnl_toolbox.algorithms.fine import FINERegularizer
    except ImportError:
        pass
    else:
        catalog.add(
            "regularizer", "fine", FINERegularizer,
            capabilities=("noisy_only", "active_forgetting", "negative_learning"),
            metadata={"paper": "FINE active forgetting and noise suppression"},
        )
        from lnl_toolbox.selectors.sed import SEDSelector
        catalog.add(
            "fine_selector", "sed", SEDSelector,
            capabilities=("fine_selection", "fine", "stateless"),
        )
    try:
        from lnl_toolbox.selectors import AllSelector, SmallLossSelector
    except ImportError:
        pass  # Generic selectors are optional with the PyTorch training stack.
    else:
        selector_capabilities = ("hard_selection", "single_batch", "stateless")
        catalog.add(
            "batch_selector",
            "all",
            AllSelector,
            capabilities=selector_capabilities,
        )
        catalog.add(
            "batch_selector",
            "small_loss",
            SmallLossSelector,
            capabilities=selector_capabilities + ("score_ranking",),
        )
    try:
        from lnl_toolbox.algorithms.cdr import CDRUpdatePolicy
        from lnl_toolbox.algorithms.update_policy import (
            StandardUpdatePolicy,
            StepMilestoneUpdatePolicy,
        )
    except ImportError:
        pass  # Parameter update policies require the PyTorch training stack.
    else:
        catalog.add(
            "parameter_update_policy",
            "standard",
            StandardUpdatePolicy,
            capabilities=("single_model", "optimizer_step", "stateless"),
        )
        catalog.add(
            "parameter_update_policy",
            "step_milestone",
            StepMilestoneUpdatePolicy,
            capabilities=("single_model", "optimizer_step", "step_schedule", "stateful"),
        )
        catalog.add(
            "parameter_update_policy",
            "cdr",
            CDRUpdatePolicy,
            capabilities=(
                "single_model",
                "optimizer_step",
                "gradient_transform",
                "parameter_mask",
                "stateless",
                "paper_reference",
            ),
            metadata={"paper": "Xia et al., ICLR 2021"},
        )
    catalog.add("noise", "symmetric", generate_symmetric, metadata={"example": True})
    catalog.add("noise", "pairflip", generate_pairflip, metadata={"example": True})
    catalog.add(
        "noise", "class_conditional", generate_class_conditional,
        capabilities=("class_conditional", "matrix_configured"),
    )
    catalog.add(
        "noise",
        "instance_dependent",
        generate_instance_dependent,
        capabilities=("per_sample_state",),
        metadata={"example": True},
    )
    catalog.add(
        "noise", "pdl", generate_pdl_idn,
        capabilities=("per_sample_state", "paper_benchmark"),
        metadata={"paper": "Xia et al., NeurIPS 2020, Algorithm 2"},
    )
    catalog.add(
        "selector",
        "coteaching_exchange",
        coteaching_exchange,
        capabilities=("multi_model", "sample_selection"),
        metadata={"example": True},
    )
    catalog.add(
        "transition_estimator",
        "anchor",
        AnchorTransitionEstimator,
        capabilities=("class_conditional", "offline", "paper_reference"),
        metadata={"paper": "Patrini et al., CVPR 2017"},
    )
    catalog.add(
        "transition_estimator",
        "dual_t",
        DualTransitionEstimator,
        capabilities=("class_conditional", "offline", "factorized", "paper_reference"),
        metadata={"paper": "Yao et al., NeurIPS 2020"},
    )
    catalog.add(
        "transition_estimator", "known", KnownTransitionEstimator,
        capabilities=("class_conditional", "configured", "offline"),
    )
    try:
        from lnl_toolbox.noise.pdl import PartTransitionEstimator
        from lnl_toolbox.algorithms.instance_transition import InstanceTransitionClassificationAlgorithm
    except ImportError:
        pass
    else:
        catalog.add(
            "instance_transition_estimator", "pdl", PartTransitionEstimator,
            capabilities=("instance_dependent", "feature_snapshot", "posterior_snapshot", "offline"),
            metadata={"paper": "Xia et al., NeurIPS 2020"},
        )
        catalog.add(
            "instance_transition_algorithm", "corrected_classification",
            InstanceTransitionClassificationAlgorithm,
            capabilities=("single_model", "instance_transition", "corrected_risk"),
        )
    try:
        from lnl_toolbox.noise.transition import TrainableTransitionModel
        from lnl_toolbox.algorithms.multi_model import SmallLossPeerExchange
        from lnl_toolbox.algorithms.jocor import JoCoRAlgorithm
    except ImportError:
        pass
    else:
        catalog.add(
            "transition_model",
            "trainable_global",
            TrainableTransitionModel,
            capabilities=("trainable", "global", "row_stochastic"),
        )
        catalog.add(
            "peer_exchange",
            "small_loss",
            SmallLossPeerExchange,
            capabilities=("multi_model", "sample_selection"),
        )
        catalog.add(
            "multi_model_algorithm",
            "jocor",
            JoCoRAlgorithm,
            capabilities=(
                "multi_model",
                "joint_selection",
                "agreement_regularization",
            ),
            metadata={"paper": "Wei et al., CVPR 2020"},
        )
    try:
        from lnl_toolbox.algorithms.transition_risk import (
            BackwardRiskCorrector,
            ForwardRiskCorrector,
        )
    except ImportError:
        pass
    else:
        catalog.add(
            "risk_corrector", "forward", ForwardRiskCorrector,
            capabilities=("transition_consumer", "corrected_risk"),
        )
    try:
        from lnl_toolbox.estimators.cwd import CWDEstimator
        from lnl_toolbox.algorithms.cwd import CWDGlobalObjective
    except ImportError:
        pass
    else:
        catalog.add(
            "statistic_estimator", "cwd", CWDEstimator,
            capabilities=("feature_snapshot", "classwise", "offline"),
            metadata={"paper": "CWD TPAMI 2022"},
        )
        catalog.add(
            "objective_consumer", "cwd", CWDGlobalObjective,
            capabilities=("feature_objective", "global_risk"),
            metadata={"paper": "CWD TPAMI 2022"},
        )
        catalog.add(
            "risk_corrector", "backward", BackwardRiskCorrector,
            capabilities=("transition_consumer", "corrected_risk"),
        )
    try:
        from lnl_toolbox.algorithms.dss import DSSObjective
    except ImportError:
        pass
    else:
        catalog.add(
            "objective_consumer",
            "dss",
            DSSObjective,
            capabilities=(
                "stateful_objective",
                "debiased_selection",
                "class_exclusion",
                "global_index",
            ),
            metadata={"paper": "Pan et al., CVPR 2026"},
        )
    try:
        from lnl_toolbox.training.pipeline import StandardNoisyERMPipeline
    except ImportError:
        pass
    else:
        catalog.add(
            "pipeline",
            "standard_noisy_erm",
            StandardNoisyERMPipeline.from_config,
            capabilities=("single_model", "stage_lifecycle", "artifact_handoff"),
        )
    try:
        from lnl_toolbox.algorithms.mentornet import MentorNetWeightProvider
    except ImportError:
        pass
    else:
        catalog.add(
            "weight_provider",
            "mentornet",
            MentorNetWeightProvider,
            capabilities=("continuous_weight", "learned_curriculum", "stateful"),
            metadata={"paper": "Jiang et al., ICML 2018"},
        )
    return catalog


def build_builtin_loss(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build a trainable loss from the shared YAML-compatible mapping."""

    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Loss configuration must be a mapping")
    values = dict(config or {"name": "ce"})
    name = str(values.pop("name", "ce")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    if name == "gce" and "eps" in values:
        raise ValueError(
            "Standard GCE does not accept eps; remove it instead of implicitly truncating p_y"
        )
    if name == "apl":
        active_config = values.pop("active", {"name": "nce"})
        passive_config = values.pop("passive", {"name": "rce", "log_zero": -4.0})
        if not isinstance(active_config, Mapping) or not isinstance(passive_config, Mapping):
            raise TypeError("APL active and passive settings must be mappings")
        active_name = str(active_config.get("name", "")).strip().lower()
        passive_name = str(passive_config.get("name", "")).strip().lower()
        if active_name != "nce":
            raise ValueError("P0 APL supports nce as its active loss")
        if passive_name not in {"mae", "rce"}:
            raise ValueError("P0 APL supports mae or rce as its passive loss")
        values["active"] = build_builtin_loss(active_config, catalog)
        values["passive"] = build_builtin_loss(passive_config, catalog)
    try:
        return catalog.build("loss", name, **values)
    except KeyError as exc:
        available = ", ".join(item.name for item in catalog.find(kind="loss")) or "none"
        raise ValueError(f"Unknown trainable loss {name!r}; available: {available}") from exc


def build_builtin_transition_estimator(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build an offline transition estimator from a YAML-compatible mapping."""

    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Transition estimator configuration must be a mapping")
    values = dict(config or {"name": "anchor"})
    name = str(values.pop("name", "anchor")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("transition_estimator", name, **values)
    except KeyError as exc:
        available = ", ".join(
            item.name for item in catalog.find(kind="transition_estimator")
        ) or "none"
        raise ValueError(
            f"Unknown transition estimator {name!r}; available: {available}"
        ) from exc


def build_builtin_risk_corrector(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build a transition-consuming risk corrector from configuration."""

    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Risk corrector configuration must be a mapping")
    values = dict(config or {"name": "forward"})
    name = str(values.pop("name", "forward")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("risk_corrector", name, **values)
    except KeyError as exc:
        available = ", ".join(
            item.name for item in catalog.find(kind="risk_corrector")
        ) or "none"
        raise ValueError(
            f"Unknown risk corrector {name!r}; available: {available}"
        ) from exc


def build_builtin_weight_provider(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build a continuous sample-weight provider from configuration."""

    if config is None:
        raise ValueError("Weight provider configuration must be explicit")
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Weight provider configuration must be a mapping")
    values = dict(config)
    if "name" not in values:
        raise ValueError("Weight provider configuration requires a name")
    name = str(values.pop("name")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("weight_provider", name, **values)
    except KeyError as exc:
        available = ", ".join(
            item.name for item in catalog.find(kind="weight_provider")
        ) or "none"
        raise ValueError(
            f"Unknown weight provider {name!r}; available: {available}"
        ) from exc


def build_builtin_instance_transition_estimator(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build an estimator that emits a per-sample transition provider."""

    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Instance transition estimator configuration must be a mapping")
    values = dict(config or {"name": "pdl"})
    name = str(values.pop("name", "pdl")).strip().lower()
    factorization = values.pop("factorization", None)
    anchors = values.pop("anchors", None)
    if factorization is not None:
        if not isinstance(factorization, Mapping):
            raise TypeError("factorization configuration must be a mapping")
        values["representation_iterations"] = int(factorization.get("iterations", 200))
        values["representation_seed"] = int(factorization.get("seed", 0))
    if anchors is not None:
        if not isinstance(anchors, Mapping):
            raise TypeError("anchors configuration must be a mapping")
        values["anchor_candidates"] = int(anchors["candidates_per_class"])
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("instance_transition_estimator", name, **values)
    except KeyError as exc:
        available = ", ".join(item.name for item in catalog.find(
            kind="instance_transition_estimator")) or "none"
        raise ValueError(
            f"Unknown instance transition estimator {name!r}; available: {available}"
        ) from exc


def build_builtin_instance_transition_algorithm(
    config: Mapping[str, Any],
    *,
    model: Any,
    optimizer: Any,
    loss: Any,
    transition: Any,
    device: Any,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build a generic consumer of ``[B,C,C]`` transition providers."""

    if not isinstance(config, Mapping):
        raise TypeError("Instance transition algorithm configuration must be a mapping")
    values = dict(config)
    name = str(values.pop("name", "corrected_classification")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("instance_transition_algorithm", name, model=model,
            optimizer=optimizer, loss=loss, transition=transition, device=device, **values)
    except KeyError as exc:
        available = ", ".join(item.name for item in catalog.find(
            kind="instance_transition_algorithm")) or "none"
        raise ValueError(
            f"Unknown instance transition algorithm {name!r}; available: {available}"
        ) from exc


def build_builtin_regularizer(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build a generic objective regularizer without paper branches in runners."""

    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Regularizer configuration must be a mapping")
    values = dict(config or {})
    name = str(values.pop("name", "")).strip().lower()
    if not name:
        raise ValueError("Regularizer configuration requires a name")
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("regularizer", name, **values)
    except KeyError as exc:
        available = ", ".join(item.name for item in catalog.find(kind="regularizer")) or "none"
        raise ValueError(f"Unknown regularizer {name!r}; available: {available}") from exc


def build_builtin_statistic_estimator(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Statistic estimator configuration must be a mapping")
    values = dict(config or {})
    name = str(values.pop("name", "")).strip().lower()
    if not name:
        raise ValueError("Statistic estimator configuration requires a name")
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("statistic_estimator", name, **values)
    except KeyError as exc:
        available = ", ".join(item.name for item in catalog.find(kind="statistic_estimator")) or "none"
        raise ValueError(f"Unknown statistic estimator {name!r}; available: {available}") from exc


def build_builtin_objective_consumer(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Objective consumer configuration must be a mapping")
    values = dict(config or {})
    name = str(values.pop("name", "")).strip().lower()
    if not name:
        raise ValueError("Objective consumer configuration requires a name")
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("objective_consumer", name, **values)
    except KeyError as exc:
        available = ", ".join(item.name for item in catalog.find(kind="objective_consumer")) or "none"
        raise ValueError(f"Unknown objective consumer {name!r}; available: {available}") from exc


def build_builtin_fine_selector(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build FINE's selector in its own plugin kind to preserve old selectors."""

    if config is not None and not isinstance(config, Mapping):
        raise TypeError("FINE selector configuration must be a mapping")
    values = dict(config or {"name": "sed"})
    name = str(values.pop("name", "sed")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("fine_selector", name, **values)
    except KeyError as exc:
        raise ValueError(f"Unknown FINE selector {name!r}") from exc


def build_builtin_transition_model(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Transition model configuration must be a mapping")
    values = dict(config or {"name": "trainable_global"})
    name = str(values.pop("name", "trainable_global")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("transition_model", name, **values)
    except KeyError as exc:
        raise ValueError(f"Unknown transition model {name!r}") from exc


def build_builtin_peer_exchange(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("peer exchange configuration must be a mapping")
    values = dict(config or {"name": "small_loss"})
    name = str(values.pop("name", "small_loss")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("peer_exchange", name, **values)
    except KeyError as exc:
        raise ValueError(f"Unknown peer exchange {name!r}") from exc


def build_builtin_multi_model_algorithm(
    config: Mapping[str, Any],
    *,
    models: Any,
    optimizer: Any,
    loss: Any,
    selector: Any,
    device: Any,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build a multi-network algorithm from injected reusable components."""

    if not isinstance(config, Mapping):
        raise TypeError("multi-model algorithm configuration must be a mapping")
    values = dict(config)
    name = str(values.pop("name", "")).strip().lower()
    if not name:
        raise ValueError("multi-model algorithm name must not be empty")
    if "lambda" in values:
        values["lambda_"] = values.pop("lambda")
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build(
            "multi_model_algorithm",
            name,
            models=models,
            optimizer=optimizer,
            loss=loss,
            selector=selector,
            device=device,
            **values,
        )
    except KeyError as exc:
        available = ", ".join(
            item.name for item in catalog.find(kind="multi_model_algorithm")
        ) or "none"
        raise ValueError(
            f"Unknown multi-model algorithm {name!r}; available: {available}"
        ) from exc


def build_builtin_pipeline(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build a lifecycle pipeline without embedding paper-specific branches."""

    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Pipeline configuration must be a mapping")
    values = dict(config or {"name": "standard_noisy_erm"})
    name = str(values.pop("name", "standard_noisy_erm")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("pipeline", name, config=values)
    except KeyError as exc:
        available = ", ".join(
            item.name for item in catalog.find(kind="pipeline")
        ) or "none"
        raise ValueError(
            f"Unknown pipeline {name!r}; available: {available}"
        ) from exc


def build_builtin_selector(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build a stateless per-batch selector from a YAML-compatible mapping."""

    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Selector configuration must be a mapping")
    values = dict(config or {"name": "all"})
    name = str(values.pop("name", "all")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    if name == "sed":
        return build_builtin_fine_selector({"name": name, **values}, catalog)
    try:
        return catalog.build("batch_selector", name, **values)
    except KeyError as exc:
        available = (
            ", ".join(item.name for item in catalog.find(kind="batch_selector"))
            or "none"
        )
        raise ValueError(
            f"Unknown batch selector {name!r}; available: {available}"
        ) from exc


def build_builtin_parameter_update_policy(
    config: Mapping[str, Any] | None,
    catalog: PluginCatalog | None = None,
) -> Any:
    """Build the owner of backward and optimizer stepping."""

    if config is not None and not isinstance(config, Mapping):
        raise TypeError("Parameter update configuration must be a mapping")
    values = dict(config or {"name": "standard"})
    name = str(values.pop("name", "standard")).strip().lower()
    catalog = catalog or create_builtin_catalog()
    try:
        return catalog.build("parameter_update_policy", name, **values)
    except KeyError as exc:
        available = ", ".join(
            item.name for item in catalog.find(kind="parameter_update_policy")
        ) or "none"
        raise ValueError(
            f"Unknown parameter update policy {name!r}; available: {available}"
        ) from exc

