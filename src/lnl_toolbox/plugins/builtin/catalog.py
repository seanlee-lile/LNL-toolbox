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
    generate_pairflip,
    generate_symmetric,
)
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
        from lnl_toolbox.algorithms.update_policy import StandardUpdatePolicy
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
        "noise",
        "instance_dependent",
        generate_instance_dependent,
        capabilities=("per_sample_state",),
        metadata={"example": True},
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
    try:
        from lnl_toolbox.noise.transition import TrainableTransitionModel
        from lnl_toolbox.algorithms.multi_model import SmallLossPeerExchange
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
        catalog.add(
            "risk_corrector", "backward", BackwardRiskCorrector,
            capabilities=("transition_consumer", "corrected_risk"),
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

