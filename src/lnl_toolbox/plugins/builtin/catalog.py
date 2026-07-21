from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from lnl_toolbox.algorithms.coteaching import coteaching_exchange
from lnl_toolbox.losses.numpy_losses import cross_entropy, generalized_cross_entropy
from lnl_toolbox.noise import generate_instance_dependent, generate_pairflip, generate_symmetric
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

