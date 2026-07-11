from __future__ import annotations

from functools import partial

from lnl_toolbox.algorithms.coteaching import coteaching_exchange
from lnl_toolbox.losses import cross_entropy, generalized_cross_entropy
from lnl_toolbox.noise import generate_instance_dependent, generate_pairflip, generate_symmetric
from lnl_toolbox.plugins import PluginCatalog


def create_builtin_catalog() -> PluginCatalog:
    """Register small examples used to validate extension points."""

    catalog = PluginCatalog()
    catalog.add("loss", "ce", cross_entropy, capabilities=("per_sample",))
    catalog.add(
        "loss",
        "gce",
        partial(generalized_cross_entropy, q=0.7),
        capabilities=("per_sample", "noise_robust"),
        metadata={"example": True},
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

