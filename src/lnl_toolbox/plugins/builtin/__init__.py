"""Optional reference plugins; none are required by the framework core."""

from .catalog import (
    build_builtin_loss,
    build_builtin_instance_transition_algorithm,
    build_builtin_instance_transition_estimator,
    build_builtin_multi_model_algorithm,
    build_builtin_objective_consumer,
    build_builtin_fine_selector,
    build_builtin_parameter_update_policy,
    build_builtin_pipeline,
    build_builtin_peer_exchange,
    build_builtin_risk_corrector,
    build_builtin_regularizer,
    build_builtin_statistic_estimator,
    build_builtin_selector,
    build_builtin_transition_estimator,
    build_builtin_transition_model,
    build_builtin_weight_provider,
    create_builtin_catalog,
)

__all__ = [
    "build_builtin_loss",
    "build_builtin_instance_transition_algorithm",
    "build_builtin_instance_transition_estimator",
    "build_builtin_multi_model_algorithm",
    "build_builtin_objective_consumer",
    "build_builtin_fine_selector",
    "build_builtin_parameter_update_policy",
    "build_builtin_pipeline",
    "build_builtin_peer_exchange",
    "build_builtin_risk_corrector",
    "build_builtin_regularizer",
    "build_builtin_statistic_estimator",
    "build_builtin_selector",
    "build_builtin_transition_estimator",
    "build_builtin_transition_model",
    "build_builtin_weight_provider",
    "create_builtin_catalog",
]

