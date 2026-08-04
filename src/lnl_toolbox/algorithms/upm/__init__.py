"""User-ready Universal Probabilistic Model workflow components."""

from .algorithm import UPMAlgorithm, UPMTargetProvider
from .config import UPMConfig, UPMConfusingConfig, UPMStageConfig
from .confusing import ConfusingProbabilityState
from .objective import (
    predict_true_posterior,
    soft_target_cross_entropy,
    update_confusing_probability,
)
from .posterior import ObservedNoisyProbabilityLookup
from .state import UPMPhase, UPMState

__all__ = [
    "ConfusingProbabilityState",
    "ObservedNoisyProbabilityLookup",
    "UPMAlgorithm",
    "UPMConfig",
    "UPMConfusingConfig",
    "UPMPhase",
    "UPMStageConfig",
    "UPMState",
    "UPMTargetProvider",
    "predict_true_posterior",
    "soft_target_cross_entropy",
    "update_confusing_probability",
]
