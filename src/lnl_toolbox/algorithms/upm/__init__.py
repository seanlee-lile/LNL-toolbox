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

# Preserve the old equation-level names used by the synthetic compatibility
# runner.  The canonical package exports the stricter tensor/state API above;
# these aliases keep the existing runner and its checkpoints callable while
# both implementations share the same public ``algorithms.upm`` namespace.
import importlib.util as _importlib_util
from pathlib import Path as _Path
import sys as _sys

_legacy_spec = _importlib_util.spec_from_file_location(
    "_lnl_toolbox_legacy_upm", _Path(__file__).resolve().parent.parent / "upm.py"
)
if _legacy_spec is not None and _legacy_spec.loader is not None:
    _legacy_module = _importlib_util.module_from_spec(_legacy_spec)
    _sys.modules[_legacy_spec.name] = _legacy_module
    _legacy_spec.loader.exec_module(_legacy_module)
    estimate_clean_posterior = _legacy_module.estimate_clean_posterior
    update_confusion_probabilities_ = _legacy_module.update_confusion_probabilities_
    upm_soft_target_objective = _legacy_module.upm_soft_target_objective

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
    "estimate_clean_posterior",
    "update_confusion_probabilities_",
    "upm_soft_target_objective",
]
