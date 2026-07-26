from .base import Algorithm, TrainState
from .coteaching import coteaching_exchange, remember_rate

__all__ = [
    "Algorithm",
    "TrainState",
    "coteaching_exchange",
    "remember_rate",
]
try:
    from .cdr import CDRUpdatePolicy, CriticalParameterMasks, critical_parameter_masks
    from .update_policy import (
        ParameterUpdateInput,
        ParameterUpdatePolicy,
        ParameterUpdateResult,
        StandardUpdatePolicy,
    )
except ImportError:
    pass  # PyTorch-backed update policies are optional with the training stack.
else:
    __all__.extend([
        "CDRUpdatePolicy",
        "CriticalParameterMasks",
        "ParameterUpdateInput",
        "ParameterUpdatePolicy",
        "ParameterUpdateResult",
        "StandardUpdatePolicy",
        "critical_parameter_masks",
    ])
try:
    from .supervised import SupervisedClassificationAlgorithm
except ImportError:
    SupervisedClassificationAlgorithm = None  # type: ignore[assignment]
try:
    from .transition_risk import BackwardRiskCorrector, ForwardRiskCorrector, RiskCorrector
except ImportError:
    pass
else:
    __all__.extend(["BackwardRiskCorrector", "ForwardRiskCorrector", "RiskCorrector"])
