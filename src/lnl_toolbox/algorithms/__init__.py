from .base import Algorithm, TrainState
from .coteaching import coteaching_exchange, remember_rate
from .multi_model import ModelGroup, PeerExchangeResult, SmallLossPeerExchange, consistency_loss
from .jocor import JoCoRAlgorithm, jocor_joint_scores, symmetric_kl_per_sample

__all__ = [
    "Algorithm",
    "TrainState",
    "coteaching_exchange",
    "remember_rate",
    "ModelGroup",
    "PeerExchangeResult",
    "SmallLossPeerExchange",
    "JoCoRAlgorithm",
    "jocor_joint_scores",
    "symmetric_kl_per_sample",
    "consistency_loss",
]
try:
    from .cdr import CDRUpdatePolicy, CriticalParameterMasks, critical_parameter_masks
    from .update_policy import (
        ParameterUpdateInput,
        ParameterUpdatePolicy,
        ParameterUpdateResult,
        StandardUpdatePolicy,
        StepMilestoneUpdatePolicy,
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
        "StepMilestoneUpdatePolicy",
        "critical_parameter_masks",
    ])
try:
    from .mentornet import MentorNetWeightProvider, MovingPercentileState
except ImportError:
    pass
else:
    __all__.extend(["MentorNetWeightProvider", "MovingPercentileState"])
try:
    from .binary_risk import BinaryRiskCorrector, LabelDependentCostRisk, NatarajanRisk, NatarajanUnbiasedRisk
except ImportError:
    pass
else:
    __all__.extend(["BinaryRiskCorrector", "LabelDependentCostRisk", "NatarajanRisk", "NatarajanUnbiasedRisk"])
try:
    from .supervised import SupervisedClassificationAlgorithm
except ImportError:
    SupervisedClassificationAlgorithm = None  # type: ignore[assignment]
try:
    from .dss import DSSObjective
    from .masked_risk import candidate_masked_cross_entropy
except ImportError:
    pass
else:
    __all__.extend(["DSSObjective", "candidate_masked_cross_entropy"])
try:
    from .transition_risk import BackwardRiskCorrector, ForwardRiskCorrector, RiskCorrector
except ImportError:
    pass
else:
    __all__.extend(["BackwardRiskCorrector", "ForwardRiskCorrector", "RiskCorrector"])
try:
    from .instance_transition import InstanceTransitionClassificationAlgorithm
except ImportError:
    pass
else:
    __all__.append("InstanceTransitionClassificationAlgorithm")
