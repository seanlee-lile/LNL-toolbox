"""Internal sample-treatment contracts and legacy Selector adaptation."""

from .base import ContributionResult, validate_contribution_result
from .reduction import ReductionSpec, reduce_per_sample_loss
from .selector_adapter import SelectorContributionAdapter
from .weights import (
    BinaryRCNWeightInput,
    BinaryRCNImportanceWeightProvider,
    WeightContributionAdapter,
    WeightInput,
    WeightProvider,
    WeightResult,
    StatefulWeightProvider,
    validate_binary_rcn_weight_input,
    validate_weight_result,
)

__all__ = [
    "BinaryRCNImportanceWeightProvider",
    "BinaryRCNWeightInput",
    "ContributionResult",
    "ReductionSpec",
    "SelectorContributionAdapter",
    "WeightContributionAdapter",
    "WeightInput",
    "WeightProvider",
    "StatefulWeightProvider",
    "WeightResult",
    "reduce_per_sample_loss",
    "validate_binary_rcn_weight_input",
    "validate_contribution_result",
    "validate_weight_result",
]
