from __future__ import annotations

"""Pure dataset/method compatibility contracts used by discovery surfaces."""

from dataclasses import dataclass
from enum import Enum

from lnl_toolbox.data.profile import (
    DatasetCapabilities,
    KnowledgeState,
    Modality,
    NoiseOrigin,
    NoiseRateStatus,
)


@dataclass(frozen=True, slots=True)
class MethodRequirements:
    method: str
    supported_modalities: frozenset[Modality]
    min_classes: int | None = 2
    max_classes: int | None = None
    exact_classes: frozenset[int] = frozenset()
    requires_noise_manifest: bool = True
    requires_dataset_true_noise_rate: bool = False
    requires_method_noise_prior: bool = False
    requires_clean_train_labels: bool = False
    requires_clean_validation: bool = False
    requires_aligned_clean_noisy_targets: bool = False
    supports_native_noisy_labels: bool = False
    supports_synthetic_noise: bool = True
    supports_unknown_noise_rate: bool = True
    validation_target: str = "noisy"
    required_pretrained_roles: tuple[str, ...] = ()
    required_source_roles: tuple[str, ...] = ("train", "test")
    method_noise_prior_paths: tuple[tuple[str, ...], ...] = ()
    pretrained_role_paths: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        modalities = frozenset(Modality(item) for item in self.supported_modalities)
        if not modalities:
            raise ValueError("method requirements need at least one supported modality")
        object.__setattr__(self, "supported_modalities", modalities)
        if self.min_classes is not None and self.min_classes < 2:
            raise ValueError("min_classes must be at least two")
        if self.max_classes is not None and self.max_classes < 2:
            raise ValueError("max_classes must be at least two")
        if self.min_classes is not None and self.max_classes is not None and self.min_classes > self.max_classes:
            raise ValueError("min_classes must not exceed max_classes")
        if any(value < 2 for value in self.exact_classes):
            raise ValueError("exact class counts must be at least two")
        if self.validation_target not in {"clean", "noisy", "any"}:
            raise ValueError("validation_target must be clean, noisy, or any")
        object.__setattr__(self, "required_pretrained_roles", tuple(sorted(set(self.required_pretrained_roles))))
        object.__setattr__(self, "required_source_roles", tuple(sorted(set(self.required_source_roles))))
        object.__setattr__(
            self,
            "method_noise_prior_paths",
            tuple(tuple(str(part) for part in path) for path in self.method_noise_prior_paths),
        )
        object.__setattr__(
            self,
            "pretrained_role_paths",
            tuple(
                (str(role), tuple(str(part) for part in path))
                for role, path in self.pretrained_role_paths
            ),
        )


class CompatibilityStatus(str, Enum):
    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_REQUIREMENTS = "compatible_with_requirements"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, slots=True)
class CompatibilityReason:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    status: CompatibilityStatus
    method: str
    dataset: str
    reasons: tuple[CompatibilityReason, ...] = ()
    warnings: tuple[CompatibilityReason, ...] = ()
    required_user_inputs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "dataset": self.dataset,
            "status": self.status.value,
            "reason_codes": [item.code for item in self.reasons],
            "reasons": [
                {"code": item.code, "message": item.message}
                for item in self.reasons
            ],
            "warnings": [
                {"code": item.code, "message": item.message}
                for item in self.warnings
            ],
            "required_user_inputs": list(self.required_user_inputs),
        }


def requirements_unavailable_result(method: str, dataset: str) -> CompatibilityResult:
    """Represent missing runner metadata without claiming compatibility."""

    reason = CompatibilityReason(
        "requirements_unavailable",
        f"runner {method!r} does not publish dataset compatibility requirements",
    )
    return CompatibilityResult(
        status=CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS,
        method=method,
        dataset=dataset,
        reasons=(reason,),
        required_user_inputs=("method_requirements",),
    )


def resolve_compatibility(
    dataset: DatasetCapabilities,
    method: MethodRequirements,
) -> CompatibilityResult:
    """Compare metadata only; this function performs no I/O or training."""

    incompatible: list[CompatibilityReason] = []
    requirements: list[CompatibilityReason] = []
    warnings: list[CompatibilityReason] = []
    inputs: set[str] = set()

    if dataset.modality is Modality.UNKNOWN:
        requirements.append(CompatibilityReason("unknown_modality", "dataset modality must be confirmed"))
        inputs.add("modality")
    elif dataset.modality not in method.supported_modalities:
        incompatible.append(CompatibilityReason("unsupported_modality", f"{method.method} does not support {dataset.modality.value} data"))

    classes = dataset.num_classes
    if method.exact_classes and classes not in method.exact_classes:
        incompatible.append(CompatibilityReason("wrong_class_count", f"{method.method} requires class count in {sorted(method.exact_classes)}"))
    elif method.min_classes is not None and classes < method.min_classes:
        incompatible.append(CompatibilityReason("wrong_class_count", f"{method.method} requires at least {method.min_classes} classes"))
    elif method.max_classes is not None and classes > method.max_classes:
        incompatible.append(CompatibilityReason("wrong_class_count", f"{method.method} supports at most {method.max_classes} classes"))

    missing_roles = sorted(set(method.required_source_roles) - set(dataset.available_splits))
    if missing_roles:
        incompatible.append(CompatibilityReason("missing_source_role", "missing source split(s): " + ", ".join(missing_roles)))

    if dataset.observed_train_labels is KnowledgeState.UNAVAILABLE:
        incompatible.append(CompatibilityReason("missing_observed_train_labels", "observed training labels are unavailable"))
    elif dataset.observed_train_labels is KnowledgeState.UNKNOWN:
        requirements.append(CompatibilityReason("unknown_observed_train_labels", "observed training-label availability must be confirmed"))
        inputs.add("observed_train_labels")

    if method.requires_clean_train_labels or method.requires_aligned_clean_noisy_targets:
        if dataset.clean_train_labels is KnowledgeState.UNAVAILABLE:
            incompatible.append(CompatibilityReason("missing_clean_train_labels", "clean training labels are unavailable"))
        elif dataset.clean_train_labels is KnowledgeState.UNKNOWN:
            requirements.append(CompatibilityReason("unknown_clean_train_labels", "clean training-label availability must be confirmed"))
            inputs.add("clean_train_labels")
    if method.requires_aligned_clean_noisy_targets:
        if dataset.aligned_clean_noisy_targets is KnowledgeState.UNAVAILABLE:
            incompatible.append(CompatibilityReason("unaligned_clean_noisy_targets", "aligned clean/noisy targets are unavailable"))
        elif dataset.aligned_clean_noisy_targets is KnowledgeState.UNKNOWN:
            requirements.append(CompatibilityReason("unknown_clean_noisy_alignment", "clean/noisy target alignment must be confirmed"))
            inputs.add("aligned_clean_noisy_targets")

    if method.requires_clean_validation:
        clean_validation = dataset.clean_validation_labels
        if clean_validation is KnowledgeState.UNAVAILABLE and dataset.clean_train_labels is not KnowledgeState.AVAILABLE:
            incompatible.append(CompatibilityReason("missing_clean_validation", "clean validation labels cannot be provided"))
        elif clean_validation is KnowledgeState.UNKNOWN and dataset.clean_train_labels is KnowledgeState.UNKNOWN:
            requirements.append(CompatibilityReason("unknown_clean_validation", "clean validation labels or splittable clean train labels must be confirmed"))
            inputs.add("clean_validation_labels")

    if (
        dataset.noise_origin is NoiseOrigin.NATIVE
        and not method.supports_native_noisy_labels
        and dataset.aligned_clean_noisy_targets is not KnowledgeState.AVAILABLE
    ):
        incompatible.append(CompatibilityReason("unsupported_native_noise", f"{method.method} does not currently support observed-only native noise"))

    if method.requires_noise_manifest:
        can_generate = (
            dataset.aligned_clean_noisy_targets is KnowledgeState.AVAILABLE
            or (dataset.supports_synthetic_corruption and method.supports_synthetic_noise)
        )
        if dataset.noise_manifest is KnowledgeState.UNAVAILABLE and not can_generate:
            incompatible.append(CompatibilityReason("missing_noise_manifest", "required noise manifest is unavailable and cannot be generated"))
        elif dataset.noise_manifest is KnowledgeState.UNKNOWN and not can_generate:
            requirements.append(CompatibilityReason("missing_noise_manifest", "a compatible noise manifest must be provided"))
            inputs.add("noise_manifest")

    if method.requires_dataset_true_noise_rate:
        if dataset.noise_rate.status not in {NoiseRateStatus.KNOWN, NoiseRateStatus.ESTIMATED}:
            requirements.append(CompatibilityReason("unknown_noise_rate", "dataset noise rate is required"))
            inputs.add("dataset_noise_rate")
    elif dataset.noise_rate.status is NoiseRateStatus.UNKNOWN and not method.supports_unknown_noise_rate:
        requirements.append(CompatibilityReason("unknown_noise_rate", "this method does not accept an unknown dataset noise rate"))
        inputs.add("dataset_noise_rate")

    if method.requires_method_noise_prior and dataset.method_noise_rate_prior.status not in {NoiseRateStatus.KNOWN, NoiseRateStatus.ESTIMATED}:
        requirements.append(CompatibilityReason("requires_noise_rate_prior", "method noise-rate prior is required independently of the dataset true rate"))
        inputs.add("noise_rate_prior")

    missing_pretrained = sorted(set(method.required_pretrained_roles) - set(dataset.pretrained_roles))
    if missing_pretrained:
        requirements.append(CompatibilityReason("missing_pretrained_source", "missing pretrained role(s): " + ", ".join(missing_pretrained)))
        inputs.update(f"pretrained:{role}" for role in missing_pretrained)

    if dataset.stable_indices is KnowledgeState.UNAVAILABLE:
        incompatible.append(CompatibilityReason("missing_stable_indices", "stable sample indices are unavailable"))
    elif dataset.stable_indices is KnowledgeState.UNKNOWN:
        requirements.append(CompatibilityReason("unknown_stable_indices", "stable sample indices must be confirmed"))
        inputs.add("stable_indices")

    if incompatible:
        status = CompatibilityStatus.INCOMPATIBLE
        reasons = tuple(incompatible)
    elif requirements:
        status = CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS
        reasons = tuple(requirements)
    else:
        status = CompatibilityStatus.COMPATIBLE
        reasons = ()
    return CompatibilityResult(
        status=status, method=method.method, dataset=dataset.dataset,
        reasons=reasons, warnings=tuple(warnings),
        required_user_inputs=tuple(sorted(inputs)),
    )


__all__ = [
    "CompatibilityReason", "CompatibilityResult", "CompatibilityStatus",
    "MethodRequirements", "requirements_unavailable_result",
    "resolve_compatibility",
]
