from __future__ import annotations

"""Pure dataset/method compatibility contracts used by discovery surfaces."""

from dataclasses import dataclass
from enum import Enum
from collections.abc import Collection

from lnl_toolbox.data.contracts import DataRequirements, DataRole
from lnl_toolbox.data.profile import (
    DatasetCapabilities,
    KnowledgeState,
    Modality,
    NoiseOrigin,
    NoiseRateStatus,
)


@dataclass(frozen=True, slots=True)
class ConfigInputRequirement:
    """Declarative, non-executing requirement for experiment configuration."""

    code: str
    paths: tuple[tuple[str, ...], ...]
    mode: str = "all"
    description: str = "required method configuration is missing"

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        paths = tuple(tuple(str(part) for part in path) for path in self.paths)
        if not code:
            raise ValueError("config input requirement code must not be empty")
        if not paths or any(not path for path in paths):
            raise ValueError("config input requirement paths must not be empty")
        if self.mode not in {"all", "any"}:
            raise ValueError("config input requirement mode must be all or any")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "paths", paths)


@dataclass(frozen=True, slots=True)
class MethodRequirements:
    method: str
    supported_modalities: frozenset[Modality]
    data_requirements: DataRequirements
    implemented_variant: str = "default"
    implementation_limits: frozenset[str] = frozenset()
    min_classes: int | None = 2
    max_classes: int | None = None
    exact_classes: frozenset[int] = frozenset()
    requires_noise_manifest: bool | None = None
    requires_dataset_true_noise_rate: bool = False
    requires_method_noise_prior: bool = False
    requires_clean_train_labels: bool = False
    requires_clean_validation: bool | None = None
    requires_aligned_clean_noisy_targets: bool = False
    supports_native_noisy_labels: bool = False
    supports_synthetic_noise: bool = True
    supports_unknown_noise_rate: bool = True
    validation_target: str | None = None
    required_pretrained_roles: tuple[str, ...] = ()
    required_source_roles: tuple[str, ...] = ("train", "test")
    method_noise_prior_paths: tuple[tuple[str, ...], ...] = ()
    pretrained_role_paths: tuple[tuple[str, tuple[str, ...]], ...] = ()
    required_config_inputs: tuple[ConfigInputRequirement, ...] = ()

    def __post_init__(self) -> None:
        method = str(self.method).strip()
        variant = str(self.implemented_variant).strip()
        if not method or not variant:
            raise ValueError("method and implemented_variant must not be empty")
        if not isinstance(self.data_requirements, DataRequirements):
            raise TypeError("data_requirements must be a DataRequirements instance")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "implemented_variant", variant)
        object.__setattr__(
            self,
            "implementation_limits",
            frozenset(str(item).strip() for item in self.implementation_limits if str(item).strip()),
        )
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
        derived_manifest = bool(self.data_requirements.needs_noise_manifest)
        if self.requires_noise_manifest is not None and bool(self.requires_noise_manifest) != derived_manifest:
            raise ValueError("requires_noise_manifest conflicts with data_requirements")
        object.__setattr__(self, "requires_noise_manifest", derived_manifest)
        clean_validation = DataRole.CLEAN_VALIDATION in self.data_requirements.roles
        if self.requires_clean_validation is not None and bool(self.requires_clean_validation) != clean_validation:
            raise ValueError("requires_clean_validation conflicts with data_requirements")
        object.__setattr__(self, "requires_clean_validation", clean_validation)
        if DataRole.CLEAN_VALIDATION in self.data_requirements.roles:
            derived_target = "clean"
        elif DataRole.NOISY_VALIDATION in self.data_requirements.roles:
            derived_target = "noisy"
        else:
            derived_target = "any"
        if self.validation_target is not None and self.validation_target != derived_target:
            raise ValueError("validation_target conflicts with data_requirements")
        object.__setattr__(self, "validation_target", derived_target)
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
        object.__setattr__(
            self,
            "required_config_inputs",
            tuple(
                item
                if isinstance(item, ConfigInputRequirement)
                else ConfigInputRequirement(**item)
                for item in self.required_config_inputs
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
    origin: str = "algorithm_requirement"


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    status: CompatibilityStatus
    method: str
    dataset: str
    reasons: tuple[CompatibilityReason, ...] = ()
    warnings: tuple[CompatibilityReason, ...] = ()
    required_user_inputs: tuple[str, ...] = ()
    required_input_paths: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = ()

    def __post_init__(self) -> None:
        normalized = []
        for code, paths in self.required_input_paths:
            normalized.append((str(code), tuple(tuple(str(part) for part in path) for path in paths)))
        object.__setattr__(self, "required_input_paths", tuple(sorted(normalized)))

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "dataset": self.dataset,
            "status": self.status.value,
            "reason_codes": [item.code for item in self.reasons],
            "reasons": [
                {"code": item.code, "message": item.message, "origin": item.origin}
                for item in self.reasons
            ],
            "warnings": [
                {"code": item.code, "message": item.message, "origin": item.origin}
                for item in self.warnings
            ],
            "required_user_inputs": list(self.required_user_inputs),
            "required_input_paths": {
                code: [list(path) for path in paths]
                for code, paths in self.required_input_paths
            },
        }


def requirements_unavailable_result(method: str, dataset: str) -> CompatibilityResult:
    """Represent missing runner metadata without claiming compatibility."""

    reason = CompatibilityReason(
        "method_metadata_error",
        f"runner {method!r} does not publish dataset compatibility requirements",
    )
    return CompatibilityResult(
        status=CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS,
        method=method,
        dataset=dataset,
        reasons=(reason,),
        required_user_inputs=(),
    )


def resolve_compatibility(
    dataset: DatasetCapabilities,
    method: MethodRequirements,
    *,
    method_noise_rate_prior: object | None = None,
    available_pretrained_roles: Collection[str] = (),
) -> CompatibilityResult:
    """Compare metadata only; this function performs no I/O or training."""

    incompatible: list[CompatibilityReason] = []
    requirements: list[CompatibilityReason] = []
    warnings: list[CompatibilityReason] = []
    inputs: set[str] = set()
    input_paths: dict[str, tuple[tuple[str, ...], ...]] = {}

    def reason(code: str, message: str, limit: str | None = None) -> CompatibilityReason:
        if limit is not None and limit in method.implementation_limits:
            return CompatibilityReason(
                code,
                f"implemented variant {method.implemented_variant!r}: {message}",
                "implemented_variant_limit",
            )
        return CompatibilityReason(code, message)

    if dataset.modality is Modality.UNKNOWN:
        requirements.append(CompatibilityReason("unknown_modality", "dataset modality must be confirmed"))
    elif dataset.modality not in method.supported_modalities:
        incompatible.append(reason("unsupported_modality", f"does not support {dataset.modality.value} data", "modality"))

    classes = dataset.num_classes
    if method.exact_classes and classes not in method.exact_classes:
        incompatible.append(reason("wrong_class_count", f"requires class count in {sorted(method.exact_classes)}", "class_count"))
    elif method.min_classes is not None and classes < method.min_classes:
        incompatible.append(reason("wrong_class_count", f"requires at least {method.min_classes} classes", "class_count"))
    elif method.max_classes is not None and classes > method.max_classes:
        incompatible.append(reason("wrong_class_count", f"supports at most {method.max_classes} classes", "class_count"))

    missing_roles = sorted(set(method.required_source_roles) - set(dataset.available_splits))
    if missing_roles:
        incompatible.append(CompatibilityReason("missing_source_role", "missing source split(s): " + ", ".join(missing_roles)))

    if dataset.observed_train_labels is KnowledgeState.UNAVAILABLE:
        incompatible.append(CompatibilityReason("missing_observed_train_labels", "observed training labels are unavailable"))
    elif dataset.observed_train_labels is KnowledgeState.UNKNOWN:
        requirements.append(CompatibilityReason("unknown_observed_train_labels", "adapter/inspection metadata does not establish observed training-label availability"))

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
            requirements.append(CompatibilityReason("unknown_clean_noisy_alignment", "adapter/inspection metadata does not establish clean/noisy target alignment"))

    if method.requires_clean_validation:
        clean_validation = dataset.clean_validation_labels
        has_protocol_test = "test" in dataset.available_splits
        if (
            clean_validation is KnowledgeState.UNAVAILABLE
            and dataset.clean_train_labels is not KnowledgeState.AVAILABLE
            and not has_protocol_test
        ):
            incompatible.append(CompatibilityReason("missing_clean_validation", "clean validation labels cannot be provided"))
        elif (
            clean_validation is KnowledgeState.UNKNOWN
            and dataset.clean_train_labels is KnowledgeState.UNKNOWN
            and not has_protocol_test
        ):
            requirements.append(CompatibilityReason("unknown_clean_validation", "adapter/inspection metadata does not establish a clean validation source"))

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

    prior_status = getattr(method_noise_rate_prior, "status", NoiseRateStatus.UNKNOWN)
    if method.requires_method_noise_prior and not method.method_noise_prior_paths:
        requirements.append(CompatibilityReason(
            "method_metadata_error",
            "method declares a noise-rate prior but does not publish its config path",
        ))
    elif method.requires_method_noise_prior and prior_status not in {NoiseRateStatus.KNOWN, NoiseRateStatus.ESTIMATED}:
        requirements.append(CompatibilityReason("requires_noise_rate_prior", "method noise-rate prior is required independently of the dataset true rate"))
        inputs.add("noise_rate_prior")
        input_paths["noise_rate_prior"] = method.method_noise_prior_paths

    missing_pretrained = sorted(set(method.required_pretrained_roles) - set(available_pretrained_roles))
    if missing_pretrained:
        requirements.append(CompatibilityReason("missing_pretrained_source", "missing pretrained role(s): " + ", ".join(missing_pretrained)))
        inputs.update(f"pretrained:{role}" for role in missing_pretrained)
        for role in missing_pretrained:
            paths = tuple(path for candidate_role, path in method.pretrained_role_paths if candidate_role == role)
            if paths:
                input_paths[f"pretrained:{role}"] = paths

    if dataset.stable_indices is KnowledgeState.UNAVAILABLE:
        incompatible.append(CompatibilityReason("missing_stable_indices", "stable sample indices are unavailable"))
    elif dataset.stable_indices is KnowledgeState.UNKNOWN:
        requirements.append(CompatibilityReason("unknown_stable_indices", "adapter/inspection metadata does not establish stable sample indices"))

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
        required_input_paths=tuple(input_paths.items()),
    )


__all__ = [
    "CompatibilityReason", "CompatibilityResult", "CompatibilityStatus",
    "ConfigInputRequirement", "MethodRequirements", "requirements_unavailable_result",
    "resolve_compatibility",
]
