from __future__ import annotations

"""Pure dataset/method compatibility contracts used by discovery surfaces."""

from dataclasses import dataclass, replace
from enum import Enum
from collections.abc import Collection, Mapping

from lnl_toolbox.data.contracts import DataRequirements, DataRole
from lnl_toolbox.data.profile import (
    DatasetCapabilities,
    KnowledgeState,
    Modality,
    NoiseOrigin,
    NoiseRateStatus,
)
from lnl_toolbox.training.prerequisites import SourceDescriptor, ValidationMetadata


@dataclass(frozen=True, slots=True)
class ConfigInputRequirement:
    """Declarative, non-executing requirement for experiment configuration."""

    code: str
    paths: tuple[tuple[str, ...], ...]
    mode: str = "all"
    description: str = "required method configuration is missing"
    implementation_limit: str | None = None

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        paths = tuple(tuple(str(part) for part in path) for path in self.paths)
        if not code:
            raise ValueError("config input requirement code must not be empty")
        if not paths or any(not path for path in paths):
            raise ValueError("config input requirement paths must not be empty")
        if self.mode not in {"all", "any"}:
            raise ValueError("config input requirement mode must be all or any")
        implementation_limit = (
            None
            if self.implementation_limit is None
            else str(self.implementation_limit).strip()
        )
        if self.implementation_limit is not None and not implementation_limit:
            raise ValueError("implementation_limit must not be empty")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "implementation_limit", implementation_limit)


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
    prerequisites: tuple[SourceDescriptor, ...] = ()

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
        object.__setattr__(
            self,
            "prerequisites",
            tuple(
                item if isinstance(item, SourceDescriptor) else SourceDescriptor(**item)
                for item in self.prerequisites
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
class CompatibilityInputGuidance:
    """Human-readable resolution guidance without changing compatibility status."""

    input_id: str
    category: str
    label: str
    missing: str
    expected_value: str
    provision: str
    input_kind: str
    config_paths: tuple[tuple[str, ...], ...] = ()
    environment_variable: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.input_id,
            "category": self.category,
            "label": self.label,
            "missing": self.missing,
            "expected_value": self.expected_value,
            "provision": self.provision,
            "input_kind": self.input_kind,
            "config_paths": [list(path) for path in self.config_paths],
            "environment_variable": self.environment_variable,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    status: CompatibilityStatus
    method: str
    dataset: str
    reasons: tuple[CompatibilityReason, ...] = ()
    warnings: tuple[CompatibilityReason, ...] = ()
    required_user_inputs: tuple[str, ...] = ()
    required_input_paths: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = ()
    input_guidance: tuple[CompatibilityInputGuidance, ...] = ()
    prerequisites: tuple[ValidationMetadata, ...] = ()

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
            "input_guidance": [item.to_dict() for item in self.input_guidance],
            "prerequisites": [item.to_dict() for item in self.prerequisites],
        }


_INPUT_GUIDANCE: dict[str, tuple[str, str, str, str, str]] = {
    "clean_train_labels": (
        "dataset_fact", "干净训练标签可用性", "尚未确认是否有干净训练标签",
        "确认该数据集是否提供干净训练标签", "需要确认的数据集信息",
    ),
    "dataset_noise_rate": (
        "dataset_fact", "数据集真实噪声率", "数据集真实噪声率仍为 Unknown",
        "提供已知或估计的真实噪声率及其来源", "需要确认的数据集信息",
    ),
    "noise_rate_prior": (
        "method_input", "方法噪声率先验", "所选方法缺少独立的噪声率先验",
        "提供 0 到 1 之间的方法噪声率先验", "当前方法噪声率先验输入框",
    ),
    "noise_manifest": (
        "method_input", "噪声 manifest", "缺少可用且对齐的噪声 manifest",
        "提供可生成 manifest 的对齐标签/合成噪声数据，或方法要求的 manifest artifact",
        "数据准备或该方法的噪声配置；当前没有通用 YAML 路径",
    ),
    "config:requires_transition_matrix": (
        "method_input", "Transition Matrix", "缺少已知转移矩阵",
        "提供与类别数一致的 K×K transition matrix", "YAML 方法配置字段",
    ),
    "config:requires_transition_source": (
        "method_input", "Transition source", "缺少转移矩阵来源",
        "提供 transition artifact 或显式 K×K matrix", "YAML 方法配置字段",
    ),
    "config:requires_mentor_artifact": (
        "method_input", "MentorArtifact", "缺少冻结的 MentorArtifact",
        "提供由 Mentor preparation workflow 生成的合法 artifact 文件", "YAML 方法配置字段",
    ),
    "config:requires_binary_noise_prior": (
        "method_input", "类别条件噪声率", "缺少二分类正/负类噪声率",
        "同时提供 rho_positive 和 rho_negative", "YAML 方法配置字段",
    ),
    "config:requires_trusted_validation": (
        "method_input", "Trusted validation source", "缺少可信验证集来源",
        "声明受支持的 trusted-validation source", "YAML 方法配置字段",
    ),
    "config:requires_trusted_manifest": (
        "method_input", "Trusted validation manifest", "缺少可信样本 manifest",
        "提供与训练数据身份匹配的 trusted manifest 文件", "YAML 方法配置字段",
    ),
    "config:requires_external_noise_labels": (
        "method_input", "外部 clean/noisy labels", "缺少对齐的外部标签输入",
        "提供标签文件路径以及 clean/noisy key", "YAML 方法配置字段",
    ),
}

_REASON_GUIDANCE: dict[str, tuple[str, str, str, str, str, str]] = {
    "unknown_modality": (
        "dataset_preparation", "数据模态", "数据模态仍为 Unknown",
        "使用可识别该数据的 adapter，或补充 adapter semantic hints",
        "重新登记/检查数据或完善 adapter", "adapter_metadata",
    ),
    "unknown_observed_train_labels": (
        "dataset_preparation", "观测训练标签", "尚未确认观测训练标签是否可用",
        "使用能够实际读取观测训练标签的 adapter", "重新检查数据或完善 adapter", "adapter_metadata",
    ),
    "unknown_clean_noisy_alignment": (
        "dataset_preparation", "clean/noisy 标签对齐", "标签对齐状态仍为 Unknown",
        "提供稳定索引和可验证的 clean/noisy 对齐关系", "数据准备或 adapter", "adapter_metadata",
    ),
    "unknown_clean_validation": (
        "dataset_preparation", "干净验证集", "尚未建立干净验证来源",
        "提供原生干净验证集，或可从干净标签派生的验证 split", "数据准备或 adapter", "adapter_metadata",
    ),
    "unknown_stable_indices": (
        "dataset_preparation", "稳定样本索引", "稳定索引状态仍为 Unknown",
        "使用能够提供稳定 global indices 的 adapter", "重新检查数据或完善 adapter", "adapter_metadata",
    ),
    "method_metadata_error": (
        "developer_error", "方法兼容性元数据", "runner 的 requirement metadata 不完整",
        "该问题需要开发者修正，用户不应伪造输入", "开发者配置", "developer_configuration_error",
    ),
    "requirements_unavailable": (
        "developer_error", "方法兼容性元数据", "runner 未发布 compatibility requirements",
        "该问题需要开发者修正，用户不应伪造输入", "开发者配置", "developer_configuration_error",
    ),
}


def build_input_guidance(
    result: "CompatibilityResult",
    *,
    environment_variables: Mapping[str, str] | None = None,
) -> tuple[CompatibilityInputGuidance, ...]:
    """Describe how to resolve existing requirements without re-evaluating them."""

    paths = dict(result.required_input_paths)
    environments = dict(environment_variables or {})
    reason_messages = {item.code: item.message for item in result.reasons}
    guidance: list[CompatibilityInputGuidance] = []
    covered_reason_codes: set[str] = set()
    prerequisite_by_key = {
        item.descriptor.key: item for item in result.prerequisites
    }
    for input_id in result.required_user_inputs:
        if input_id.startswith("pretrained:"):
            role = input_id.split(":", 1)[1]
            prerequisite = prerequisite_by_key.get(role)
            environment = environments.get(input_id)
            if prerequisite is not None:
                descriptor = prerequisite.descriptor
                guidance.append(CompatibilityInputGuidance(
                    input_id=input_id,
                    category="method_input",
                    label=descriptor.name,
                    missing=prerequisite.message,
                    expected_value=descriptor.requirement + (
                        "; supported: " + ", ".join(descriptor.supported_sources)
                        if descriptor.supported_sources else ""
                    ),
                    provision=descriptor.provide,
                    input_kind=(
                        "environment_directory"
                        if descriptor.environment_variable
                        else "artifact_source"
                    ),
                    config_paths=descriptor.config_paths,
                    environment_variable=descriptor.environment_variable,
                ))
                covered_reason_codes.add("missing_pretrained_source")
                continue
            provision = (
                f"设置环境变量 {environment}=<run directory>"
                if environment else "按 recipe 的 pretrained source 配置提供运行目录"
            )
            guidance.append(CompatibilityInputGuidance(
                input_id=input_id,
                category="method_input",
                label=f"预训练运行结果：{role}",
                missing=f"缺少 pretrained role：{role}",
                expected_value="兼容的预训练 run directory，其中包含 best.pt 和 noise_manifest.npz",
                provision=provision,
                input_kind="environment_directory" if environment else "artifact_directory",
                config_paths=paths.get(input_id, ()),
                environment_variable=environment,
            ))
            covered_reason_codes.add("missing_pretrained_source")
            continue
        spec = _INPUT_GUIDANCE.get(input_id)
        if spec is None and input_id.startswith("config:"):
            code = input_id.split(":", 1)[1]
            description = reason_messages.get(code, "缺少方法配置输入")
            spec = (
                "method_input", code, description, "提供该方法要求的配置值", "YAML 方法配置字段",
            )
        if spec is None:
            spec = (
                "method_input", input_id, reason_messages.get(input_id, f"缺少 {input_id}"),
                "提供该方法要求的输入", "YAML 或方法配置",
            )
        category, label, missing, expected, provision = spec
        input_paths = paths.get(input_id, ())
        if input_paths and category == "method_input" and input_id != "noise_rate_prior":
            provision = "YAML：" + " 或 ".join(".".join(path) for path in input_paths)
        guidance.append(CompatibilityInputGuidance(
            input_id=input_id,
            category=category,
            label=label,
            missing=missing,
            expected_value=expected,
            provision=provision,
            input_kind=("dataset_declaration" if category == "dataset_fact" else "config_value"),
            config_paths=input_paths,
        ))
        covered_reason_codes.add({
            "clean_train_labels": "unknown_clean_train_labels",
            "dataset_noise_rate": "unknown_noise_rate",
            "noise_rate_prior": "requires_noise_rate_prior",
            "noise_manifest": "missing_noise_manifest",
        }.get(input_id, input_id.removeprefix("config:")))
    for reason in result.reasons:
        if reason.code in covered_reason_codes:
            continue
        spec = _REASON_GUIDANCE.get(reason.code)
        if spec is None:
            continue
        category, label, missing, expected, provision, input_kind = spec
        guidance.append(CompatibilityInputGuidance(
            input_id=reason.code,
            category=category,
            label=label,
            missing=missing,
            expected_value=expected,
            provision=provision,
            input_kind=input_kind,
        ))
    return tuple(guidance)


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
        aliases = {
            "class_count": frozenset({"class_count", "class_count_evidence_unresolved"}),
        }
        limit_names = aliases.get(limit, frozenset({limit}) if limit is not None else frozenset())
        if limit_names.intersection(method.implementation_limits):
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
    result = CompatibilityResult(
        status=status, method=method.method, dataset=dataset.dataset,
        reasons=reasons, warnings=tuple(warnings),
        required_user_inputs=tuple(sorted(inputs)),
        required_input_paths=tuple(input_paths.items()),
    )
    return replace(result, input_guidance=build_input_guidance(result))


__all__ = [
    "CompatibilityInputGuidance", "CompatibilityReason", "CompatibilityResult",
    "CompatibilityStatus", "build_input_guidance",
    "ConfigInputRequirement", "MethodRequirements", "requirements_unavailable_result",
    "resolve_compatibility",
]
