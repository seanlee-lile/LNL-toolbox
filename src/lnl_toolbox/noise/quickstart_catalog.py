from __future__ import annotations

"""Human-facing noise capabilities for the Quick Start workflow.

The catalog is intentionally separate from reproduction recipes.  Its public
configuration output uses the existing experiment ``noise`` contract.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QuickStartNoiseSpec:
    key: str
    label: str
    description: str
    backend_name: str | None
    visible: bool
    requires_rate: bool
    requires_seed: bool
    category: str


_SPECS: tuple[QuickStartNoiseSpec, ...] = (
    QuickStartNoiseSpec("clean", "保持干净", "直接使用数据集提供的训练标签。", None, True, False, False, "clean"),
    QuickStartNoiseSpec("symmetric", "对称噪声", "按给定比例随机翻转到其他类别。", "symmetric", True, True, True, "synthetic"),
    QuickStartNoiseSpec("pairflip", "Pair-flip 噪声", "按类别顺序将标签翻转到下一个类别。", "pairflip", True, True, True, "synthetic"),
    QuickStartNoiseSpec("class_conditional", "类条件噪声", "使用配置的类别转移矩阵生成标签噪声。", "class_conditional", False, True, True, "synthetic"),
    QuickStartNoiseSpec("instance_dependent", "实例依赖噪声", "噪声概率由每个样本的输入特征决定。", "instance_dependent", False, True, True, "synthetic"),
    QuickStartNoiseSpec("pdl", "PDL 实例依赖噪声", "使用 PDL 论文规定的实例依赖噪声生成流程。", "pdl", True, True, True, "synthetic"),
    QuickStartNoiseSpec("external_torch", "外部标签文件", "由已有实验资源提供的外部标签映射。", "external_torch", False, False, False, "source_only"),
    QuickStartNoiseSpec("official_uniform_flip", "官方固定标签源", "特定论文或官方数据发布的固定标签来源。", "official_uniform_flip", False, True, True, "source_only"),
    QuickStartNoiseSpec("binary_asymmetric_rcn", "二分类非对称噪声", "二分类风险实验的非对称随机分类噪声。", "binary_asymmetric_rcn", False, True, True, "source_only"),
)
_BY_KEY = {item.key: item for item in _SPECS}


def quick_start_noise_specs() -> tuple[QuickStartNoiseSpec, ...]:
    return _SPECS


def visible_synthetic_noise_specs() -> tuple[QuickStartNoiseSpec, ...]:
    return tuple(item for item in _SPECS if item.visible and item.category == "synthetic")


def quick_start_noise_spec(key: str) -> QuickStartNoiseSpec:
    normalized = str(key).strip().lower().replace("-", "_")
    try:
        return _BY_KEY[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown Quick Start noise capability: {key!r}") from exc


def build_noise_config(
    key: str,
    *,
    rate: float | None,
    seed: int | None,
) -> dict[str, object] | None:
    """Build an existing experiment noise mapping, or ``None`` when inputs are insufficient."""

    spec = quick_start_noise_spec(key)
    if spec.key == "clean":
        return None
    if not spec.visible or spec.backend_name is None:
        raise ValueError(f"noise capability {spec.key!r} is not available in Quick Start")
    if spec.requires_rate and rate is None:
        raise ValueError(f"noise capability {spec.key!r} requires a noise rate")
    if rate is not None and not 0.0 <= float(rate) <= 1.0:
        raise ValueError("noise rate must be in [0, 1]")
    if spec.requires_seed and seed is None:
        raise ValueError(f"noise capability {spec.key!r} requires a random seed")

    config: dict[str, object] = {
        "name": spec.backend_name,
        "rate": float(rate) if rate is not None else None,
    }
    if seed is not None:
        config["seed"] = int(seed)
    return config


__all__ = [
    "QuickStartNoiseSpec",
    "build_noise_config",
    "quick_start_noise_spec",
    "quick_start_noise_specs",
    "visible_synthetic_noise_specs",
]
