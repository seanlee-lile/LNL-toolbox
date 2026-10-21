from __future__ import annotations

"""Human-facing noise capabilities for the Quick Start workflow.

The catalog is intentionally separate from reproduction recipes.  Its public
configuration output uses the existing experiment ``noise`` contract.
"""

from dataclasses import dataclass
import math
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
    QuickStartNoiseSpec("class_conditional", "类条件噪声", "使用类别转移矩阵指定每类标签的翻转概率；噪声率记录本次实验设置的翻转强度。", "class_conditional", True, True, True, "synthetic"),
    QuickStartNoiseSpec("instance_dependent", "实例依赖噪声", "噪声概率由每个样本的输入特征决定。", "instance_dependent", False, True, True, "synthetic"),
    QuickStartNoiseSpec("pdl", "PDL 实例依赖噪声", "使用 PDL 论文规定的实例依赖噪声生成流程。", "pdl", True, True, True, "synthetic"),
    QuickStartNoiseSpec("external_torch", "外部标签文件", "读取与当前训练样本顺序一致的干净标签和噪声标签文件。", "external_torch", True, False, False, "external"),
    QuickStartNoiseSpec("official_uniform_flip", "L2RW 标签翻转", "按 L2RW 原实现替换指定比例的训练标签，再打乱样本顺序；实际错误比例可能低于替换比例。", "official_uniform_flip", True, True, True, "synthetic"),
    QuickStartNoiseSpec("binary_asymmetric_rcn", "二分类非对称噪声", "分别设置正类与负类标签翻转概率。", "binary_asymmetric_rcn", False, False, True, "source_only"),
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
    rho_positive: float | None = None,
    rho_negative: float | None = None,
    transition_matrix: tuple[tuple[float, ...], ...] | None = None,
    path: str | None = None,
) -> dict[str, object] | None:
    """Build an existing experiment noise mapping, or ``None`` when inputs are insufficient."""

    spec = quick_start_noise_spec(key)
    if spec.key == "clean":
        return None
    if spec.key == "binary_asymmetric_rcn":
        if rho_positive is None or rho_negative is None or seed is None:
            raise ValueError("binary asymmetric noise requires both flip rates and a seed")
        positive, negative = float(rho_positive), float(rho_negative)
        if not 0.0 <= positive < 1.0 or not 0.0 <= negative < 1.0 or positive + negative >= 1.0:
            raise ValueError("binary flip rates must be in [0, 1) and sum to less than 1")
        return {"name": spec.backend_name, "rho_positive": positive, "rho_negative": negative, "seed": int(seed)}
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
    if spec.key == "class_conditional":
        if not transition_matrix:
            raise ValueError("类条件噪声需要类别转移矩阵")
        matrix = [[float(value) for value in row] for row in transition_matrix]
        if len(matrix) < 2 or any(len(row) != len(matrix) for row in matrix):
            raise ValueError("类别转移矩阵必须是至少 2 类的方阵")
        if any(not math.isfinite(value) or value < 0 for row in matrix for value in row) or any(not math.isclose(sum(row), 1.0, abs_tol=1e-8) for row in matrix):
            raise ValueError("类别转移矩阵各项须非负且有限，每行概率之和须为 1")
        config.update(transition_matrix=matrix, rng="numpy_legacy")
        if rate is None:
            config["rate"] = sum(1.0 - matrix[i][i] for i in range(len(matrix))) / len(matrix)
    if spec.key == "external_torch":
        if not path or not str(path).strip():
            raise ValueError("请选择与当前训练样本对齐的外部标签文件")
        if rate is None:
            config.pop("rate", None)
        config.update(path=str(path).strip(), clean_key="clean_label_train", noisy_key="noise_label_train")
    return config


__all__ = [
    "QuickStartNoiseSpec",
    "build_noise_config",
    "quick_start_noise_spec",
    "quick_start_noise_specs",
    "visible_synthetic_noise_specs",
]
