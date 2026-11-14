from __future__ import annotations
"""Shared parameter schema, permissions and YAML editing; no HTTP dependencies."""
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any
from lnl_toolbox.catalog import find_project_root
from lnl_toolbox.quickstart.templates import SPLIT_COUNT_PATHS, reference_train_size

ROOT = find_project_root(Path(__file__))
_QUICK_START_OVERRIDE_AUTH = object()

def _display_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)

def _project_path(value: object) -> Path:
    """Resolve a user path while keeping web writes inside this project."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("请输入 YAML 文件路径")
    candidate = Path(raw).expanduser()
    destination = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    try:
        destination.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("YAML 路径必须位于项目目录内") from exc
    if destination.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("YAML 文件必须使用 .yaml 或 .yml 扩展名")
    return destination


def _config_payload(
    recipe_id: object = None,
    *,
    path_value: object = None,
) -> dict[str, object]:
    from lnl_toolbox.catalog import load_yaml, recipe_by_id, validate_config

    if str(path_value or "").strip():
        path = _project_path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"YAML 文件不存在：{_display_path(path)}")
        config = load_yaml(path)
        runner = validate_config(config).name
        method_value = config.get("method", runner)
        method = (
            str(method_value.get("name", runner))
            if isinstance(method_value, dict)
            else str(method_value)
        )
        recipe_name = ""
    else:
        recipe = recipe_by_id(str(recipe_id), ROOT)
        path = recipe.config_path.resolve()
        config = load_yaml(path)
        runner = recipe.runner
        method = recipe.method
        recipe_name = recipe.id
    return {
        "recipe": recipe_name,
        "runner": runner,
        "method": method,
        "path": _display_path(path),
        "content": path.read_text(encoding="utf-8"),
        "config": config,
    }


def _builtin_config_paths() -> set[Path]:
    from lnl_toolbox.catalog import discover_recipes

    return {recipe.config_path.resolve() for recipe in discover_recipes(ROOT, include_conditional=True)}


_LEGACY_EDITABLE_SECTIONS = {
    "data",
    "noise",
    "loader",
    "model",
    "loss",
    "selector",
    "parameter_update",
    "optimizer",
    "scheduler",
    "trainer",
    "fine",
    "dividemix",
    "ca2c",
    "l2rw",
    "lend",
    "cnlcu",
    "coteaching",
    "dual_t",
    "dld",
    "pcse",
    "cal",
    "mc_ldce",
    "pdl",
    "upm",
    "volminnet",
    "t_revision",
    "importance_reweighting",
    "meta",
    "evidence",
}
_LEGACY_PROTECTED_LEAF_NAMES = {
    "method",
    "runner",
    "name",
    "type",
    "model",
    "loss",
    "selector",
    "parameter_update",
    "execution",
    "fidelity",
    "ensemble",
    "checkpoint_selection",
    "selection_split",
    "primary",
    "source",
}


def _legacy_editable_config_fields(
    config: object, *, prefix: str = ""
) -> list[dict[str, object]]:
    """Compatibility policy for non-paper recipes without registry metadata."""

    fields: list[dict[str, object]] = []
    if not isinstance(config, dict):
        return fields
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            if prefix or key in _LEGACY_EDITABLE_SECTIONS:
                fields.extend(_legacy_editable_config_fields(value, prefix=path))
            continue
        if not prefix and key not in {"seed", "output_root"}:
            continue
        root = path.split(".", 1)[0]
        if root not in _LEGACY_EDITABLE_SECTIONS and path not in {"seed", "output_root"}:
            continue
        if str(key) in _LEGACY_PROTECTED_LEAF_NAMES and path != "data.name":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            kind = "boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "text"
            fields.append(
                {
                    "path": path,
                    "label": path,
                    "value": value,
                    "kind": kind,
                    "level": "basic",
                    "editable": True,
                    "default_expanded": True,
                    "note": "通用配置参数。",
                }
            )
        elif isinstance(value, list):
            fields.append(
                {
                    "path": path,
                    "label": path,
                    "value": value,
                    "kind": "list",
                    "level": "basic",
                    "editable": True,
                    "default_expanded": True,
                    "note": "通用配置参数。",
                }
            )
    return fields


_PARAMETER_REGISTRY_PATH = Path(__file__).with_name("parameter_metadata.yaml")
_PARAMETER_LEVEL_ORDER = ("basic", "paper", "advanced", "locked")


@lru_cache(maxsize=1)
def _parameter_registry() -> dict[str, Any]:
    """Load and minimally validate the Web parameter authorization registry."""

    try:
        import yaml

        value = yaml.safe_load(_PARAMETER_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"参数元数据 registry 无法加载：{exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("methods"), dict):
        raise ValueError("参数元数据 registry 缺少 methods")
    if not isinstance(value.get("levels"), dict):
        raise ValueError("参数元数据 registry 缺少 levels")
    if not isinstance(value.get("formal_recipe_bindings"), dict):
        raise ValueError("参数元数据 registry 缺少 formal_recipe_bindings")
    return value


def _parameter_display_policy() -> dict[str, object]:
    policy = _parameter_registry().get("parameter_display_policy", {})
    return policy if isinstance(policy, dict) else {}


def _internal_seed_path(path: str) -> bool:
    return path != "seed" and path.rsplit(".", 1)[-1].endswith("seed")


def _parameter_presentation(path: str, method: str = "") -> tuple[str, str]:
    """Return the method-aware UI destination and optional display label.

    This is a presentation policy only.  Hidden fields remain available in the
    persisted recipe; common experiment controls, component selectors and
    resources are kept outside the method's default/advanced parameter groups.
    """

    policy = _parameter_display_policy()
    if method == "loss_correction" and path in {"pipeline.risk_corrector.name", "pipeline.warmup_epochs"}:
        return "parameter", ""
    if method == "loss_correction" and path in {"pipeline.transition_estimator.name", "pipeline.pretraining_protocol", "model.variant", "evaluation.selection_metric", "evaluation.corrected_loss"}:
        return "hidden", ""
    # Trusted-supervision wiring is resolved by the selected method/data
    # protocol. Raw trusted_validation.* keys are not meaningful controls in
    # the ordinary Web parameter editor.
    if path.startswith("trusted_validation."):
        return "hidden", ""
    if path in {
        "dld.feature_extractor.model.weights",
        "dld.feature_extractor.model.feature_dimension",
        "dld.feature_extractor.external.adapter",
        "dld.feature_extractor.external.weights",
        "dld.feature_extractor.external.input_contract",
    }:
        # The selected frozen extractor defines these implementation details.
        return "hidden", ""
    if method == "jocor" and path == "selector.keep_rate.end":
        return "hidden", ""
    if path in {
        "data.validation_split.source", "data.validation_split.rng",
        "data.validation_split.strategy", "evaluation.selection_split",
    }:
        return "hidden", ""
    # These caps only shrink a run for smoke/debugging.  A method's common
    # controls must not promote them into paper-facing parameters.
    if path in {
        "data.max_train_samples",
        "data.max_validation_samples",
        "data.max_test_samples",
    }:
        return "hidden", ""
    method_key = _method_registry_key(method)
    override = {
        "common_paths": policy.get("common_paths_by_method", {}).get(method_key, ()),
        "selection_paths": policy.get("selection_paths_by_method", {}).get(method_key, ()),
        "resource_paths": policy.get("resource_paths_by_method", {}).get(method_key, ()),
        "hidden_paths": policy.get("hidden_paths_by_method", {}).get(method_key, ()),
    }
    # Method-specific hard-screen destinations take precedence over generic
    # implementation fields (for example, a max-sample limit can be a
    # supported experiment control for one recipe and absent in another).
    if path in override.get("hidden_paths", ()):
        return "hidden", ""
    # The top-level experiment seed is the only user-facing seed control.
    # Component seeds are derived from it when the config is normalized.
    if _internal_seed_path(path):
        return "hidden", ""
    if path in override.get("selection_paths", ()):
        labels = policy.get("selection_labels", {})
        label = labels.get(path, "独立组件选择") if isinstance(labels, dict) else "独立组件选择"
        return "selection", str(label)
    if path in override.get("resource_paths", ()):
        labels = policy.get("resource_labels", {})
        label = labels.get(path, "外部资源/运行前置条件") if isinstance(labels, dict) else "外部资源/运行前置条件"
        return "resource", str(label)
    # Explicit common controls may override generic hidden prefixes, except
    # for the smoke-only sample caps handled above.
    if path in override.get("common_paths", ()):
        return "common", "公共实验设置"
    hidden_paths = policy.get("hidden_paths", ())
    hidden_prefixes = policy.get("hidden_prefixes", ())
    if path in hidden_paths or any(path.startswith(str(prefix)) for prefix in hidden_prefixes):
        return "hidden", ""
    resource_paths = policy.get("resource_paths", ())
    resource_prefixes = policy.get("resource_prefixes", ())
    if path in resource_paths or any(path.startswith(str(prefix)) for prefix in resource_prefixes):
        labels = policy.get("resource_labels", {})
        label = labels.get(path, "资源/前置条件") if isinstance(labels, dict) else "资源/前置条件"
        return "resource", str(label)
    return "parameter", ""


def _method_registry_key(method: object) -> str:
    return str(method or "").strip().lower().replace("-", "_")


def _config_method(config: object, fallback: object = "") -> str:
    if not isinstance(config, dict):
        return _method_registry_key(fallback)
    value = config.get("method", fallback)
    if isinstance(value, dict):
        value = value.get("name", fallback)
    method = _method_registry_key(value)
    registry = _parameter_registry()
    if method in registry["methods"]:
        return method
    record = config.get("parameter_record", {})
    meta = config.get("meta", {})
    web_record = meta.get("web_parameter_record", {}) if isinstance(meta, dict) else {}
    formal_recipe = (
        web_record.get("formal_recipe") if isinstance(web_record, dict) else ""
    ) or (record.get("formal_recipe") if isinstance(record, dict) else "")
    candidate_recipe = str(formal_recipe or fallback or "")
    # Historical dataset variants use the paper ID, not the executor/component
    # name (for example Loss Correction's component is called "forward").
    for suffix in ("-cifar10", "-cifar100"):
        if candidate_recipe.endswith(suffix):
            candidate_method = _method_registry_key(candidate_recipe[:-len(suffix)])
            if candidate_method in registry["methods"]:
                return candidate_method
    for candidate_method, recipe_id in registry["formal_recipe_bindings"].items():
        if candidate_recipe == str(recipe_id):
            return str(candidate_method)
    return method


def _config_path_value(config: object, path: str) -> tuple[bool, object]:
    value = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _value_kind(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "text"


def _metadata_field_kind(metadata: dict[str, object], current: object) -> tuple[str, bool]:
    """Return the editor kind and whether an empty value means ``null``.

    Registry metadata is authoritative here.  Inferring a field's type from
    its current value breaks nullable numeric settings because their current
    value is legitimately ``None`` (and previously exposed them as text).
    """

    value_type = str(metadata.get("value_type", "")).strip().lower()
    if value_type in {"integer", "number", "nullable"}:
        return "number", value_type == "nullable"
    if value_type == "boolean":
        return "boolean", False
    if value_type == "list":
        return "list", False
    if value_type in {"mapping", "object"}:
        return "object", False
    if value_type == "string":
        return "text", False
    return _value_kind(current), False


@lru_cache(maxsize=1)
def _registry_recipe_paths() -> dict[str, Path]:
    from lnl_toolbox.catalog import discover_recipes

    return {
        recipe.id: recipe.config_path
        for recipe in discover_recipes(ROOT, include_conditional=True)
    }


@lru_cache(maxsize=None)
def _formal_registry_config(method: str) -> tuple[str, dict[str, Any]] | None:
    registry = _parameter_registry()
    recipe_id = registry["formal_recipe_bindings"].get(method)
    if not recipe_id:
        return None
    from lnl_toolbox.catalog import load_yaml

    try:
        recipe_path = _registry_recipe_paths()[str(recipe_id)]
    except KeyError as exc:
        raise ValueError(f"registry formal recipe 不存在：{recipe_id}") from exc
    config = load_yaml(recipe_path)
    if method == "jocor":
        from lnl_toolbox.core.config_schema import normalize_experiment_config

        config = normalize_experiment_config(config)
    return str(recipe_id), config


def _research_parameter_category(
    group: str, path: str, view: dict[str, Any]
) -> str:
    """Classify a visible config field without changing its edit permission."""

    if not group:
        return ""
    if path in view.get("implementation_paths", ()):
        return "runtime"
    if group == "method":
        if path in view.get("method_protocol_paths", ()) or path.endswith(
            tuple(view.get("method_protocol_suffixes", ()))
        ):
            return "method_protocol"
        return "method"
    if group == "data":
        return "data" if path in view.get("focus_data_paths", ()) else "data_protocol"
    if group == "training":
        return (
            "comparison"
            if path in view.get("comparison_paths", ())
            else "training_protocol"
        )
    raise ValueError(f"unknown research parameter group: {group}")


def _parameter_display_group(field: dict[str, object], method: str) -> str:
    """Classify visible fields without changing registry edit permissions."""

    if not field.get("editable"):
        return "restricted"
    if field.get("model_editor"):
        return "default"
    if tuple(str(field.get("path", "")).split(".")) in SPLIT_COUNT_PATHS:
        return "default"
    if (
        field.get("kind") in {"list", "object"}
        or isinstance(field.get("value"), (list, dict))
    ):
        return "advanced"
    policy = _parameter_registry().get("default_parameter_paths", {})
    if not isinstance(policy, dict):
        return "advanced"
    methods = policy.get("methods", {})
    method_paths = methods.get(_method_registry_key(method), ()) if isinstance(methods, dict) else ()
    exclusions = policy.get("exclude", {})
    excluded = exclusions.get(_method_registry_key(method), ()) if isinstance(exclusions, dict) else ()
    selected = (set(policy.get("common", ())) | set(method_paths)) - set(excluded)
    if field.get("presentation") == "hidden":
        return "advanced"
    path = str(field.get("path", ""))
    if path in selected:
        return "default"
    if path in {
        "data.validation_size", "data.augment", "data.strong_augment",
        "data.preprocessing", "data.folds", "data.fold_index",
        "trainer.max_steps", "noise.path", "dld.feature_extractor.source",
        "pipeline.weight_provider.artifact_path", "posterior_stage.rate_estimator",
        "selector.keep_rate.start", "selector.keep_rate.end",
        "algorithm.estimated_noise_rate",
    }:
        return "default"
    if path == "model.name" or path.endswith(".model.name"):
        return "default"
    if ".scheduler." in path or path.startswith("scheduler."):
        return "default"
    if path.endswith((".epochs", ".lr", ".learning_rate")) and not any(
        part in path for part in (".gmm.", ".loss_history.")
    ):
        return "default"
    return "advanced"


def _parameter_context(path: str, method: str = "", *, brief: bool = False) -> str:
    """Name the operation/model, not an opaque stage number."""
    contexts = {
        "t_revision.stage1": ("初始分类器", "用带噪标签训练初始分类器，供初始转移矩阵估计使用"),
        "t_revision.classifier_initialization": ("校正分类器", "固定估计的转移矩阵，使用校正损失训练分类器"),
        "t_revision.revision": ("联合修正", "联合更新分类器和转移矩阵的修正量"),
        "upm.stage1": ("初始分类器", "用带噪标签训练初始分类器，生成观测标签后验概率"),
        "upm.main": ("软标签分类器", "使用估计的软标签训练分类器"),
        "upm.confusing_probability": ("混淆概率", "样本混淆概率更新"),
        "dld.diffusion.optimizer.direction": ("方向预测器", "扩散训练中的方向预测器"),
        "dld.diffusion.scheduler.direction": ("方向预测器", "扩散训练中的方向预测器"),
        "dld.diffusion.optimizer.noise": ("噪声预测器", "扩散训练中的噪声预测器"),
        "dld.diffusion.scheduler.noise": ("噪声预测器", "扩散训练中的噪声预测器"),
        "dld.diffusion": ("扩散训练", "标签扩散模型训练"),
        "dld.feature_extractor": ("特征提取器", "冻结的预训练特征提取器"),
        "dld.precorrection": ("标签预校正", "基于特征近邻的标签预校正"),
        "volminnet.optimizer.classifier": ("分类器", "分类器训练"),
        "volminnet.scheduler.classifier": ("分类器", "分类器训练"),
        "volminnet.optimizer.transition": ("转移矩阵", "转移矩阵学习"),
        "volminnet.scheduler.transition": ("转移矩阵", "转移矩阵学习"),
        "posterior_stage": ("初始模型", "带噪标签后验概率模型训练"),
        "final_stage": ("最终模型", "噪声校正后的分类器训练"),
        "pretraining_stage": ("特征预训练", "特征模型预训练"),
        "transition_stage": ("噪声矩阵估计", "噪声转移矩阵学习"),
        "ensemble_stage": ("集成训练", "集成分类器训练"),
        "dividemix.warmup": ("双模型预热", "两个分类器的带噪标签预热训练"),
        "dividemix.training": ("协同训练", "两个分类器的 DivideMix 协同训练"),
        "warmup": ("预热", "带噪标签预热训练"),
        "transition": ("噪声矩阵估计", "噪声转移矩阵学习"),
        "revision": ("联合修正", "分类器与实例相关转移矩阵联合修正"),
        "phases.correction_epochs": ("校正分类器", "固定实例相关转移矩阵的分类器训练"),
        "phases.revision_epochs": ("联合修正", "分类器与实例相关转移矩阵联合修正"),
        "lend.training": ("主模型", "LEND 分类器训练"),
        "fine.warmup": ("预热", "FINE 分类器预热训练"),
        "ca2c.warmup": ("预热", "CA2C 分类器预热训练"),
    }
    for prefix in sorted(contexts, key=len, reverse=True):
        if path == prefix or path.startswith(prefix + ".") or (
            prefix.endswith(".warmup") and path.startswith(prefix + "_")
        ):
            return contexts[prefix][0 if brief else 1]
    if path.startswith(("optimizer.", "scheduler.", "model.")):
        peers = method in {"coteaching", "cnlcu", "jocor", "dividemix"}
        return ("双模型共同设置" if peers else "主模型") if brief else ("两个同伴分类器的共同设置" if peers else "主分类器训练")
    return ""


def _parameter_user_note(path: str, original: str, method: str = "") -> str:
    if method == "loss_correction" and path == "pipeline.warmup_epochs":
        return "估计 T 前的 CE 训练轮数。按带噪验证损失选择最佳 CE 模型，再用训练集和验证集预测估计 T。CIFAR-10 默认 120 轮。"
    if method == "loss_correction" and path == "pipeline.transition_estimator.name":
        return "known 使用提供的转移矩阵；anchor 从观测标签数据估计，不需要真实 T 或干净标签。"
    spec = _parameter_ui_spec(path)
    if spec:
        context = _parameter_context(path, method)
        return (context + "：" if context else "") + str(spec[1])
    context = _parameter_context(path, method)
    note = _parameter_base_note(path, original)
    if context:
        note = note.replace("本阶段", "").replace("该训练阶段", "").replace("主模型", "")
        return context + "：" + note if note else context + "。"
    return note


def _parameter_base_note(path: str, original: str) -> str:
    """Keep field help specific to that field, without registry policy boilerplate."""

    if path == "loader.batch_size":
        return "每次参数更新使用的训练样本数。"
    if path == "trainer.epochs":
        return "训练数据最多完整遍历的次数。"
    if path == "optimizer.lr":
        return "主模型优化器的起始学习率。"
    if path == "meta.virtual_learning_rate":
        return "L2RW 的 official 实现固定为 1；paper 实现才使用此值计算虚拟更新。"
    if path == "data.num_clean":
        return "用于计算样本权重的干净验证样本数。"
    if path == "data.num_val":
        return "从训练数据划出的验证样本数。"
    if path == "data.dimension":
        return "每个样本的特征维度；模型输入维度会随之更新。"
    if path == "noise.rate":
        return "生成噪声时目标标签翻转比例。"
    if path in {"noise.rho_positive", "noise.rho_negative"}:
        return ("正类标签翻转为负类" if path.endswith("positive") else "负类标签翻转为正类") + "的概率；两类概率之和必须小于 1。"
    if path in {"noise.transition_matrix", "pipeline.transition_estimator.matrix"}:
        return "类别间标签转移概率矩阵；每行对应一个原始类别。"
    if path == "trainer.max_steps":
        return "训练最多执行的参数更新次数。"
    if path == "data.augment":
        return "是否对训练输入应用随机增强。"
    if path == "data.strong_augment":
        return "是否生成强增强训练视图。"
    if path == "data.preprocessing":
        return "输入图像的转换和归一化方案。"
    if path == "data.validation_size":
        return "从训练数据中划出的验证样本数。"
    if path in {"data.train_size", "data.test_size"}:
        return "生成的" + ("训练" if path == "data.train_size" else "测试") + "样本数；该数据协议要求正偶数。"
    if path in {"data.split_strategy", "data.validation_split.strategy"}:
        return "训练与验证样本的划分规则。"
    if path == "data.folds":
        return "交叉验证划分的份数。"
    if path == "data.fold_index":
        return "本次使用的交叉验证份编号，从 0 开始。"
    if path == "noise.path":
        return "外部观测标签或噪声文件的路径。"
    if path.endswith(".model.name") or path == "model.name":
        return "本阶段使用的模型网络架构。"
    if path.endswith(".scheduler.name") or path == "scheduler.name":
        return "本阶段的学习率变化方式。"
    if path.endswith(".t_max") and "scheduler" in path:
        return "余弦学习率下降曲线的周期长度。"
    if path.endswith(".eta_min") and "scheduler" in path:
        return "余弦学习率曲线的最低学习率。"
    if path.endswith(".start_epoch") and "scheduler" in path:
        return "学习率开始下降的轮次。"
    if path.endswith(".end_epoch") and "scheduler" in path:
        return "学习率下降到终点值的轮次。"
    if path.endswith(".step_milestones"):
        return "学习率变化的累计更新步数节点。"
    if path.endswith(".milestones") and "scheduler" in path:
        return "学习率变化的训练轮次节点。"
    if path.endswith(".gamma") and "scheduler" in path:
        return "每个学习率节点使用的衰减倍率。"
    if path.endswith(".epochs"):
        return "该训练阶段遍历数据的次数。"
    if path in {"phases.correction_epochs", "phases.revision_epochs"}:
        return "完整遍历训练数据的次数。"
    if path.endswith(".warmup_epochs"):
        return "预热阶段遍历数据的次数。"
    if path.endswith((".lr", ".learning_rate")):
        return "该训练阶段优化器的起始学习率。"
    if path.endswith(".batch_size"):
        return "每批从该数据角色取出的样本数。"
    if path == "posterior_stage.rate_estimator":
        return "后验阶段的噪声率估计方法。"
    if path == "dld.feature_extractor.source":
        return "DLD 特征提取器的来源。"
    if path == "pipeline.weight_provider.artifact_path":
        return "MentorNet 预训练权重文件路径。"
    if path == "pipeline.weight_provider.decay":
        return "MentorNet 权重提供器使用的衰减系数。"
    if path == "seed":
        return "控制本次实验随机过程的种子。"
    if path in {"data.name", "noise.name", "optimizer.name"} or path.endswith(".optimizer.name"):
        return {"data.name": "训练使用的数据集。", "noise.name": "标签噪声类型。"}.get(path, "优化器类型。")
    if path.endswith(".momentum"):
        return "优化器的动量系数。"
    if path.endswith(".weight_decay"):
        return "优化器的权重衰减系数。"
    if path.endswith(".nesterov"):
        return "是否启用 Nesterov 动量。"
    if path.endswith(".model.base_width") or path == "model.base_width":
        return "模型的基础通道数。"
    details = {
        "optimizer.beta1": "Adam 一阶梯度移动平均的衰减系数。",
        "optimizer.beta2": "Adam 二阶梯度平方移动平均的衰减系数。",
        "optimizer.betas": "Adam 的一阶、二阶移动平均衰减系数，按此顺序填写。",
        "scheduler.lr_values": "按节点顺序填写各节点开始使用的学习率。",
        "scheduler.initial_lr": "学习率变化曲线的起始值。",
        "scheduler.final_lr": "学习率变化曲线的终点值。",
        "model.dropout": "训练时随机置零神经元输出的比例。",
        "model.width_multiplier": "网络通道宽度相对基础宽度的倍数。",
        "model.num_residual_units": "每组残差模块中的残差单元数。",
        "model.leakiness": "LeakyReLU 在负输入区间的斜率。",
        "model.bias": "是否在模型中使用偏置参数。",
        "loader.num_workers": "并行读取训练数据的工作进程数。",
        "loader.drop_last": "是否舍弃最后一个不足批次大小的训练批次。",
        "loader.pin_memory": "是否使用锁页内存加速数据到 GPU 的传输。",
        "early_stopping.monitor": "用于判断是否提前停止训练的指标名称。",
        "early_stopping.mode": "被监测指标的改善方向：min 表示降低，max 表示升高。",
        "early_stopping.patience": "指标连续未改善多少轮后停止训练。",
        "early_stopping.min_delta": "将指标变化认定为改善所需的最小幅度。",
        "parameter_update.max_grad_norm": "梯度裁剪的最大范数；超过此值时缩放梯度。",
        "noise.sampling": "依据转移概率生成观测标签时使用的抽样方式。",
        "noise.mode": "选择标签噪声的生成或读取方式。",
        "data.split_rng": "数据划分使用的随机数生成器类型。",
        "data.validation_split.rng": "验证集划分使用的随机数生成器类型。",
        "data.normalization.mean": "输入归一化时各颜色通道减去的均值，按通道顺序填写。",
        "data.normalization.std": "输入归一化时各颜色通道除以的标准差，按通道顺序填写。",
        "data.strong_policy": "生成强增强图像视图的变换策略。",
        "data.strong_magnitude": "强数据增强操作的强度。",
        "evaluation.selection_split": "用于选择最佳模型检查点的数据划分。",
        "evaluation.allow_test_selection": "是否允许用测试集指标选择模型检查点。",
        "evaluation.loss.name": "评估时计算损失使用的函数。",
        "evaluation.primary": "结果汇总使用的主要评估指标。",
        "evaluation.report_last_epochs": "结果汇总统计最后多少个训练轮次。",
        "cwd.variant": "CWD 使用的算法变体。",
        "cwd.dynamic_centroid": "是否依据当前批次的特征和观测标签重新计算类别质心。",
        "feature_stage.layers": "PCSE 提取特征时使用的网络层名称列表。",
        "parameter_update.critical_scope": "CDR 筛选关键参数时考察的模型参数范围。",
        "parameter_update.compatibility_mode": "CDR 参数更新采用 paper 公式版本还是 official_code 实现版本。",
        "pipeline.objective_consumer.mda": "是否启用 DSS 的边际分布对齐（MDA）。",
        "pipeline.objective_consumer.ccs": "是否启用 DSS 的 CCS 筛选：根据跨轮次预测变化趋势排除类别。",
        "selector.keep_rate.name": "JoCoR 保留样本比例随训练进度变化的规则。",
        "models": "JoCoR 两个同伴分类器的模型配置列表。",
        "cnlcu.variant": "CNLCU 样本不确定性估计采用的算法变体。",
        "parameter_update.milestones": "参数更新策略切换的训练轮次节点。",
        "parameter_update.gamma": "参数更新策略在节点使用的衰减倍率。",
        "pipeline.weight_provider.total_epochs": "MentorNet 计算训练进度百分比时使用的总轮数。",
        "pipeline.weight_provider.fixed_epoch_after_burn_in": "预热结束后，是否固定输入 MentorNet 的进度特征。",
        "pipeline.weight_provider.fixed_label": "输入 MentorNet 的固定标签特征；为空时使用各样本的观测标签。",
        "mc_ldce.transition_model": "MC-LDCE 估计噪声转移矩阵使用的模型方式。",
        "mc_ldce.feature_mode": "MC-LDCE 分类阶段采用固定特征或可训练特征的方式。",
        "mc_ldce.classifier_bias": "是否为 MC-LDCE 线性分类器启用偏置项。",
        "algorithm.correction": "PDL 使用的标签噪声损失校正方式。",
        "dividemix.gmm.loss_history.name": "DivideMix 拟合干净/带噪损失分布时采用的历史损失汇总规则。",
        "dividemix.gmm.loss_history.high_noise_rate": "达到此噪声率时启用高噪声条件下的历史损失汇总。",
        "dividemix.gmm.loss_history.window_epochs": "DivideMix 汇总损失时使用的最近训练轮数。",
        "lend.graph.metric": "LEND 构建特征近邻图时使用的距离度量。",
        "lend.graph.normalize_features": "是否在构建 LEND 近邻图前对特征向量归一化。",
    }
    for suffix, note in details.items():
        if path == suffix or path.endswith("." + suffix):
            return note
    if path in {"data.max_train_samples", "data.max_validation_samples", "data.max_test_samples"}:
        return "本次运行最多使用的" + {"data.max_train_samples": "训练", "data.max_validation_samples": "验证", "data.max_test_samples": "测试"}[path] + "样本数。"
    if original in {
        "可修改的高级实验/实现选项；改变它可能改变数据协议、模型、优化器、fidelity/variant 或评估细节，但“偏离论文”本身不是锁定理由。",
        "优化器或学习率调度细节；影响数值复现，但不定义算法身份。",
        "可修改的重要论文参数或实验条件；修改后实验仍可运行，但应标记为偏离当前论文/正式 recipe 设置。",
        "工程实现或估计器内部细节；默认折叠。",
        "结构性运行契约；普通参数面板中的独立修改会破坏 schema、runner、组件 wiring、label-space/cardinality 或 lifecycle 契约。",
    }:
        return ""
    return original


def _parameter_ui_spec(path: str) -> list[str] | None:
    if path == "model.weights" or path.endswith(".model.weights"):
        return [path, "模型初始权重来源。none 表示随机初始化；ImageNet 权重加载后会按当前类别数重新建立分类头。"]
    if "scheduler" in path.split(".") and path.endswith(".step_size"):
        return [path, "每经过多少个训练轮次，将学习率乘以衰减倍率。"]
    if path.endswith((".early_stopping", ".warmup_early_stopping")):
        return [path, "启用后，验证指标连续15轮没有改善时结束训练，使用保存的最佳模型继续后续流程；关闭后按设置的训练预算运行。"]
    if path == "algorithm.estimated_noise_rate":
        return ["algorithm.estimated_noise_rate", "预计训练标签中错误标签的比例，用于确定样本筛选的最终保留比例（1−此值）。人工加噪时自动使用所设噪声率；干净数据为 0；原生带噪且噪声率未知时需要填写。修改此值不会改变标签或加噪率。"]
    ui = _parameter_registry().get("parameter_ui", {})
    exact = ui.get("fields", {}).get(path)
    if exact:
        return exact
    # Optimizers/schedulers may be named by their role (classifier, direction…).
    parts = path.split(".")
    normalized = path
    for component in ("optimizer", "scheduler", "model"):
        if component in parts:
            normalized = component + "." + parts[-1]
            break
    suffixes = ui.get("suffixes", {})
    for suffix in sorted(suffixes, key=len, reverse=True):
        if path == suffix or path.endswith("." + suffix) or normalized == suffix:
            return suffixes[suffix]
    return None


def _parameter_choices(field: dict[str, object], method: str) -> list[dict[str, str]] | None:
    if field.get("kind") not in {"string", "text"}:
        return None
    if field.get("visible") is False:
        return None
    path = str(field["path"])
    if path == "noise.manifest" or path.endswith((".path", ".artifact_path")):
        return None  # A user-supplied filesystem path is not an enumeration.
    ui = _parameter_registry().get("parameter_ui", {})
    choices = ui.get("choices", {}).get(path)
    if method == "loss_correction" and path == "pipeline.transition_estimator.name":
        choices = ["known", "anchor"]
    if method == "loss_correction" and path == "pipeline.risk_corrector.name":
        choices = ["forward", "backward"]
    parts = path.split(".")
    category = next((part for part in ("optimizer", "scheduler", "model", "loss") if part in parts), "")
    if category == "model" and parts[-1] in {"variant", "weights"} and "model_name" in field:
        model_name = str(field["model_name"]).lower()
        model_variant = str(field.get("model_variant", "standard")).lower()
        if parts[-1] == "variant":
            choices = ["standard"]
            if model_name == "preact_resnet18":
                choices.append("dss_specific")
            elif model_name in {"resnet32", "resnet44"}:
                choices.append("loss_correction_specific")
            elif model_name in {"resnet18", "resnet34", "resnet50", "resnet101"} and not (method == "cwd" and path == "model.variant"):
                choices.append("imagenet_standard")
            elif model_name == "torchvision_resnet34":
                choices = ["imagenet_standard"]
        else:
            choices = ["none"]
            if model_name == "torchvision_resnet34":
                choices = ["IMAGENET1K_V1"]
            elif model_variant == "imagenet_standard" and model_name in {"resnet18", "resnet34", "resnet50", "resnet101"} and not (method == "cwd" and path == "model.weights"):
                choices.append("imagenet1k_v1")
                if model_name in {"resnet50", "resnet101"}:
                    choices.append("imagenet1k_v2")
    if choices is None and parts[-1] == "name" and category:
        choices = ui["choices"].get(category + ".name")
    if category == "model" and parts[-1] == "name":
        # Use the vocabulary of the actual model factory for this method,
        # rather than offering every model mentioned anywhere in the toolbox.
        choices = ["tiny_cnn", "cnn8", "cnn9", "resnet14", "resnet32",
                   "resnet18", "resnet34", "resnet50", "resnet101",
                   "preact_resnet18", "mentor_wide_resnet", "feature_mlp"]
        if method in {"apl", "gce", "cdr", "dss", "mentornet", "loss_correction", "pdl"}:
            choices = [value for value in choices if value not in {"cnn9", "feature_mlp"}]
        extra = {"jocor": "cifar_six_conv", "cnlcu": "cifar_six_conv", "fine": "fine_seven_cnn",
                 "ca2c": "ca2c_seven_cnn", "l2rw": "l2rw_resnet32", "mc_ldce": "mc_ldce_cnn"}.get(method)
        if extra:
            choices.append(extra)
        if method == "l2rw":
            choices.append("wide_resnet28_10")
        if method == "loss_correction":
            choices.append("resnet44")
        if method == "importance_reweighting":
            choices = ["linear", "resnet18", "resnet34", "cnn8"]
        elif method == "pcse":
            choices = ["pcse_mlp"] if field.get("value") == "pcse_mlp" else ["resnet18", "resnet34", "resnet50", "resnet101"]
        elif method == "dld" and field.get("value") == "torchvision_resnet34":
            choices = ["torchvision_resnet34"]
    if path == "data.name":
        from lnl_toolbox.training.data_service import DATASETS
        choices = list(DATASETS.names())
        current = str(field.get("value", ""))
        if current not in choices:
            DATASETS.get(current)  # Keep supported aliases, not arbitrary strings.
            choices.append(current)
    if category == "optimizer" and method in {"binary_risk", "cwd"} and path == "optimizer.name":
        choices = ["sgd"] if method == "binary_risk" else ["adam"]
    if category == "model" and path == "model.name":
        if method == "binary_risk":
            choices = ["linear", "mlp"]
        elif method == "cwd":
            choices = ["resnet34", "cifar_resnet34", "tiny_cnn", "feature_mlp"]
    if category == "scheduler":
        if method == "fine":
            choices = ["cosine"]
        elif method == "jocor":
            choices = ["none", "linear_decay"]
        elif method == "cal":
            choices = ["none", "step", "multistep"]
        elif method in {"dld", "volminnet"}:
            choices = ["none", "cosine", "multistep"]
        elif method == "mc_ldce" and path == "transition.scheduler.name":
            choices = ["none", "multistep"]
        elif method in {"apl", "gce", "cdr", "dss", "mentornet", "loss_correction", "pdl"}:
            # The supervised configuration contract excludes linear_after.
            choices = ["none", "cosine", "multistep"]
    if path == "evaluation.primary":
        # Dedicated evaluators expose their own metric, not an interchangeable
        # menu of metrics that may never be produced by the selected runner.
        choices = [str(field["value"])]
    if path == "transition.estimator" and field.get("value") != "known_smoke":
        choices = [value for value in choices if value != "known_smoke"]
    if not field.get("editable"):
        choices = [str(field.get("value", ""))]
    if choices is None:
        return None
    result = []
    for value in choices:
        value = str(value)
        description = ui.get("choice_descriptions_by_path", {}).get(path, {}).get(
            value, ui.get("choice_descriptions", {}).get(value, "")
        )
        if category == "model":
            description = ui.get("model_descriptions", {}).get(value, description)
        if path == "data.name":
            description = ui.get("dataset_descriptions", {}).get(value, description)
        result.append({"value": value, "label": value, "description": description})
    return result


@lru_cache(maxsize=32)
def _feature_layer_choices(model_json: str, num_classes: int) -> list[dict[str, str]]:
    import torch
    from lnl_toolbox.training.experiment import build_model

    # Read architecture names without loading weights, allocating tensors or
    # changing the experiment's random state.
    model_config = json.loads(model_json)
    if model_config.get("name") == "pcse_mlp":
        return [{"value": "hidden1", "label": "hidden1", "description": "隐藏层一"}, {"value": "hidden2", "label": "hidden2", "description": "隐藏层二"}]
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        model = build_model(model_config, num_classes)
    names = [name for name, _ in model.named_children()
             if name not in {"fc", "classifier", "head", "output", "logits"}]
    descriptions = {"layer1": "残差组一", "layer2": "残差组二", "layer3": "残差组三",
                    "layer4": "残差组四", "features": "卷积特征层", "avgpool": "平均池化层",
                    "conv1": "首层卷积", "bn1": "首层批归一化", "relu": "首层激活", "maxpool": "最大池化层"}
    return [{"value": name, "label": name, "description": descriptions.get(name, "网络中间层")} for name in names]


def _parameter_list_rows(field: dict[str, object], config: dict[str, Any], method: str) -> list[list[dict[str, object]]] | None:
    path = field["path"]
    if path not in {"models", "feature_stage.layers"}:
        return None
    rows = []
    for index, item in enumerate(field.get("value") or []):
        row = []
        for key, value in item.items():
            child_path = "model." + key if path == "models" else "feature_layer." + key
            spec = _parameter_ui_spec(child_path)
            if spec is None:
                raise ValueError(f"列表参数缺少说明：{path}.{key}")
            child = {"path": child_path, "key": key, "index": index, "label": key,
                     "note": spec[1], "value": value, "kind": "boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "text",
                     "editable": field["editable"], "visible": True}
            if path == "models":
                child["choices"] = _parameter_choices(child, method)
            elif key == "pooling":
                child["choices"] = [{"value": "global_average", "label": "global_average", "description": "各空间位置取平均"}, {"value": "flatten", "label": "flatten", "description": "全部位置展开成向量"}]
            elif key == "name":
                model = config.get("pretraining_stage", {}).get("model", {})
                child["choices"] = _feature_layer_choices(json.dumps(model, sort_keys=True), int(config["data"]["num_classes"]))
            row.append(child)
        rows.append(row)
    return rows


def _parameter_user_label(path: str, method: str) -> str:
    labels = {
        "seed": "实验随机种子", "data.augment": "训练数据增强",
        "data.strong_augment": "强数据增强", "data.preprocessing": "输入预处理",
        "data.validation_size": "验证集样本数", "data.folds": "交叉验证份数",
        "data.fold_index": "当前验证份", "trainer.max_steps": "总更新步数",
        "trainer.epochs": "训练轮数", "loader.batch_size": "训练批次大小",
        "meta.virtual_learning_rate": "虚拟更新步长", "data.num_clean": "干净验证样本数",
        "noise.path": "外部噪声/标签文件", "dld.feature_extractor.source": "DLD 特征来源",
        "pipeline.weight_provider.artifact_path": "MentorNet 权重文件",
    }
    context = _parameter_context(path, method, brief=True)
    if context:
        return path + "（" + context + "）"
    if path in labels:
        return path + "（" + labels[path] + "）"
    if path == "model.name" or path.endswith(".model.name"):
        return path + "（模型架构）"
    return path


def _registry_config_fields(
    config: dict[str, Any], method: str, *, allow_missing: bool = False
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    if method == "jocor":
        from lnl_toolbox.core.config_schema import normalize_experiment_config

        config = normalize_experiment_config(config)
    registry = _parameter_registry()
    method_metadata = registry["methods"].get(method)
    if not isinstance(method_metadata, dict):
        return None
    parameters = method_metadata.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"registry method {method} 缺少 parameters")
    levels = registry["levels"]
    formal = _formal_registry_config(method)
    recipe_id, baseline = formal if formal is not None else ("", config)
    paper_metadata = method_metadata.get("paper", {})
    evidence = paper_metadata.get("evidence", {}) if isinstance(paper_metadata, dict) else {}
    research_view = registry.get("research_parameter_view", {})
    research_groups = research_view.get("groups", {})
    research_method_paths = set(research_view.get("method_paths", {}).get(method, ()))
    research_data_paths = set(research_view.get("data_paths", ()))
    research_training_paths = set(research_view.get("training_paths", ()))
    research_categories = research_view.get("categories", {})
    fields: list[dict[str, object]] = []
    for level in _PARAMETER_LEVEL_ORDER:
        level_metadata = levels.get(level, {})
        entries = parameters.get(level, {})
        if not isinstance(entries, dict):
            raise ValueError(f"registry method {method}.{level} 必须是 mapping")
        for path, metadata in entries.items():
            if not isinstance(metadata, dict):
                raise ValueError(f"registry 参数 {method}.{path} 元数据无效")
            presentation, resource_label = _parameter_presentation(str(path), method)
            found, current = _config_path_value(config, str(path))
            if not found:
                if method == "importance_reweighting" and str(path).startswith("posterior_stage."):
                    # Neural stage controls are absent from KDE/KLIEP recipes;
                    # density bandwidth is absent from neural recipes.
                    continue
                if method == "loss_correction" and str(path) in {"pipeline.transition_estimator.matrix", "pipeline.transition_estimator.percentile", "model.variant"}:
                    continue
                if allow_missing:
                    continue
                raise ValueError(f"registry 参数路径不存在：{method}.{path}")
            baseline_found, baseline_value = _config_path_value(baseline, str(path))
            evidence_ref = metadata.get("evidence_ref")
            kind, nullable = _metadata_field_kind(metadata, current)
            permission, runtime_lock_reason, _ = _parameter_edit_rule(
                config, method, str(path), level
            )
            research_group = ""
            # Research-facing groups describe what the ordinary parameter
            # panel exposes. Structural/hidden fields and external resources
            # must not masquerade as method/data/training variables.
            if presentation == "parameter" and level != "locked":
                if path in research_method_paths:
                    research_group = "method"
                elif path in research_data_paths:
                    research_group = "data"
                elif level == "paper":
                    research_group = "method"
                elif path in research_training_paths or str(path).endswith(
                    (".optimizer.lr", ".model.name", ".scheduler.milestones", ".scheduler.gamma")
                ):
                    research_group = "training"
            research_category = (
                _research_parameter_category(research_group, str(path), research_view)
                if presentation == "parameter"
                else "resource" if presentation == "resource" else ""
            )
            fields.append(
                {
                    "path": str(path),
                    "label": _parameter_user_label(str(path), method),
                    "value": current,
                    "kind": kind,
                    "nullable": nullable,
                    "number_type": str(metadata.get("value_type", ""))
                    if kind == "number"
                    else "",
                    "level": level,
                    # Show only named algorithm/component contracts as
                    # read-only. Provenance and implementation-only locked
                    # fields (including extra seeds) stay hidden.
                    "visible": presentation != "hidden" or (
                        level == "locked" and str(path).endswith(".name")
                    ),
                    "presentation": presentation,
                    "presentation_label": resource_label,
                    "research_group": research_group,
                    "research_category": research_category,
                    "level_label": level_metadata.get("label_zh", level),
                    "editable": permission == "editable" and not _internal_seed_path(str(path)),
                    "default_expanded": bool(
                        level_metadata.get("default_expanded", False)
                    ),
                    "note": _parameter_user_note(str(path), str(metadata.get("note", "")), method),
                    "reproduction_impact": str(
                        metadata.get("reproduction_impact", "")
                    ),
                    "origin": str(metadata.get("origin", "")),
                    "lock_reason": runtime_lock_reason or str(metadata.get("lock_reason", "")),
                    "evidence_ref": str(evidence_ref or ""),
                    "evidence": str(evidence.get(evidence_ref, "")),
                    "linked_fields": list(metadata.get("linked_fields", ())),
                    "changed_from": baseline_value if baseline_found else None,
                    "changed_from_paper": bool(
                        level == "paper"
                        and baseline_found
                        and current != baseline_value
                    ),
                }
            )
    if method in {"binary_risk", "cwd"}:
        # The binary runner's ordinary hold-out is optional, not a method hyperparameter.
        fields.append({
            "path": "data.validation_size", "label": _parameter_user_label("data.validation_size", method),
            "value": config.get("data", {}).get("validation_size", 0),
            "kind": "number", "number_type": "integer", "nullable": False,
            "level": "advanced", "visible": True, "editable": True,
            "presentation": "common", "research_group": "", "research_category": "",
            "linked_fields": [], "note": _parameter_user_note("data.validation_size", "", method),
        })
    if allow_missing:
        known = {str(field["path"]) for field in fields}
        for field in _legacy_editable_config_fields(config):
            path = str(field["path"])
            if path not in known:
                presentation, presentation_label = _parameter_presentation(path, method)
                fields.append({
                    **field, "level": "advanced", "visible": presentation != "hidden",
                    "editable": bool(field["editable"]) and not _internal_seed_path(path),
                    "presentation": presentation, "presentation_label": presentation_label,
                    "research_group": "", "research_category": "", "linked_fields": [],
                })
    paper_changes = [
        field for field in fields if field["level"] == "paper" and field["changed_from_paper"]
    ]
    if method == "loss_correction":
        estimated = (config.get("pipeline", {}).get("transition_estimator", {}) or {}).get("name") == "anchor"
        for field in fields:
            if field["path"] == "pipeline.warmup_epochs":
                field["visible"] = estimated
            if field["path"] == "pipeline.transition_estimator.percentile":
                field["visible"] = estimated
    paper_change_paths = (
        [str(change["path"]) for change in _paper_parameter_changes(config, method)[1]]
        if allow_missing else [str(field["path"]) for field in paper_changes]
    )
    return fields, {
        "registry_version": str(registry.get("registry_version", "")),
        "formal_recipe": recipe_id,
        "paper_title": str(paper_metadata.get("title", "")),
        "paper_source": str(paper_metadata.get("original_source", "")),
        "modified_from_paper": bool(paper_change_paths),
        "paper_changes": paper_change_paths,
        "research_groups": [
            {
                "id": group,
                "label": str(research_groups.get(group, {}).get("label_zh", group)),
                "description": str(research_groups.get(group, {}).get("description_zh", "")),
            }
            for group in ("method", "data", "training")
        ],
        "research_categories": [
            {
                "id": category,
                "label": str(details.get("label_zh", category)),
                "description": str(details.get("description_zh", "")),
                "default_expanded": bool(details.get("default_expanded", False)),
            }
            for category, details in research_categories.items()
        ],
        "levels": [
            {
                "id": level,
                "label": str(levels.get(level, {}).get("label_zh", level)),
                "default_expanded": bool(
                    levels.get(level, {}).get("default_expanded", False)
                ),
            }
            for level in _PARAMETER_LEVEL_ORDER
        ],
    }


def _config_schema(
    recipe_id: object = None,
    *,
    path_value: object = None,
    recipe_hint: object = None,
) -> dict[str, object]:
    payload = _config_payload(recipe_id, path_value=path_value)
    return schema_from_payload(payload, recipe_hint=recipe_hint)


def schema_from_payload(payload: dict[str, Any], *, recipe_hint=None) -> dict[str, object]:
    """Describe an in-memory draft using exactly the Web field policy."""
    method = _config_method(
        payload["config"], payload["recipe"] or recipe_hint or payload["method"]
    )
    formal_recipe = _parameter_registry()["formal_recipe_bindings"].get(method, "")
    if method == "jocor":
        from lnl_toolbox.core.config_schema import normalize_experiment_config

        payload["config"] = normalize_experiment_config(payload["config"])
    strict_registry = bool(payload["recipe"] and payload["recipe"] == formal_recipe)
    registry_fields = _registry_config_fields(
        payload["config"], method, allow_missing=not strict_registry
    )
    fields, metadata = (
        registry_fields
        if registry_fields is not None
        else (
            _legacy_editable_config_fields(payload["config"]),
            {
                "registry_version": "",
                "formal_recipe": "",
                "paper_title": "",
                "paper_source": "",
                "modified_from_paper": False,
                "paper_changes": [],
                "levels": [
                    {"id": "basic", "label": "基础参数", "default_expanded": True}
                ],
            },
        )
    )
    fields = [field for field in fields if not str(field["path"]).startswith("early_stopping.")]
    fields = list({field["path"]: field for field in [*fields, *_schedule_editor_fields(payload["config"], method=method)]}.values())
    fields.extend(_model_editor_fields(payload["config"], fields))
    original = (
        _config_payload(recipe_hint)["config"]
        if recipe_hint and payload["recipe"] != recipe_hint
        else payload["config"]
    )
    original_count = reference_train_size(original)
    for field in fields:
        field["display_group"] = _parameter_display_group(field, method)
        if method in _parameter_registry()["methods"] and field.get("visible") is not False and not _parameter_ui_spec(str(field["path"])):
            raise ValueError(f"参数尚未提供具体说明：{field['path']}")
        if _parameter_ui_spec(str(field["path"])):
            field["label"] = _parameter_user_label(str(field["path"]), method)
            field["note"] = _parameter_user_note(str(field["path"]), "", method)
        _annotate_model_field(field, payload["config"])
        field["choices"] = _parameter_choices(field, method)
        field["rows"] = _parameter_list_rows(field, payload["config"], method)
        path = tuple(str(field.get("path", "")).split("."))
        if original_count and path in SPLIT_COUNT_PATHS:
            source = original
            for part in path:
                source = source.get(part) if isinstance(source, dict) else None
            if isinstance(source, int) and not isinstance(source, bool):
                field["split_reference"] = {"count": source, "total": original_count}
    fields.extend(_stage_early_stopping_fields(payload["config"]))
    display_order = _parameter_registry().get("default_parameter_paths", {}).get("methods", {}).get(_method_registry_key(method), ())
    positions = {path: index for index, path in enumerate(display_order)}
    fields.sort(key=lambda field: positions.get(str(field["path"]), len(positions))
                if field.get("display_group") == "default" else len(positions))
    if method == "pcse":
        stages = ("pretraining_stage", "transition_stage", "feature_stage",
                  "statistics_stage", "gda_stage", "ensemble_stage")
        for stage in stages:
            controls = [field for field in fields if field["path"] == stage + ".early_stopping"]
            for control in controls:
                fields.remove(control)
                indices = [index for index, field in enumerate(fields)
                           if field.get("display_group") == control["display_group"]
                           and str(field["path"]).startswith(stage + ".")]
                fields.insert(indices[-1] + 1 if indices else len(fields), control)
    return {
        "recipe": payload["recipe"],
        "source_path": payload["path"],
        "runner": payload["runner"],
        "method": method,
        "fields": fields,
        **metadata,
    }


def _field_map(config: dict[str, Any], method: str) -> dict[str, dict[str, object]]:
    registry_fields = _registry_config_fields(config, method, allow_missing=True)
    fields = (
        registry_fields[0]
        if registry_fields is not None
        else _legacy_editable_config_fields(config)
    )
    result = {str(field["path"]): field for field in fields
              if not str(field["path"]).startswith("early_stopping.")}
    for model_field in _model_editor_fields(config, list(result.values())):
        result[str(model_field["path"])] = model_field
    for schedule_field in _schedule_editor_fields(config, method=method):
        result[str(schedule_field["path"])] = schedule_field
    for stage_field in _stage_early_stopping_fields(config):
        result[str(stage_field["path"])] = stage_field
    for field in result.values():
        _annotate_model_field(field, config)
        field["choices"] = _parameter_choices(field, method)
        field["rows"] = _parameter_list_rows(field, config, method)
    return result


def _annotate_model_field(field: dict[str, object], config: dict[str, Any]) -> None:
    path = str(field.get("path", ""))
    if path.endswith((".model.variant", ".model.weights")) or path in {"model.variant", "model.weights"}:
        parent = path.rsplit(".", 1)[0]
        _, model = _config_path_value(config, parent)
        if isinstance(model, dict):
            field["model_name"] = str(model.get("name", ""))
            field["model_variant"] = str(model.get("variant", "standard"))


def _model_editor_fields(config: dict[str, Any], fields: list[dict[str, object]]) -> list[dict[str, object]]:
    """Expose architecture and weight settings alongside editable model names."""
    known = {str(field["path"]) for field in fields}
    result = []
    for field in fields:
        path = str(field["path"])
        if not field.get("editable") or not (path == "model.name" or path.endswith(".model.name")):
            continue
        parent = path.rsplit(".", 1)[0]
        _, model = _config_path_value(config, parent)
        if not isinstance(model, dict):
            continue
        for suffix in ("variant", "weights"):
            field_path = parent + "." + suffix
            if field_path in known:
                continue
            result.append({"path": field_path, "value": model.get(suffix, "standard" if suffix == "variant" else "none"),
                           "kind": "text", "level": "basic", "visible": True, "editable": True,
                           "presentation": "parameter", "research_group": "", "research_category": "",
                           "linked_fields": [], "note": "", "virtual": suffix not in model, "model_editor": True})
    return result


def _stage_early_stopping_fields(config: dict[str, Any]) -> list[dict[str, object]]:
    from lnl_toolbox.core.config_schema import training_stage_controls

    method = _config_method(config)
    return [{
        "path": control["path"], "label": _parameter_user_label(control["path"], method),
        "value": control["value"], "kind": "boolean", "nullable": False,
        "level": "basic", "display_group": "default", "editable": True,
        "visible": True, "presentation": "parameter", "choices": None,
        "note": _parameter_user_note(control["path"], "", method),
        "research_group": "", "research_category": "", "linked_fields": [],
    } for control in training_stage_controls(config)]


def _schedule_epochs(config: dict[str, Any], path: str, *, method: str = "") -> int:
    """Match the epoch budget passed by the runner to this schedule."""
    method = method or _config_method(config)
    budget = config.get("trainer", {}).get("epochs", 1)
    owner = config
    for part in path.split(".")[:-1]:
        owner = owner.get(part, {}) if isinstance(owner, dict) else {}
        budget = owner.get("epochs", budget)
    if method == "dld":
        budget = config.get("dld", {}).get("diffusion", {}).get("epochs", budget)
    if method == "dividemix":
        dm = config.get("dividemix", {})
        budget = dm.get("warmup", {}).get("epochs", 0) + dm.get("training", {}).get("epochs", budget)
    return int(budget)


def _schedule_editor_fields(config: dict[str, Any], *, method: str = "") -> list[dict[str, object]]:
    """Expose absolute stage rates wherever the training scheduler supports them."""
    result: list[dict[str, object]] = []
    execution = config.get("execution") or {}
    method = method or _config_method(config)
    cwd = config.get("method") == "cwd" or isinstance(execution, dict) and execution.get("runner") == "cwd"
    parameters = _parameter_registry()["methods"].get(method, {}).get("parameters", {})
    registered_paths = {path for entries in parameters.values() if isinstance(entries, dict) for path in entries}

    def editable_schedule(path: str) -> bool:
        for level, entries in parameters.items():
            if isinstance(entries, dict) and path + ".name" in entries:
                return _parameter_edit_rule(config, method, path + ".name", level)[0] == "editable"
        return cwd and path == "scheduler"

    if cwd and isinstance(config.get("scheduler"), dict):
        result.append({
            "path": "scheduler.name", "label": _parameter_user_label("scheduler.name", method),
            "value": config["scheduler"].get("name", "multistep"), "kind": "text",
            "level": "advanced", "research_group": "", "research_category": "",
            "editable": True, "visible": True, "presentation": "parameter",
            "note": _parameter_user_note("scheduler.name", "", method),
        })

    def visit(value: object, path: str = "") -> None:
        if not isinstance(value, dict):
            return
        if "scheduler" in path.split(".") and editable_schedule(path) and ("name" in value or path == "scheduler" and cwd):
            name_field = {"path": path + ".name", "kind": "text", "value": value.get("name", "multistep"),
                          "editable": True, "visible": True}
            modes = [option["value"] for option in (_parameter_choices(name_field, method) or [])]
            budget = _schedule_epochs(config, path, method=method)
            defaults: dict[str, object] = {}
            if "step_milestones" not in value:
                if "multistep" in modes:
                    defaults.update(milestones=[], gamma=0.1, lr_values=None)
                if "cosine" in modes:
                    defaults.update(t_max=budget, eta_min=5e-4 if method == "fine" else 0.0)
                if "linear_after" in modes:
                    if method == "mc_ldce" and path == "scheduler":
                        defaults.update(decay_start=0)
                    else:
                        defaults.update(start_epoch=0, end_epoch=budget)
                if "step" in modes:
                    defaults.update(step_size=1, gamma=0.1)
            for suffix, default in defaults.items():
                field_path = path + "." + suffix
                if suffix in value and field_path in registered_paths:
                    continue
                result.append({
                    "path": field_path, "label": _parameter_user_label(field_path, method),
                    "note": _parameter_user_note(field_path, "", method), "value": value.get(suffix, default),
                    "kind": "list" if suffix in {"milestones", "lr_values"} else "number",
                    "number_type": "integer" if suffix in {"start_epoch", "end_epoch", "decay_start", "t_max", "step_size"} else "float",
                    "nullable": suffix == "lr_values", "virtual": suffix not in value,
                    "editable": True, "visible": True, "level": "advanced", "presentation": "parameter",
                    "research_group": "", "research_category": "",
                })
            if method == "fine":
                # FINE owns its post-warmup cosine schedule; name/t_max are not switches.
                duration = max(1, int(budget) - int(config.get("fine", {}).get("warmup_epochs", 0)))
                result.extend([
                    {**name_field, "value": "cosine", "level": "advanced", "label": "scheduler.name", "note": "预热结束后使用余弦下降。"},
                    {"path": path + ".t_max", "value": duration, "kind": "number", "number_type": "integer",
                     "editable": False, "visible": True, "level": "advanced", "label": "scheduler.t_max",
                     "note": "下降周期为主训练阶段的轮数。"},
                ])
        if "scheduler" in path.split(".") and (
            (value.get("name") in {"multistep", "none"} and isinstance(value.get("milestones"), list))
            or (path == "scheduler" and isinstance(value.get("step_milestones"), list))
            or (path == "scheduler" and cwd and isinstance(value.get("milestones"), list))
        ):
            result.append({
                "path": path + ".lr_values", "label": _parameter_user_label(path + ".lr_values", method),
                "value": value.get("lr_values"), "kind": "list", "nullable": True,
                "level": "advanced", "research_group": "", "research_category": "",
                "editable": True, "visible": True, "presentation": "parameter",
                "note": _parameter_user_note(path + ".lr_values", "", method),
            })
        for key, child in value.items():
            if isinstance(child, dict):
                visit(child, f"{path}.{key}" if path else key)

    visit(config)
    return result


def _validate_schedule_editor_rates(config: dict[str, Any]) -> None:
    execution = config.get("execution") or {}
    cwd = config.get("method") == "cwd" or isinstance(execution, dict) and execution.get("runner") == "cwd"
    def visit(value: object, path: str = "") -> None:
        if not isinstance(value, dict):
            return
        if "scheduler" in path.split(".") and value.get("lr_values") is not None:
            nodes = value.get("step_milestones", value.get("milestones"))
            rates = value["lr_values"]
            if ("step_milestones" not in value
                    and str(value.get("name", "multistep" if cwd else "none")) != "multistep"):
                raise ValueError(f"{path}.lr_values 仅适用于分段学习率")
            if not isinstance(nodes, list) or not isinstance(rates, list) or len(nodes) != len(rates):
                raise ValueError(f"{path}.lr_values requires one value per milestone")
            if (any(isinstance(node, bool) or not isinstance(node, int) or node <= 0 for node in nodes)
                    or nodes != sorted(set(nodes))
                    or any(isinstance(rate, bool) or not isinstance(rate, (int, float))
                           or not math.isfinite(rate) or rate <= 0 for rate in rates)):
                raise ValueError(f"{path}.lr_values 需要与递增且不重复的节点一一对应，且学习率必须为正数")
        for key, child in value.items():
            if isinstance(child, dict):
                visit(child, f"{path}.{key}" if path else key)

    visit(config)


def _set_config_path(config: dict[str, object], path: str, value: object, *, method: str = "") -> None:
    from lnl_toolbox.core.config_schema import training_stage_controls

    if path in {control["path"] for control in training_stage_controls(config)}:
        # Use the same typed override implementation as `lnl run --set`.
        from lnl_toolbox.core.config_overrides import apply_override

        updated = apply_override(config, path, value)
        config.clear()
        config.update(updated)
        return
    parts = path.split(".")
    current: dict[str, object] = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"不允许新增或重组配置字段：{path}")
        current = child
    execution = config.get("execution") or {}
    cwd = config.get("method") == "cwd" or isinstance(execution, dict) and execution.get("runner") == "cwd"
    if path.endswith(".lr_values") or (path == "scheduler.name" and cwd):
        if value is None:
            current.pop(parts[-1], None)
        else:
            current[parts[-1]] = value
            if cwd and path == "scheduler.lr_values" and "name" not in current:
                # CWD's existing unnamed scheduler is implicitly multistep.
                current["name"] = "multistep"
        return
    if parts[-1] not in current:
        if parts[-1] in {"variant", "weights"} and parts[-2] == "model" and "name" in current:
            current[parts[-1]] = value
            return
        if any(field["path"] == path and field.get("virtual") for field in _schedule_editor_fields(config, method=method)):
            current[parts[-1]] = value
            return
        if path == "data.validation_size" and _config_method(config, config.get("execution", {}).get("runner", "")) in {"binary_risk", "cwd"}:
            current[parts[-1]] = value
            return
        raise ValueError(f"配置字段不存在：{path}")
    current[parts[-1]] = value


def _normalize_registry_numeric_fields(config: dict[str, Any], method: str) -> None:
    """Convert numeric Web-editor values before a config is persisted.

    Full-YAML editing can bypass the typed form controls, so normalize only
    paths whose registry metadata declares a numeric value.  In particular,
    an empty nullable numeric field is ``None`` rather than the invalid
    ``""`` that later reaches the data service.  String-declared fields are
    deliberately left untouched.
    """

    registry_fields = _registry_config_fields(config, method, allow_missing=True)
    if registry_fields is None:
        return
    fields, _ = registry_fields
    for field in fields:
        if field.get("kind") != "number":
            continue
        path = str(field["path"])
        found, value = _config_path_value(config, path)
        if not found:
            continue
        nullable = bool(field.get("nullable", False))
        number_type = str(field.get("number_type", ""))
        if value is None:
            if not nullable:
                raise ValueError(f"参数 {path} 必须是数字")
            continue
        if isinstance(value, bool):
            raise ValueError(f"参数 {path} 必须是数字")
        if isinstance(value, str):
            text = value.strip()
            if not text:
                if nullable:
                    _set_config_path(config, path, None)
                    continue
                raise ValueError(f"参数 {path} 必须是数字")
            try:
                value = int(text) if number_type == "integer" else float(text)
            except ValueError as exc:
                raise ValueError(f"参数 {path} 必须是数字") from exc
        if number_type == "integer":
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"参数 {path} 必须是整数")
            _set_config_path(config, path, int(value))
        elif isinstance(value, (int, float)):
            _set_config_path(config, path, value)


def _coerce_patch_value(field: dict[str, object], value: object) -> object:
    kind = field["kind"]
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"参数 {field['path']} 必须是布尔值")
        return value
    if kind == "number":
        if field.get("nullable") and isinstance(value, str) and not value.strip():
            return None
        if field.get("nullable") and value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"参数 {field['path']} 必须是数字")
        if field.get("number_type") == "integer":
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"参数 {field['path']} 必须是整数")
            return int(value)
        return value
    if kind == "list":
        if field.get("nullable") and value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(f"参数 {field['path']} 必须是列表")
        rows = field.get("rows")
        if rows:
            if len(value) != len(rows):
                raise ValueError(f"{field['label']}：请保留 {len(rows)} 项配置")
            for index, (item, row) in enumerate(zip(value, rows)):
                if not isinstance(item, dict) or set(item) != {child["key"] for child in row}:
                    raise ValueError(f"{field['label']}：第 {index + 1} 项格式错误")
                for child in row:
                    _coerce_patch_value(child, item[child["key"]])
        return value
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"参数 {field['path']} 必须是对象")
        return value
    if not isinstance(value, str):
        raise ValueError(f"参数 {field['path']} 必须是文本")
    choices = field.get("choices")
    if choices and value not in {option["value"] for option in choices}:
        raise ValueError(f"{field['label']}：请从下拉选项中选择")
    return value


def _parameter_edit_rule(
    config: dict[str, Any], method: str, path: str, level: str
) -> tuple[str, str, object | None]:
    """One permission decision for both the parameter panel and YAML saves.

    `fixed` cannot change independently; `required` must equal a mode-specific
    value; `derived` is computed from another editable parameter.
    """

    if method == "apl" and path == "evaluation.selection_split" and config.get("trainer", {}).get("early_stopping") is True:
        return "derived", "开启早停时自动使用验证集选择模型。", "validation"
    if level == "locked":
        return "fixed", "", None
    if method == "loss_correction" and path == "pipeline.transition_estimator.name":
        return "fixed", "根据当前噪声是否提供转移矩阵自动选择。", None
    if method == "loss_correction" and path == "pipeline.pretraining_protocol":
        return "fixed", "Loss Correction 使用两阶段预训练协议。", None
    registry = _parameter_registry()
    fixed = registry.get("runtime_fixed_parameter_paths", {})
    if path in fixed.get(method, ()):
        return "fixed", "当前方法实现要求此值固定；不能单独修改。", None
    derived = registry.get("runtime_derived_parameter_paths", {})
    if path in derived.get(method, ()):
        if method == "importance_reweighting":
            model = config.get("model", {})
            if isinstance(model, dict) and model.get("name", "linear") != "linear":
                return "fixed", "当前模型不使用线性输入维度参数。", None
            _, dimension = _config_path_value(config, "data.dimension")
            if dimension is None:
                return "fixed", "输入特征数由当前数据决定。", None
            return "derived", "此值跟随数据维度，请修改数据维度。", int(dimension)
        if method in {"cnlcu", "coteaching"}:
            _, noise_rate = _config_path_value(config, f"{method}.noise_rate")
            return "derived", f"此值由算法估计噪声率计算，请修改 {method}.noise_rate。", 1.0 - float(noise_rate)
        _, noise_rate = _config_path_value(config, "noise.rate")
        expected = 1.0 - float(noise_rate) if path.endswith("remember_schedule.end") else float(noise_rate)
        return "derived", "此值由噪声率自动计算，请修改噪声率。", expected
    if method == "importance_reweighting" and path == "data.dimension":
        data = config.get("data", {})
        required = {"synthetic_binary_2d": 2, "uci_statlog_heart": 13}.get(
            data.get("name") if isinstance(data, dict) else None
        )
        if required is not None:
            return "required", "当前数据集要求固定输入维度。", required
    if method == "l2rw" and path == "meta.virtual_learning_rate":
        meta = config.get("meta", {})
        if isinstance(meta, dict) and str(meta.get("implementation", "paper")).lower() == "official":
            return "required", "official L2RW 实现固定为 1，不执行显式虚拟参数更新", 1.0
    if method == "cdr":
        update = config.get("parameter_update", {})
        mode = str(update.get("compatibility_mode", "paper")).lower() if isinstance(update, dict) else ""
        if mode == "paper" and path in {"optimizer.momentum", "optimizer.weight_decay"}:
            return "required", "CDR paper 模式要求此优化器参数为 0。", 0
        if mode == "official_code" and path == "parameter_update.l1_decay":
            return "required", "CDR official_code 模式要求 L1 衰减为 0。", 0
    return "editable", "", None


def _synchronize_web_derived_parameters(config: dict[str, Any], method: str) -> None:
    if method == "apl" and config.get("trainer", {}).get("early_stopping") is True:
        _set_config_path(config, "evaluation.selection_split", "validation")
    if method == "loss_correction":
        from lnl_toolbox.quickstart.templates import configure_loss_correction_transition

        configure_loss_correction_transition(config)
    if method == "jocor":
        from lnl_toolbox.core.config_schema import normalize_experiment_config

        config.update(normalize_experiment_config(config))
    derived = _parameter_registry().get("runtime_derived_parameter_paths", {})
    for path in derived.get(method, ()):
        if method == "importance_reweighting" and path == "model.in_features" and config.get("model", {}).get("name") != "linear":
            continue
        _, _, expected = _parameter_edit_rule(config, method, path, "advanced")
        _set_config_path(config, path, expected)


def _validate_changed_model_configs(
    before: dict[str, Any], after: dict[str, Any], method: str
) -> None:
    """Catch unsupported Web model edits before a training job is created."""

    import torch

    def check_factory(factory, *args) -> None:
        with torch.random.fork_rng(devices=[]):
            factory(*args)

    if method == "jocor":
        if before.get("models") != after.get("models"):
            from lnl_toolbox.training.multi_model_experiment import _build_member

            for model in after.get("models", ()):
                check_factory(_build_member, model, int(after.get("data", {}).get("num_classes", 10)))
        return
    parameters = _parameter_registry()["methods"].get(method, {}).get("parameters", {})
    model_paths = {
        str(path).removesuffix(".name")
        for entries in parameters.values() if isinstance(entries, dict)
        for path in entries if path == "model.name" or str(path).endswith(".model.name")
    }
    classes = int(after.get("data", {}).get("num_classes", 10))
    for path in model_paths:
        if path.startswith("dld.feature_extractor."):
            continue  # Its separate source/adapter contract validates this model.
        before_found, previous = _config_path_value(before, path)
        after_found, model = _config_path_value(after, path)
        if not after_found or (before_found and previous == model):
            continue
        if not isinstance(model, dict):
            raise ValueError(f"模型配置必须是 mapping：{path}")
        # Validation must not download pretrained weights.  The architecture is
        # checked with random weights; the configured weights remain untouched.
        model = dict(model)
        if str(model.get("weights", "none")).lower() != "none":
            if int(model.get("input_channels", 3)) != 3:
                raise ValueError("ImageNet pretrained weights require three input channels")
            model["weights"] = "none"
        if method == "binary_risk" or (method == "importance_reweighting" and model.get("name", "linear") == "linear"):
            from lnl_toolbox.training.binary_experiment import build_binary_model

            dimension = int(after.get("data", {}).get("dimension", model.get("in_features", 2)))
            check_factory(build_binary_model, dimension, model)
        elif method == "cwd":
            from lnl_toolbox.training.cwd_experiment import _build_model

            check_factory(_build_model, model)
        elif method == "fine":
            from lnl_toolbox.training.fine_experiment import _build_fine_model

            check_factory(_build_fine_model, model, classes)
        elif method in {"cal", "ca2c", "l2rw", "mc_ldce"}:
            from lnl_toolbox.training.reproduction_data import build_reproduction_model

            candidate = dict(model)
            if candidate.get("name") == "feature_mlp" and "input_dim" not in candidate:
                candidate["input_dim"] = int(after.get("data", {}).get("dimension", 1))
            check_factory(build_reproduction_model, candidate, after.get("data", {}), classes)
        else:
            from lnl_toolbox.training.experiment import build_model

            candidate = dict(model)
            if candidate.get("name") == "feature_mlp" and "input_dim" not in candidate:
                candidate["input_dim"] = int(after.get("data", {}).get("dimension", 1))
            check_factory(build_model, candidate, classes)


def _validate_changed_data_noise_selectors(before: dict[str, Any], after: dict[str, Any]) -> None:
    previous_data, current_data = before.get("data") or {}, after.get("data") or {}
    if previous_data.get("name") != current_data.get("name"):
        from lnl_toolbox.training.data_service import DATASETS

        DATASETS.get(after["data"]["name"])
    previous_noise, current_noise = before.get("noise") or {}, after.get("noise") or {}
    if previous_noise.get("name") != current_noise.get("name"):
        name = str(current_noise.get("name", "clean")).lower()
        direct = {
            "", "clean", "none", "native", "real_world", "external",
            "pdl", "official_uniform_flip", "binary_asymmetric_rcn",
            "asymmetric_rcn", "external_torch",
        }
        if name not in direct:
            from lnl_toolbox.training.noisy_labels import _generated_spec

            _generated_spec(after)


def _validate_changed_training_selectors(
    before: dict[str, Any], after: dict[str, Any], method: str
) -> None:
    """Use the actual optimizer/scheduler builders for changed Web fields."""

    parameters = _parameter_registry()["methods"].get(method, {}).get("parameters", {})
    paths = {
        str(path).rsplit(".", 1)[0]
        for entries in parameters.values() if isinstance(entries, dict)
        for path in entries
        if str(path).endswith(("optimizer.name", "scheduler.name"))
    }
    for path in paths:
        before_found, previous = _config_path_value(before, path)
        after_found, value = _config_path_value(after, path)
        if not after_found or (before_found and previous == value):
            continue
        if not isinstance(value, dict):
            raise ValueError(f"训练参数必须是 mapping：{path}")
        import torch

        with torch.random.fork_rng(devices=[]):
            dummy = torch.nn.Linear(1, 1)
        if path.endswith("optimizer"):
            if method == "pcse" and path == "transition_stage.optimizer":
                continue  # PCSE validates its two-learning-rate optimizer directly.
            if method == "jocor":
                from lnl_toolbox.training.multi_model_experiment import _build_optimizer

                _build_optimizer(dummy.parameters(), value)
            else:
                from lnl_toolbox.training.experiment import build_optimizer

                build_optimizer(dummy, value)
        else:
            name = str(value.get("name", "none")).lower()
            if method == "mc_ldce" and path == "transition.scheduler" and name not in {"none", "multistep"}:
                raise ValueError("MC-LDCE transition scheduler 只支持 none 或 multistep")
            optimizer = torch.optim.SGD(dummy.parameters(), lr=0.01)
            epochs = _schedule_epochs(after, path, method=method)
            if method == "cal" and path == "scheduler":
                from lnl_toolbox.training.experiment import build_alpha_scaled_scheduler

                build_alpha_scaled_scheduler(optimizer, value)
            elif method == "jocor":
                from lnl_toolbox.training.multi_model_experiment import _apply_epoch_optimizer_schedule

                _apply_epoch_optimizer_schedule(optimizer, value, epoch=0, total_epochs=epochs)
            elif method == "mc_ldce" and path == "scheduler" and name == "linear_after":
                # MC-LDCE's runner uses decay_start, not the generic start_epoch.
                start = value.get("decay_start")
                if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < epochs:
                    raise ValueError("MC-LDCE scheduler.decay_start 必须为非负整数且小于训练轮数")
            else:
                from lnl_toolbox.training.experiment import build_scheduler

                build_scheduler(optimizer, value, epochs)


def _assert_locked_parameters_unchanged(
    before: dict[str, Any], after: dict[str, Any], method: str,
    *, allowed_changes: set[str] | None = None,
) -> None:
    allowed_changes = allowed_changes or set()
    parameters = _parameter_registry()["methods"].get(method, {}).get("parameters", {})
    for level, entries in parameters.items():
        if not isinstance(entries, dict):
            continue
        for path in entries:
            if any(
                str(path) == allowed or str(path).startswith(allowed + ".")
                for allowed in allowed_changes
            ):
                continue
            permission, reason, required = _parameter_edit_rule(after, method, str(path), level)
            if permission not in {"fixed", "required", "derived"}:
                continue
            before_found, before_value = _config_path_value(before, str(path))
            after_found, after_value = _config_path_value(after, str(path))
            if permission == "fixed" and (
                before_found != after_found or before_value != after_value
            ):
                raise ValueError(f"锁定参数不能通过普通 WebUI 修改：{path}")
            if permission in {"required", "derived"} and (
                not after_found or after_value != required
            ):
                raise ValueError(f"{path}：{reason}")


def _paper_parameter_changes(
    config: dict[str, Any], method: str
) -> tuple[str, list[dict[str, object]]]:
    if method == "jocor":
        from lnl_toolbox.core.config_schema import normalize_experiment_config

        config = normalize_experiment_config(config)
    formal = _formal_registry_config(method)
    if formal is None:
        return "", []
    recipe_id, baseline = formal
    method_metadata = _parameter_registry()["methods"].get(method, {})
    parameters = method_metadata.get("parameters", {}) if isinstance(method_metadata, dict) else {}
    paper = parameters.get("paper", {}) if isinstance(parameters, dict) else {}
    changes: list[dict[str, object]] = []
    if isinstance(paper, dict):
        for path in paper:
            baseline_found, baseline_value = _config_path_value(baseline, str(path))
            current_found, current_value = _config_path_value(config, str(path))
            if not baseline_found and method == "loss_correction" and str(path) == "pipeline.transition_estimator.percentile":
                continue
            if not baseline_found:
                raise ValueError(f"论文参数路径不存在：{method}.{path}")
            if not current_found or current_value != baseline_value:
                changes.append(
                    {
                        "path": str(path),
                        "changed_from": baseline_value,
                        "value": current_value if current_found else None,
                    }
                )
    return recipe_id, changes


def _record_paper_parameter_status(
    config: dict[str, Any], method: str, *, acknowledged: bool
) -> None:
    recipe_id, changes = _paper_parameter_changes(config, method)
    if not recipe_id:
        return
    existing = config.get("parameter_record", {})
    legacy_web_record = (
        dict(existing)
        if isinstance(existing, dict)
        and "formal_recipe" in existing
        and not {"paper", "sampling_seed", "parameters", "candidates"}.issubset(existing)
        else {}
    )
    if legacy_web_record:
        config.pop("parameter_record", None)
    meta_value = config.get("meta", {})
    meta = dict(meta_value) if isinstance(meta_value, dict) else {}
    stored = meta.get("web_parameter_record", {})
    record = dict(stored) if isinstance(stored, dict) else legacy_web_record
    previously_acknowledged = bool(record.get("paper_change_acknowledged", False))
    if changes and not (acknowledged or previously_acknowledged):
        paths = ", ".join(str(change["path"]) for change in changes)
        raise ValueError(f"修改论文参数需要确认复现影响：{paths}")
    record.update(
        {
            "parameter_metadata_registry_version": str(
                _parameter_registry().get("registry_version", "")
            ),
            "formal_recipe": recipe_id,
            "modified_from_paper": bool(changes),
            "paper_change_acknowledged": bool(changes),
            "paper_parameter_changes": changes,
            "effective_reproduction_status": (
                "modified_from_paper" if changes else "formal_recipe"
            ),
        }
    )
    meta["web_parameter_record"] = record
    config["meta"] = meta


def prepare_config(payload: object, *, for_review: bool = False) -> tuple[dict[str, Any], str]:
    """Validate/coerce an edit without creating a directory or writing YAML."""
    if not isinstance(payload, dict):
        raise ValueError("请求内容必须是 JSON 对象")
    recipe_id = payload.get("recipe")
    source_path = payload.get("source_path")
    content = payload.get("content")
    patches = payload.get("patches")
    system_overrides = payload.get("_quick_start_overrides")
    authorized_system_overrides = (
        payload.get("_quick_start_override_auth") is _QUICK_START_OVERRIDE_AUTH
    )
    if system_overrides is not None and not authorized_system_overrides:
        raise ValueError("Quick Start 内部配置只能由运行计划生成")
    if system_overrides is not None and not isinstance(system_overrides, dict):
        raise ValueError("Quick Start 内部配置格式错误")
    from lnl_toolbox.catalog import load_yaml, recipe_by_id, validate_config

    source_config: dict[str, Any] | None = None
    source_method = ""
    if isinstance(recipe_id, str) and recipe_id.strip():
        source_recipe = recipe_by_id(recipe_id, ROOT)
        source_config = load_yaml(source_recipe.config_path)
        source_method = _config_method(source_config, recipe_id or source_recipe.method)
    elif str(source_path or "").strip():
        source = _project_path(source_path)
        if not source.is_file():
            raise FileNotFoundError("来源 YAML 不存在")
        source_config = load_yaml(source)
        source_method = _config_method(source_config, payload.get("recipe_hint", ""))

    if content is not None:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("YAML 内容不能为空")
        try:
            import yaml

            parsed = yaml.safe_load(content)
        except Exception as exc:
            raise ValueError(f"YAML 解析失败：{exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("YAML 顶层必须是 mapping")
    else:
        if source_config is None:
            raise ValueError("保存 YAML 必须指定来源 recipe 或项目 YAML")
        parsed = load_yaml(recipe_by_id(recipe_id, ROOT).config_path) if (
            isinstance(recipe_id, str) and recipe_id.strip()
        ) else load_yaml(_project_path(source_path))
        if not isinstance(patches, list):
            raise ValueError("只能通过参数菜单提交 YAML 修改")
        method = source_method or _config_method(parsed, recipe_id)
        fields = _field_map(parsed, method)
        for patch in patches:
            if not isinstance(patch, dict) or not isinstance(patch.get("path"), str):
                raise ValueError("YAML 参数修改格式错误")
            path = patch["path"]
            if _internal_seed_path(path):
                raise ValueError(f"内部随机种子由 seed 自动生成，不能单独修改：{path}")
            field = fields.get(path)
            if field is None:
                raise ValueError(f"不允许修改配置结构或组件：{path}")
            if not field.get("editable", False):
                raise ValueError(f"锁定参数不能通过普通 WebUI 修改：{path}")
            # Model choices depend on the final name/variant; validate the
            # complete combination after all patches rather than against the
            # old model's dropdown choices.
            coercion_field = dict(field)
            if "model_name" in field and path.endswith((".variant", ".weights")):
                coercion_field["choices"] = None
            _set_config_path(parsed, path, patch.get("value") if for_review else _coerce_patch_value(coercion_field, patch.get("value")), method=method)
        for patch in patches:
            path = patch["path"]
            if path.endswith(".name") and "scheduler" in path.split("."):
                _, schedule = _config_path_value(parsed, path.rsplit(".", 1)[0])
                if isinstance(schedule, dict) and schedule.get("name") != "multistep" and "step_milestones" not in schedule:
                    schedule.pop("lr_values", None)
        from lnl_toolbox.catalog import _config_parameter_errors
        if not for_review or not _config_parameter_errors(parsed):
            _synchronize_web_derived_parameters(parsed, method)
    allowed_system_changes: set[str] = set()
    for raw_path, value in (system_overrides or {}).items():
        path = str(raw_path).strip()
        if not path:
            raise ValueError("Quick Start 内部配置路径不能为空")
        if path == "noise":
            if not isinstance(value, dict):
                raise ValueError("Quick Start 噪声配置必须是 mapping")
            parsed["noise"] = dict(value)
        else:
            _set_config_path(parsed, path, value)
        allowed_system_changes.add(path)
    method = source_method or _config_method(source_config or parsed, _config_method(parsed))
    if method == "loss_correction":
        pipeline_values = parsed.get("pipeline", {}) or {}
        if (pipeline_values.get("transition_estimator", {}) or {}).get("name") == "anchor":
            explicitly_zero = any(isinstance(item, dict) and item.get("path") == "pipeline.warmup_epochs" and int(item.get("value", 0)) <= 0 for item in (patches or []))
            if explicitly_zero or (content is not None and int(pipeline_values.get("warmup_epochs", 0)) <= 0):
                raise ValueError("未知 T 必须先训练 CE 分类器；预训练轮数必须大于 0。")
        _synchronize_web_derived_parameters(parsed, method)
    if source_config is None:
        formal = _formal_registry_config(method)
        source_config = formal[1] if formal is not None else parsed
    from lnl_toolbox.catalog import _config_parameter_errors
    raw_errors = _config_parameter_errors(parsed) if for_review else []
    if not for_review:
        _normalize_registry_numeric_fields(parsed, method)
        _validate_schedule_editor_rates(parsed)
    from lnl_toolbox.core.config_schema import synchronize_experiment_seed

    if not raw_errors:
        if method == "jocor":
            # Both Web entry points must describe the same initial rate. The
            # training scheduler would otherwise overwrite optimizer.lr.
            rate_paths = {"optimizer.lr", "scheduler.initial_lr"}
            if content is None:
                changed_rates = {item["path"] for item in (patches or [])
                                 if isinstance(item, dict) and item.get("path") in rate_paths}
            else:
                changed_rates = {path for path in rate_paths
                                 if _config_path_value(source_config, path) != _config_path_value(parsed, path)}
            if changed_rates:
                optimizer_rate = _config_path_value(parsed, "optimizer.lr")[1]
                schedule_rate = _config_path_value(parsed, "scheduler.initial_lr")[1]
                if changed_rates == rate_paths and optimizer_rate != schedule_rate:
                    raise ValueError("JoCoR 起始学习率设置冲突：optimizer.lr 与 scheduler.initial_lr 必须一致")
                rate = optimizer_rate if "optimizer.lr" in changed_rates else schedule_rate
                _set_config_path(parsed, "optimizer.lr", rate)
                _set_config_path(parsed, "scheduler.initial_lr", rate)
        if method == "apl":
            _synchronize_web_derived_parameters(parsed, method)
        parsed = synchronize_experiment_seed(parsed)
        _assert_locked_parameters_unchanged(
            source_config, parsed, method, allowed_changes=allowed_system_changes,
        )
        if payload.get("automatic_validation_split"):
            from lnl_toolbox.quickstart.templates import configure_validation_split

            configure_validation_split(parsed)
    _record_paper_parameter_status(
        parsed,
        method,
        acknowledged=bool(payload.get("acknowledge_paper_impact", False)),
    )
    if not for_review:
        validate_config(parsed)
        _validate_changed_data_noise_selectors(source_config, parsed)
        _validate_changed_model_configs(source_config, parsed, method)
        _validate_changed_training_selectors(source_config, parsed, method)
    dataset_alias = str(payload.get("dataset_alias") or "").strip()
    if dataset_alias and not for_review:
        from lnl_toolbox.training.service import ExperimentService

        service = ExperimentService()
        _, compatibility = service.list_config_compatibility(
            dataset_alias, {"candidate": parsed}
        )[0]
        if compatibility.status.value != "compatible":
            reasons = "; ".join(
                f"{item.code}: {item.message}" for item in compatibility.reasons
            )
            raise ValueError(
                f"所选数据集 {dataset_alias!r} 与当前配置不兼容："
                f"{compatibility.status.value}; {reasons}"
            )
    return parsed, dataset_alias


def _save_config(payload: object, *, for_review: bool = False, destination: Path | None = None) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("请求内容必须是 JSON 对象")
    destination = destination if destination is not None else _project_path(payload.get("path"))
    parsed, dataset_alias = prepare_config(payload, for_review=for_review)
    try:
        import yaml

        content = yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True)
    except Exception as exc:
        raise ValueError(f"YAML 生成失败：{exc}") from exc
    overwrite = bool(payload.get("overwrite", False))
    if destination.resolve() in _builtin_config_paths():
        raise ValueError("不能直接覆盖内置 recipe，请另存为新的 YAML")
    if destination.exists() and not overwrite:
        raise FileExistsError("目标文件已存在；如需修改请使用覆盖保存")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return {
        "path": str(destination) if for_review else _display_path(destination),
        "bytes": destination.stat().st_size,
        "content": content,
        "dataset_alias": dataset_alias or None,
    }


