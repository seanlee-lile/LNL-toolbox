from __future__ import annotations

"""CLI-only language preferences. Machine records and backend logs stay untouched."""

import argparse
import builtins
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re

_LANGUAGE = ContextVar("lnl_cli_language", default="zh")


def preference_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CONFIG_HOME")
    return Path(base) / "lnl-toolbox" / "cli.json" if base else Path.home() / ".config" / "lnl-toolbox" / "cli.json"


def saved_language() -> str | None:
    try:
        value = json.loads(preference_path().read_text(encoding="utf-8")).get("language")
        return value if value in {"zh", "en"} else None
    except (OSError, ValueError, AttributeError):
        return None


def save_language(value: str) -> None:
    if value not in {"zh", "en"}:
        raise ValueError("language must be zh or en")
    path = preference_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"language": value}) + "\n", encoding="utf-8")
    temporary.replace(path)
    _LANGUAGE.set(value)


def language() -> str:
    return _LANGUAGE.get()


def select_language(explicit: str | None = None):
    value = explicit or os.environ.get("LNL_LANG") or saved_language() or "zh"
    if value not in {"zh", "en"}:
        raise ValueError("LNL_LANG must be zh or en")
    return _LANGUAGE.set(value)


def reset_language(token) -> None:
    _LANGUAGE.reset(token)


MESSAGES = {
    "显示帮助并退出": "show this help message and exit",
    "参数组合实验预演时跳过数据路径/布局检查": "Skip data-path and layout checks when planning a sweep",
    "清理命令行确认属于 LNL 的旧监听进程": "Stop stale listeners whose command line identifies them as LNL",
    "只清理指定端口；可重复传入。默认扫描所有 LNL 监听端口": "Clean only the specified ports; repeatable. By default scan all LNL listeners",
    "清理当前 Web 端口上的旧 LNL 进程后重新启动 Web UI": "Stop the old LNL listener on the selected port and restart the Web UI",
    "登记并验收本机数据集": "register and verify machine-local datasets",
    "登记本机数据集路径": "register a local dataset path",
    "列出数据集登记与验收状态": "list registration and verification states",
    "查看已登记数据集路径": "show a registered dataset path",
    "查看一条本机数据集登记": "show one local dataset record",
    "加载并检查训练集和测试集布局": "load and validate train/test layout",
    "记录已确认的数据集事实": "record explicit dataset knowledge",
    "删除本机数据集登记": "remove a local registration",
    "运行一轮训练并保存验收证据": "run one epoch and store evidence",
    "准备和查看离线 MentorNet 权重": "prepare and inspect the offline MentorNet artifact",
    "查看数据集与方法兼容性": "discover dataset/method compatibility",
    "启动本机数据管理 Web 界面": "start the local Data Management Web UI",
    "比较已完成的运行目录": "compare completed run directories",
    "生成 Markdown、CSV 和 JSON 报告": "write Markdown, CSV, and JSON run reports",
    "每个 recipe 都是一份可直接预检和运行的完整配置。": "Each recipe is a complete configuration that can be validated and run.",
    "没有符合条件的实验；请调整 --profile 或 --dataset。": "No matching experiments; adjust --profile or --dataset.",
    "组件是构建算法的零件，不一定代表一篇可直接运行的完整论文。": "Components are building blocks, not necessarily complete runnable paper methods.",
    "参数组合实验": "Parameter sweep", "参数组合实验计划": "Parameter sweep plan",
    "  先预览：": "  Validate first:", "能力": "Capabilities", "论文": "Paper",
    "方法": "Method", "规模": "Profile", "实验配置": "Experiment configurations",
    "输出位置": "Output location", "配置": "Configuration",
    "LNL Toolbox 统一命令行入口": "LNL Toolbox command-line interface",
    "检查环境、项目和数据路径": "Check the environment, project and data paths",
    "浏览可运行实验或底层组件": "Browse experiment configurations or components",
    "列出内置实验配置": "List built-in experiment configurations",
    "列出底层可组合组件": "List available components",
    "运行内置 recipe 或自定义配置": "Run a built-in recipe or custom configuration",
    "从运行目录自动恢复": "Resume an existing run",
    "查看并清理 LNL Web 监听进程": "Inspect and clean up LNL Web listeners",
    "参数组合实验：按笛卡尔积运行多组实验": "Run a Cartesian grid of experiment parameters",
    "查看有可运行配置的知名论文": "Browse papers",
    "列出有可运行 config 的论文": "List papers",
    "解释论文在 toolbox 中的详细实现": "View and edit a paper's default parameters",
    "查看论文 YAML 配置": "Print a paper configuration as YAML",
    "实验 recipe 名称或 YAML 路径；与 --recipe、--config 三选一": "Recipe ID or YAML path; choose one of source, --recipe or --config",
    "使用内置实验配置；不要同时填写 source 或 --config": "Use a built-in recipe; mutually exclusive with source and --config",
    "读取实验 YAML 文件；不要同时填写 source 或 --recipe": "Read a YAML file; mutually exclusive with source and --recipe",
    "显式指定项目根目录": "Specify the project root",
    "使用 lnl data register 登记的本机数据集别名，覆盖配置中的数据来源": "Use a registered local dataset alias instead of the configured data source",
    "输出格式；human 适合阅读，tsv 适合脚本处理（默认: human）": "Output format: human (default), TSV or JSON",
    "同时显示需要外部 artifact 的条件可用配置": "Include configurations that require external artifacts",
    "从 checkpoint 文件恢复；保持该运行的配置身份": "Resume a checkpoint without changing the run identity",
    "覆盖该方法支持的训练阶段轮数；多阶段方法按现有规则处理": "Override the method's supported training-stage epoch budget",
    "已有运行目录，包含 resolved_config.yaml；不是 checkpoint 文件路径": "Existing run directory containing resolved_config.yaml, not a checkpoint path",
    "选择 last.pt 或 best.pt；已完成的运行不重新训练": "Choose last.pt or best.pt; completed runs are not retrained",
    "停止当前 Web 端口上的旧 LNL 进程后重新启动 Web UI": "Stop the old LNL listener on this port and restart the Web UI",
    "检查并终止明确属于 LNL 的旧监听进程": "Stop stale listeners verified to belong to LNL",
    "只检查指定端口，可重复传入。默认扫描所有 LNL 相关端口": "Check the specified ports; repeatable. By default scan all LNL ports",
    "只列出候选进程，不终止": "List candidate processes without stopping them",
    "参数组合实验预演时跳过数据路径/条件检查": "Skip data checks while planning a sweep",
    "参数组合实验目录，用于 'lnl sweep status <path>'": "Sweep directory for 'lnl sweep status <path>'",
    "错误": "Error", "配置有效": "Configuration valid", "执行器": "Runner",
    "项目根目录": "Project root", "运行完成": "Run completed", "找到": "Found",
    "标题": "Title", "出处": "Source", "论文链接": "Paper URL", "问题": "Summary",
    "核心机制": "Mechanism", "默认参数": "Default parameters", "高级参数": "Advanced parameters",
    "只读参数": "Read-only parameters", "已取消。": "Cancelled.", "输出目录": "Output directory",
    "输入无效，请重试。": "Invalid input; please try again.",
    "建议": "Next steps", "训练配置": "Training configuration", "训练预算": "Training budget",
    "配置路径": "Configuration path", "数据集": "Dataset", "模型": "Model",
    "噪声": "Noise", "学习率": "Learning rate", "训练轮数": "Training epochs",
    "实验随机种子": "Experiment seed", "训练批次大小": "Training batch size",
}


def text(zh: str, en: str | None = None) -> str:
    if language() == "zh":
        return next((key for key, value in MESSAGES.items() if value == zh), zh) if en is None else zh
    if en is not None:
        return en
    if zh in MESSAGES:
        return MESSAGES[zh]
    patterns = (
        (r"参数 (.+) 必须是数字", r"Parameter \1 must be numeric"),
        (r"参数 (.+) 必须是整数", r"Parameter \1 must be an integer"),
        (r"参数 (.+) 必须是布尔值", r"Parameter \1 must be a boolean"),
        (r"参数 (.+) 必须是列表", r"Parameter \1 must be a list"),
        (r"参数 (.+) 必须是对象", r"Parameter \1 must be an object"),
        (r"锁定参数不能通过普通 WebUI 修改：(.+)", r"This parameter is read-only: \1"),
        (r"找到 (\d+) 个可运行实验(.*)", r"Found \1 runnable experiments\2"),
        (r"找到 (\d+) 个可组合组件，共 (\d+) 类(.*)", r"Found \1 components in \2 categories\3"),
        (r"【(.+)】(\d+) 个", r"[\1] \2 components"),
        (r"(\s*)数据集：(.*?)    噪声：(.*)", r"\1Dataset: \2    Noise: \3"),
        (r"(\s*)方法：(.*?)    执行器：(.*)", r"\1Method: \2    Runner: \3"),
        (r"(\s*)规模：(.*?)    训练轮数：(.*)", r"\1Profile: \2    Epochs: \3"),
    )
    for pattern, replacement in patterns:
        if re.fullmatch(pattern, zh):
            return re.sub(pattern, replacement, zh).replace("（规模=", " (profile=").replace("（类型=", " (kind=").replace("）", ")")
    # Translate labels without touching values, paths, JSON, or backend markers.
    match = re.match(r"^(\s*)([^:：\n]+)([:：])(.*)$", zh, re.S)
    if match and match[2] in MESSAGES:
        return match[1] + MESSAGES[match[2]] + ":" + match[4]
    return zh


def translated_print(*values, **kwargs):
    builtins.print(*(text(v) if isinstance(v, str) and not v.lstrip().startswith(("{", "[{", '["')) else v for v in values), **kwargs)


class _LocalizedSubparsers(argparse._SubParsersAction):
    def add_parser(self, name, **kwargs):
        if kwargs.get("help"):
            kwargs["help"] = text(kwargs["help"])
        return super().add_parser(name, **kwargs)


class LocalizedParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        for key in ("description", "epilog"):
            if kwargs.get(key):
                kwargs[key] = text(kwargs[key])
        super().__init__(*args, **kwargs)
        self.register("action", "parsers", _LocalizedSubparsers)

    def add_argument(self, *args, **kwargs):
        if kwargs.get("help") and kwargs["help"] != argparse.SUPPRESS:
            kwargs["help"] = text(kwargs["help"])
        return super().add_argument(*args, **kwargs)

    def format_help(self):
        result = super().format_help()
        if language() == "en":
            for zh, en in sorted(MESSAGES.items(), key=lambda item: len(item[0]), reverse=True):
                result = result.replace(zh, en)
        else:
            result = result.replace("usage:", "用法:").replace("positional arguments:", "位置参数:").replace("options:", "选项:")
        return result
