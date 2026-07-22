from __future__ import annotations

"""Shared terminal prompts and configuration helpers for command-line entry points."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
from typing import Any, TypeVar


T = TypeVar("T")


class PromptCancelled(Exception):
    """Raised when an interactive terminal session is interrupted."""


class PromptSession:
    """Small dependency-free prompt adapter with injectable terminal I/O."""

    def __init__(
        self,
        read: Callable[[str], str] | None = None,
        write: Callable[[str], None] | None = None,
    ) -> None:
        self._read_fn = read or input
        self._write_fn = write or print

    def write(self, message: str = "") -> None:
        self._write_fn(message)

    def _read(self, prompt: str) -> str:
        try:
            return self._read_fn(prompt).strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise PromptCancelled from exc

    def text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        required: bool = False,
        validator: Callable[[str], bool] | None = None,
        error: str = "输入无效，请重试。",
    ) -> str:
        suffix = f" [{default}]" if default is not None else ""
        while True:
            value = self._read(f"{prompt}{suffix}: ")
            if not value and default is not None:
                value = default
            if not value and required:
                self.write("此项不能为空。")
                continue
            if validator is not None and value and not validator(value):
                self.write(error)
                continue
            return value

    def choose(
        self,
        prompt: str,
        choices: Sequence[tuple[str, T]],
        *,
        default: T | None = None,
    ) -> T:
        if not choices:
            raise ValueError("choices cannot be empty")
        self.write(f"\n{prompt}")
        for index, (label, _) in enumerate(choices, start=1):
            self.write(f"  {index}. {label}")
        default_index = next(
            (index for index, (_, value) in enumerate(choices, start=1) if value == default),
            1,
        )
        while True:
            raw = self._read(f"请选择 [默认 {default_index}]: ")
            if not raw:
                return choices[default_index - 1][1]
            if raw.isdigit() and 1 <= int(raw) <= len(choices):
                return choices[int(raw) - 1][1]
            lowered = raw.casefold()
            for label, value in choices:
                if lowered in {label.casefold(), str(value).casefold()}:
                    return value
            self.write("选项无效，请输入编号或选项名称。")

    def integer(
        self,
        prompt: str,
        *,
        default: int | None = None,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        while True:
            raw = self.text(prompt, default=None if default is None else str(default), required=True)
            try:
                value = int(raw)
            except ValueError:
                self.write("请输入整数。")
                continue
            if minimum is not None and value < minimum:
                self.write(f"数值不能小于 {minimum}。")
                continue
            if maximum is not None and value > maximum:
                self.write(f"数值不能大于 {maximum}。")
                continue
            return value

    def floating(
        self,
        prompt: str,
        *,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
        maximum_inclusive: bool = True,
    ) -> float:
        while True:
            raw = self.text(prompt, default=None if default is None else str(default), required=True)
            try:
                value = float(raw)
            except ValueError:
                self.write("请输入数值。")
                continue
            if not math.isfinite(value):
                self.write("请输入有限数值。")
                continue
            if minimum is not None:
                invalid = value < minimum if minimum_inclusive else value <= minimum
                if invalid:
                    operator = "小于" if minimum_inclusive else "小于或等于"
                    self.write(f"数值不能{operator} {minimum}。")
                    continue
            if maximum is not None:
                invalid = value > maximum if maximum_inclusive else value >= maximum
                if invalid:
                    operator = "大于" if maximum_inclusive else "大于或等于"
                    self.write(f"数值不能{operator} {maximum}。")
                    continue
            return value

    def path(
        self,
        prompt: str,
        *,
        default: Path | None = None,
        required: bool = False,
        must_exist: bool = False,
        file_only: bool = False,
        directory_only: bool = False,
    ) -> Path | None:
        while True:
            raw = self.text(
                prompt,
                default=None if default is None else str(default),
                required=required,
            )
            if not raw:
                return None
            value = Path(raw).expanduser()
            if must_exist and not value.exists():
                self.write(f"路径不存在：{value}")
                continue
            if file_only and value.exists() and not value.is_file():
                self.write(f"需要文件路径：{value}")
                continue
            if directory_only and value.exists() and not value.is_dir():
                self.write(f"需要目录路径：{value}")
                continue
            return value

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        marker = "Y/n" if default else "y/N"
        while True:
            raw = self._read(f"{prompt} [{marker}]: ").casefold()
            if not raw:
                return default
            if raw in {"y", "yes", "1", "是"}:
                return True
            if raw in {"n", "no", "0", "否"}:
                return False
            self.write("请输入 y 或 n。")


@dataclass(slots=True)
class TrainingSelection:
    config: dict[str, Any]
    source: Path
    output_dir: Path | None = None
    resume: Path | None = None
    seeds: list[int] | None = None


def command_arguments(argv: Sequence[str] | None) -> list[str]:
    return list(sys.argv[1:] if argv is None else argv)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"配置文件必须包含 YAML mapping：{path}")
    return dict(value)


def _experiment_templates(*, clean: bool) -> list[tuple[Path, dict[str, Any]]]:
    directories = [Path.cwd() / "configs" / "experiment", repository_root() / "configs" / "experiment"]
    discovered: dict[Path, dict[str, Any]] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.yaml")):
            resolved = path.resolve()
            if resolved in discovered:
                continue
            try:
                config = _load_yaml_mapping(resolved)
            except (OSError, ValueError):
                continue
            model_name = str(config.get("model", {}).get("name", "tiny_cnn")).lower()
            if clean and config.get("noise"):
                continue
            if clean or model_name == "tiny_cnn":
                discovered[resolved] = config
    return list(discovered.items())


def _choose_template(session: PromptSession, *, clean: bool) -> tuple[Path, dict[str, Any]]:
    templates = _experiment_templates(clean=clean)
    if templates:
        preferred = "cifar10_clean_smoke.yaml" if clean else "cifar10_smoke.yaml"
        default = next((path for path, _ in templates if path.name == preferred), templates[0][0])
        path = session.choose(
            "选择实验配置模板",
            [(item.name, item) for item, _ in templates],
            default=default,
        )
    else:
        selected = session.path("实验 YAML 路径", required=True, must_exist=True, file_only=True)
        assert selected is not None
        path = selected
    return path, _load_yaml_mapping(path)


def _prompt_loss(session: PromptSession, current: Mapping[str, Any]) -> dict[str, Any]:
    from lnl_toolbox.plugins.builtin import create_builtin_catalog

    names = [item.name for item in create_builtin_catalog().find(kind="loss")]
    if not names:
        raise RuntimeError("当前环境没有可训练的 PyTorch loss")
    current_name = str(current.get("name", "ce")).lower()
    default = current_name if current_name in names else "ce"
    name = session.choose("选择 loss", [(item.upper(), item) for item in names], default=default)
    if name == "gce":
        return {"name": name, "q": session.floating(
            "GCE q", default=float(current.get("q", 0.7)), minimum=0.0,
            maximum=1.0, minimum_inclusive=False, maximum_inclusive=True,
        )}
    if name == "nce":
        return {"name": name, "eps": session.floating(
            "NCE eps", default=float(current.get("eps", 1e-8)), minimum=0.0,
            minimum_inclusive=False,
        )}
    if name == "mae":
        return {"name": name, "scale": session.floating(
            "MAE scale", default=float(current.get("scale", 2.0)), minimum=0.0,
            minimum_inclusive=False,
        )}
    if name == "rce":
        return {"name": name, "log_zero": session.floating(
            "RCE log_zero（必须为负数）", default=float(current.get("log_zero", -4.0)),
            maximum=0.0, maximum_inclusive=False,
        )}
    if name == "apl":
        passive_current = current.get("passive", {})
        passive_name = session.choose(
            "选择 APL passive loss", [("MAE", "mae"), ("RCE", "rce")],
            default=str(passive_current.get("name", "rce")).lower(),
        )
        passive: dict[str, Any] = {"name": passive_name}
        if passive_name == "mae":
            passive["scale"] = session.floating(
                "Passive MAE scale", default=float(passive_current.get("scale", 2.0)),
                minimum=0.0, minimum_inclusive=False,
            )
        else:
            passive["log_zero"] = session.floating(
                "Passive RCE log_zero", default=float(passive_current.get("log_zero", -4.0)),
                maximum=0.0, maximum_inclusive=False,
            )
        active_current = current.get("active", {})
        alpha = session.floating(
            "APL alpha", default=float(current.get("alpha", 1.0)), minimum=0.0,
            minimum_inclusive=False,
        )
        beta = session.floating(
            "APL beta", default=float(current.get("beta", 1.0)), minimum=0.0,
            minimum_inclusive=False,
        )
        return {
            "name": "apl",
            "alpha": alpha,
            "beta": beta,
            "active": {
                "name": "nce",
                "eps": session.floating(
                    "Active NCE eps", default=float(active_current.get("eps", 1e-8)),
                    minimum=0.0, minimum_inclusive=False,
                ),
            },
            "passive": passive,
        }
    return {"name": name}


def _parse_seeds(value: str) -> list[int]:
    pieces = [item for item in re.split(r"[\s,]+", value.strip()) if item]
    seeds = [int(item) for item in pieces]
    if not seeds or any(item < 0 for item in seeds):
        raise ValueError
    return seeds


def _prompt_scheduler(
    session: PromptSession,
    current: Mapping[str, Any] | None,
    epochs: int,
) -> dict[str, Any]:
    values = dict(current or {"name": "none"})
    name = session.choose(
        "选择学习率 scheduler",
        [("不使用", "none"), ("Cosine", "cosine"), ("MultiStep", "multistep")],
        default=str(values.get("name", "none")).lower(),
    )
    if name == "cosine":
        return {
            "name": name,
            "t_max": session.integer("Cosine T_max", default=int(values.get("t_max", epochs)), minimum=1),
            "eta_min": session.floating(
                "Cosine eta_min", default=float(values.get("eta_min", 0.0)), minimum=0.0
            ),
        }
    if name == "multistep":
        default_milestones = " ".join(str(item) for item in values.get("milestones", [max(1, epochs // 2)]))
        while True:
            raw = session.text("Milestones（空格或逗号分隔）", default=default_milestones, required=True)
            try:
                milestones = _parse_seeds(raw)
            except ValueError:
                session.write("Milestones 必须是一个或多个非负整数。")
                continue
            if milestones != sorted(set(milestones)):
                session.write("Milestones 必须严格递增且不能重复。")
                continue
            break
        return {
            "name": name,
            "milestones": milestones,
            "gamma": session.floating(
                "MultiStep gamma", default=float(values.get("gamma", 0.1)),
                minimum=0.0, minimum_inclusive=False,
            ),
        }
    return {"name": "none"}


def _prompt_noise(
    session: PromptSession,
    current: Mapping[str, Any] | None,
    default_seed: int,
) -> dict[str, Any] | None:
    values = dict(current or {})
    if values.get("manifest"):
        current_mode = "external"
    elif str(values.get("name", "clean")).lower() in {"symmetric", "pairflip"}:
        current_mode = "generated"
    else:
        current_mode = "clean"
    mode = session.choose(
        "选择标签模式",
        [("干净标签", "clean"), ("运行时生成噪声", "generated"),
         ("使用已有 Noise Manifest", "external")],
        default=current_mode,
    )
    if mode == "clean":
        return None
    filename = session.text(
        "运行目录中的 manifest 文件名",
        default=str(values.get("manifest_filename", "noise_manifest.npz")),
        required=True,
        validator=lambda item: Path(item).name == item,
        error="请输入不含目录的文件名。",
    )
    if mode == "external":
        default_manifest = Path(str(values["manifest"])) if values.get("manifest") else None
        manifest = session.path(
            "Noise Manifest 文件",
            default=default_manifest,
            required=True,
            must_exist=True,
            file_only=True,
        )
        assert manifest is not None
        return {"manifest": str(manifest), "manifest_filename": filename}
    name = session.choose(
        "选择噪声类型",
        [("Symmetric", "symmetric"), ("Pairflip", "pairflip")],
        default=str(values.get("name", "symmetric")).lower(),
    )
    return {
        "name": name,
        "rate": session.floating(
            "噪声率", default=float(values.get("rate", 0.4)), minimum=0.0, maximum=1.0
        ),
        "seed": session.integer(
            "噪声随机种子", default=int(values.get("seed", default_seed)), minimum=0
        ),
        "manifest_filename": filename,
    }


def prompt_training_selection(session: PromptSession, *, clean: bool) -> TrainingSelection | None:
    """Build an in-memory training configuration through a terminal wizard."""

    source, loaded = _choose_template(session, clean=clean)
    config = deepcopy(loaded)
    data = config.setdefault("data", {})
    dataset = session.choose(
        "选择数据集", [("CIFAR-10", "cifar10"), ("CIFAR-100", "cifar100")],
        default=str(data.get("name", "cifar10")).lower(),
    )
    data["name"] = dataset
    root = session.path(
        "数据目录", default=Path(str(data.get("root", f"data/{dataset}"))),
        required=True, must_exist=True, directory_only=True,
    )
    assert root is not None
    data["root"] = str(root)

    if not clean:
        current_noise = config.get("noise")
        selected_noise = _prompt_noise(
            session,
            current_noise if isinstance(current_noise, Mapping) else None,
            int(config.get("seed", 1)),
        )
        if selected_noise is None:
            config.pop("noise", None)
        else:
            config["noise"] = selected_noise

    model = config.setdefault("model", {})
    if clean:
        model_name = session.choose(
            "选择模型",
            [("TinyCNN", "tiny_cnn"), ("CIFAR ResNet-18", "resnet18"),
             ("PreActResNet-18", "preact_resnet18")],
            default=str(model.get("name", "preact_resnet18")).lower(),
        )
        if model_name == "tiny_cnn":
            config["model"] = {"name": model_name, "width": session.integer(
                "TinyCNN width", default=int(model.get("width", 64)), minimum=1
            )}
        else:
            config["model"] = {"name": model_name, "base_width": session.integer(
                "ResNet base width", default=int(model.get("base_width", 64)), minimum=1
            )}
    else:
        config["model"] = {"name": "tiny_cnn", "width": session.integer(
            "TinyCNN width", default=int(model.get("width", 64)), minimum=1
        )}

    config["loss"] = _prompt_loss(session, config.get("loss", {"name": "ce"}))
    optimizer = config.setdefault("optimizer", {})
    optimizer_name = session.choose(
        "选择优化器", [("SGD", "sgd"), ("AdamW", "adamw")],
        default=str(optimizer.get("name", "adamw")).lower(),
    )
    new_optimizer: dict[str, Any] = {
        "name": optimizer_name,
        "lr": session.floating(
            "Learning rate", default=float(optimizer.get("lr", 0.001)),
            minimum=0.0, minimum_inclusive=False,
        ),
        "weight_decay": session.floating(
            "Weight decay", default=float(optimizer.get("weight_decay", 0.0)), minimum=0.0
        ),
    }
    if optimizer_name == "sgd":
        new_optimizer["momentum"] = session.floating(
            "SGD momentum", default=float(optimizer.get("momentum", 0.9)), minimum=0.0, maximum=1.0
        )
        if clean:
            new_optimizer["nesterov"] = session.confirm(
                "启用 Nesterov", default=bool(optimizer.get("nesterov", False))
            )
    config["optimizer"] = new_optimizer

    loader = config.setdefault("loader", {})
    loader["batch_size"] = session.integer(
        "Batch size", default=int(loader.get("batch_size", 128)), minimum=1
    )
    trainer = config.setdefault("trainer", {})
    trainer["epochs"] = session.integer(
        "训练 epochs", default=int(trainer.get("epochs", 1)), minimum=1
    )
    if clean:
        config["scheduler"] = _prompt_scheduler(
            session, config.get("scheduler"), int(trainer["epochs"])
        )
    import torch

    devices = [("Auto", "auto"), ("CPU", "cpu")]
    if torch.cuda.is_available():
        devices.append((f"CUDA ({torch.cuda.get_device_name(0)})", "cuda"))
    current_device = str(trainer.get("device", "auto")).lower()
    trainer["device"] = session.choose(
        "选择设备", devices,
        default=current_device if current_device in {value for _, value in devices} else "auto",
    )
    config["seed"] = session.integer("随机种子", default=int(config.get("seed", 1)), minimum=0)

    modes = [("新实验", "new"), ("从 checkpoint 恢复", "resume")]
    if clean:
        modes.append(("多随机种子实验", "seeds"))
    mode = session.choose("选择运行模式", modes, default="new")
    output_dir: Path | None = None
    resume: Path | None = None
    seeds: list[int] | None = None
    if mode == "resume":
        resume = session.path("Checkpoint 路径", required=True, must_exist=True, file_only=True)
    elif mode == "seeds":
        while True:
            raw = session.text("随机种子列表（空格或逗号分隔）", default="1 2 3", required=True)
            try:
                seeds = _parse_seeds(raw)
            except ValueError:
                session.write("请输入一个或多个非负整数。")
                continue
            break
        default_output = Path(str(config.get("output_root", "artifacts/runs"))) / "seed-suite"
        output_dir = session.path("输出目录", default=default_output)
    else:
        output_dir = session.path("输出目录（留空则自动创建）")

    import yaml

    session.write("\n即将运行的配置：")
    session.write(yaml.safe_dump(config, allow_unicode=True, sort_keys=False).rstrip())
    session.write(f"配置模板：{source}")
    session.write(f"运行模式：{mode}")
    if output_dir is not None:
        session.write(f"输出目录：{output_dir}")
    if resume is not None:
        session.write(f"Checkpoint：{resume}")
    if seeds is not None:
        session.write(f"Seeds：{seeds}")
    if not session.confirm("确认开始运行", default=False):
        session.write("已取消。")
        return None
    return TrainingSelection(config, source, output_dir, resume, seeds)


__all__ = [
    "PromptCancelled",
    "PromptSession",
    "TrainingSelection",
    "command_arguments",
    "prompt_training_selection",
    "repository_root",
]

