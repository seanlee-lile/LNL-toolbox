from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from lnl_toolbox.cli import PromptCancelled, PromptSession, command_arguments, repository_root
from lnl_toolbox.data import load_cifar10, load_cifar100, summarize_cifar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate local CIFAR pickle files")
    parser.add_argument("dataset", choices=("cifar10", "cifar100"))
    parser.add_argument("--root", type=Path, default=None, help="Dataset directory; uses package data by default")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    return parser


def _execute(dataset: str, root: Path | None, split: str) -> None:
    loader = load_cifar10 if dataset == "cifar10" else load_cifar100
    splits = ("train", "test") if split == "all" else (split,)
    summaries = [summarize_cifar(loader(root, item)) for item in splits]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def _interactive(session: PromptSession) -> tuple[str, Path, str] | None:
    dataset = session.choose(
        "选择数据集", [("CIFAR-10", "cifar10"), ("CIFAR-100", "cifar100")],
        default="cifar10",
    )
    default_root = repository_root() / "data" / dataset
    root = session.path(
        "数据目录", default=default_root, required=True, must_exist=True, directory_only=True
    )
    assert root is not None
    split = session.choose(
        "选择数据划分", [("全部", "all"), ("训练集", "train"), ("测试集", "test")],
        default="all",
    )
    session.write("\n即将检查：")
    session.write(json.dumps({"dataset": dataset, "root": str(root), "split": split},
                             ensure_ascii=False, indent=2))
    if not session.confirm("确认开始检查", default=False):
        session.write("已取消。")
        return None
    return dataset, root, split


def main(
    argv: Sequence[str] | None = None,
    session: PromptSession | None = None,
) -> int:
    arguments = command_arguments(argv)
    active_session = session or PromptSession()
    try:
        if not arguments:
            selection = _interactive(active_session)
            if selection is None:
                return 0
            _execute(*selection)
        else:
            args = build_parser().parse_args(arguments)
            _execute(args.dataset, args.root, args.split)
    except PromptCancelled:
        active_session.write("\n已取消。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

