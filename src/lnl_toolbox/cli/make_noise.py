from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np

from lnl_toolbox.cli import PromptCancelled, PromptSession, command_arguments
from lnl_toolbox.noise import generate_pairflip, generate_symmetric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a reusable label-noise manifest")
    parser.add_argument("labels", type=Path, help="Path to a one-dimensional NumPy label array")
    parser.add_argument("output", type=Path, help="Output .npz manifest path")
    parser.add_argument("--kind", choices=("symmetric", "pairflip"), default="symmetric")
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", default="unknown")
    return parser


def _execute(
    labels_path: Path,
    output: Path,
    kind: str,
    rate: float,
    classes: int,
    seed: int,
    dataset: str,
) -> None:
    labels = np.load(labels_path)
    generator = generate_symmetric if kind == "symmetric" else generate_pairflip
    manifest = generator(labels, classes, rate, seed, dataset)
    manifest.save(output)
    print(f"saved {output}; realized_rate={manifest.realized_rate:.4f}")


def _interactive(session: PromptSession) -> tuple[Path, Path, str, float, int, int, str] | None:
    while True:
        labels_path = session.path("标签 .npy 文件", required=True, must_exist=True, file_only=True)
        assert labels_path is not None
        if labels_path.suffix.lower() == ".npy":
            break
        session.write("标签文件必须使用 .npy 后缀。")
    kind = session.choose(
        "选择噪声类型", [("Symmetric", "symmetric"), ("Pairflip", "pairflip")],
        default="symmetric",
    )
    rate = session.floating("噪声率", default=0.2, minimum=0.0, maximum=1.0)
    classes = session.integer("类别数", default=10, minimum=2)
    seed = session.integer("随机种子", default=1, minimum=0)
    dataset = session.text("数据集名称", default="unknown", required=True)
    default_output = labels_path.with_name(f"{labels_path.stem}-{kind}-{rate:g}.npz")
    output = session.path("输出 manifest", default=default_output, required=True)
    assert output is not None
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")
    preview = {
        "labels": str(labels_path), "output": str(output), "kind": kind, "rate": rate,
        "classes": classes, "seed": seed, "dataset": dataset,
    }
    session.write("\n即将生成：")
    session.write(json.dumps(preview, ensure_ascii=False, indent=2))
    if not session.confirm("确认生成 manifest", default=False):
        session.write("已取消。")
        return None
    return labels_path, output, kind, rate, classes, seed, dataset


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
            _execute(
                args.labels, args.output, args.kind, args.rate, args.classes, args.seed, args.dataset
            )
    except PromptCancelled:
        active_session.write("\n已取消。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

