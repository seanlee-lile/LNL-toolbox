from __future__ import annotations

import argparse
import json
from pathlib import Path

from lnl_toolbox.data import load_cifar10, load_cifar100, summarize_cifar


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local CIFAR pickle files")
    parser.add_argument("dataset", choices=("cifar10", "cifar100"))
    parser.add_argument("--root", type=Path, default=None, help="Dataset directory; uses package data by default")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    args = parser.parse_args()

    loader = load_cifar10 if args.dataset == "cifar10" else load_cifar100
    splits = ("train", "test") if args.split == "all" else (args.split,)
    summaries = [summarize_cifar(loader(args.root, split)) for split in splits]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

