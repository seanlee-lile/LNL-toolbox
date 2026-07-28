from __future__ import annotations

"""Create a split-aware CIFAR noise manifest for an external-manifest run."""

import argparse
from pathlib import Path

from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.torch_cifar import train_validation_split
from lnl_toolbox.noise.split_manifest import (
    generate_split_symmetric_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-size", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = load_cifar10 if args.dataset == "cifar10" else load_cifar100
    dataset = loader(args.data_root, "train")
    train_indices, validation_indices = train_validation_split(
        dataset.labels,
        args.validation_size,
        args.seed,
        strategy="classwise_legacy",
        rng="numpy_legacy",
    )
    manifest = generate_split_symmetric_manifest(
        dataset.labels,
        (train_indices, validation_indices),
        num_classes=10 if args.dataset == "cifar10" else 100,
        rate=args.rate,
        seed=args.seed,
        dataset=args.dataset,
    )
    destination = Path(args.output)
    manifest.save(destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
