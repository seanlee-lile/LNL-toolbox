from __future__ import annotations

import argparse

import numpy as np

from lnl_toolbox.noise import generate_pairflip, generate_symmetric


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reusable label-noise manifest")
    parser.add_argument("labels", help="Path to a one-dimensional NumPy label array")
    parser.add_argument("output", help="Output .npz manifest path")
    parser.add_argument("--kind", choices=("symmetric", "pairflip"), default="symmetric")
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", default="unknown")
    args = parser.parse_args()

    labels = np.load(args.labels)
    generator = generate_symmetric if args.kind == "symmetric" else generate_pairflip
    manifest = generator(labels, args.classes, args.rate, args.seed, args.dataset)
    manifest.save(args.output)
    print(f"saved {args.output}; realized_rate={manifest.realized_rate:.4f}")


if __name__ == "__main__":
    main()

