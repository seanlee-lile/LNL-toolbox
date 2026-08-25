from __future__ import annotations

import unittest
from pathlib import Path

import torch

from lnl_toolbox.scratch import ScratchContext
from lnl_toolbox.scratch.blocks.data import (
    assign_data_roles,
    build_loaders,
    build_noise_manifest,
    build_prepared_data,
    configure_loader,
    configure_preprocessing,
    configure_views,
    create_dataset_split,
    inspect_dataset_semantics,
    load_dataset,
    select_label_source,
    apply_noise,
)
from lnl_toolbox.scratch.data_runtime import ScratchSample, ScratchSplit


def _same_index_source() -> dict[str, ScratchSplit]:
    train = ScratchSplit(
        "shared", "train",
        (ScratchSample(torch.zeros(3, 4, 4), 0, 0, 0),
         ScratchSample(torch.ones(3, 4, 4), 1, 1, 1)),
        2,
    )
    test = ScratchSplit(
        "shared", "test",
        (ScratchSample(torch.full((3, 4, 4), 2.0), 0, 0, 0),),
        2,
    )
    return {"train": train, "test": test}


def _run_blocks(source: dict[str, ScratchSplit], *, validation_size: int = 1,
                roles: list[str] | None = None) -> ScratchContext:
    ctx = ScratchContext(seed=3)
    load_dataset(ctx, "custom", options={"source": source})
    inspect_dataset_semantics(ctx)
    create_dataset_split(ctx, validation_size=validation_size, split_seed=7)
    select_label_source(ctx, validation="observed")
    apply_noise(ctx, name="pairflip", rate=1.0, seed=5)
    build_noise_manifest(ctx)
    configure_preprocessing(ctx, preprocessing="tensor_only")
    configure_views(ctx, views=["weak", "strong"])
    assign_data_roles(ctx, roles=roles or ["train", "noisy_validation", "test"])
    configure_loader(ctx, batch_size=1, num_workers=0)
    build_prepared_data(ctx)
    build_loaders(ctx)
    return ctx


class DataBlockExecutionTest(unittest.TestCase):
    def test_symbolic_transition_matrix_is_resolved_and_exposed(self) -> None:
        train = ScratchSplit("cifar10", "train", tuple(
            ScratchSample(torch.zeros(3, 4, 4), i, i, i) for i in range(10)
        ), 10)
        test = ScratchSplit("cifar10", "test", tuple(
            ScratchSample(torch.ones(3, 4, 4), i, i, i) for i in range(10)
        ), 10)
        ctx = ScratchContext(seed=1)
        load_dataset(ctx, "custom", options={"source": {"train": train, "test": test}})
        create_dataset_split(ctx, validation_size=0)
        select_label_source(ctx)
        apply_noise(ctx, name="class_conditional", rate=0.4, seed=1,
                    options={"transition_matrix": "formal_loss_correction"})
        build_noise_manifest(ctx)
        matrix = torch.as_tensor(ctx["noise_manifest"].transition_matrix)
        self.assertEqual(tuple(matrix.shape), (10, 10))
        self.assertAlmostEqual(float(matrix[2, 0]), 0.4)
        self.assertAlmostEqual(float(matrix[2, 2]), 0.6)

    def test_train_test_namespace_is_independent_even_when_indices_repeat(self) -> None:
        ctx = _run_blocks(_same_index_source())
        prepared = ctx["prepared_data"]
        self.assertEqual(prepared.datasets["train"].split.namespace, ("shared", "train", "1"))
        self.assertEqual(prepared.datasets["test"].split.namespace, ("shared", "test", "1"))
        self.assertEqual(prepared.datasets["test"].indices, [0])

    def test_blocks_emit_real_outputs_and_canonical_batch(self) -> None:
        ctx = _run_blocks(_same_index_source())
        self.assertIsInstance(ctx["train_source"], ScratchSplit)
        self.assertIsInstance(ctx["train_split"], ScratchSplit)
        self.assertIsInstance(ctx["noisy_train_split"], ScratchSplit)
        self.assertIn("role_datasets", ctx)
        batch = next(iter(ctx["train_loader"]))
        self.assertEqual(set(batch), {"inputs", "targets", "indices", "clean_targets", "views", "strong_input"})
        self.assertTrue(torch.equal(batch["targets"], 1 - batch["clean_targets"]))

    def test_validation_never_falls_back_to_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation"):
            _run_blocks(_same_index_source(), validation_size=0,
                        roles=["train", "noisy_validation", "test"])

    def test_preprocessing_and_views_change_input_without_metadata_changes(self) -> None:
        ctx = _run_blocks(_same_index_source())
        row = ctx["prepared_data"].dataset_for("train")[0]
        self.assertEqual(row["indices"], ctx["prepared_data"].dataset_for("train").indices[0])
        self.assertEqual(row["targets"], 1 - row["clean_targets"])
        self.assertIn("strong", row["views"])
        self.assertEqual(tuple(row["inputs"].shape), (3, 4, 4))

    def test_convenience_blocks_are_retained_only_for_existing_distinct_references(self) -> None:
        root = Path(__file__).parents[1]
        data_source = (root / "blocks" / "data.py").read_text(encoding="utf-8")
        examples = list((root / "recipes" / "examples").glob("*.yaml"))
        example_text = "\n".join(path.read_text(encoding="utf-8") for path in examples)
        for block_id in ("select_dataset", "configure_noise", "create_loader", "load_synthetic"):
            self.assertIn(block_id, data_source)
            self.assertIn(block_id, example_text)


if __name__ == "__main__":
    unittest.main()
