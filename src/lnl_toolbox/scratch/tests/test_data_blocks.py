from __future__ import annotations

import unittest
from pathlib import Path

import torch

from lnl_toolbox.scratch import ScratchContext
from lnl_toolbox.scratch.registry import describe_block
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
from lnl_toolbox.scratch.data_runtime import ScratchSample, ScratchSplit, apply_noise_to_split


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
    def test_data_block_metadata_exposes_real_slots(self) -> None:
        expected = {
            "load_dataset": ((), ("save_as", "data_spec", "train_source", "validation_source", "test_source", "num_classes")),
            "create_dataset_split": (("train_source",), ("train_split", "validation_split")),
            "apply_noise": (("train_split",), ("noisy_train_split", "noise_state")),
            "build_noise_manifest": (("noise_state",), ("noise_manifest",)),
            "configure_preprocessing": (("train_split",), ("preprocessing_transform", "preprocessed_datasets")),
            "configure_views": (("preprocessed_datasets", "preprocessing_transform"), ("view_transforms", "view_datasets")),
            "assign_data_roles": (("noisy_train_split", "validation_split", "test_source", "preprocessing_transform", "view_transforms"), ("role_datasets",)),
            "configure_loader": ((), ("loader_spec",)),
            "build_loaders": (("role_datasets", "loader_spec"), ("train_loader", "train_eval_loader", "validation_loader", "trusted_loader", "test_loader")),
        }
        for block_id, (requires, provides) in expected.items():
            metadata = describe_block(block_id)
            self.assertEqual(tuple(metadata["requires"]), requires, block_id)
            self.assertEqual(tuple(metadata["provides"]), provides, block_id)

    def test_manifest_build_does_not_rematerialize_noise(self) -> None:
        ctx = ScratchContext(seed=3)
        load_dataset(ctx, "custom", options={"source": _same_index_source()})
        create_dataset_split(ctx, validation_size=1)
        apply_noise(ctx, name="pairflip", rate=1.0, seed=5)
        noisy_before = ctx["noisy_train_split"]
        targets_before = tuple(sample.observed_target for sample in noisy_before.samples)
        build_noise_manifest(ctx)
        self.assertIs(ctx["noisy_train_split"], noisy_before)
        self.assertEqual(tuple(sample.observed_target for sample in noisy_before.samples), targets_before)

    def test_roles_consume_configured_transform_objects(self) -> None:
        ctx = _run_blocks(_same_index_source())
        train_dataset = ctx["role_datasets"]["train"]
        self.assertIs(train_dataset.transform, ctx["preprocessing_transform"])
        self.assertIs(train_dataset.views["weak"], ctx["view_transforms"]["weak"])

    def test_instance_dependent_noise_requires_scores_and_is_feature_conditioned(self) -> None:
        train = ScratchSplit("toy", "train", tuple(
            ScratchSample(torch.tensor([float(i), 0.0]), i, i % 3, i % 3) for i in range(6)
        ), 3)
        with self.assertRaisesRegex(ValueError, "class_scores"):
            apply_noise_to_split(train, {"name": "instance_dependent", "rate": 0.5, "seed": 1})
        scores = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        noisy, manifest, _ = apply_noise_to_split(
            train, {"name": "instance_dependent", "rate": 0.5, "seed": 1, "class_scores": scores}
        )
        self.assertEqual(tuple(manifest.per_sample_transition.shape), (6, 3))
        self.assertFalse(torch.equal(manifest.per_sample_transition[0], manifest.per_sample_transition[3]))
        self.assertTrue(any(a.observed_target != b.observed_target for a, b in zip(train.samples, noisy.samples)))

    def test_pdl_noise_uses_input_features_and_publishes_transitions(self) -> None:
        def make(value: float) -> ScratchSplit:
            return ScratchSplit("toy", "train", tuple(
                ScratchSample(torch.tensor([value, float(i + 1), float((i + 1) ** 2)]), i, i % 3, i % 3) for i in range(6)
            ), 3)
        first, manifest, _ = apply_noise_to_split(make(0.0), {"name": "pdl", "rate": 0.5, "seed": 7})
        second, manifest_two, _ = apply_noise_to_split(make(3.0), {"name": "pdl", "rate": 0.5, "seed": 7})
        self.assertEqual(tuple(manifest.per_sample_transition.shape), (6, 3))
        self.assertFalse(torch.equal(manifest.per_sample_transition, manifest_two.per_sample_transition))
        self.assertEqual(len(first.samples), len(second.samples))

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
