import tempfile
import unittest
from pathlib import Path

import torch

from lnl_toolbox.scratch.data_runtime import (
    ScratchSample,
    ScratchSplit,
    collate_scratch_batch,
    materialize_plan,
)


def _source(train_clean=True, test_clean=True, train_size=8, test_size=3):
    train = tuple(
        ScratchSample(torch.tensor([float(i), float(i + 1)]), i, i % 2,
                      i % 2 if train_clean else None)
        for i in range(train_size)
    )
    test = tuple(
        ScratchSample(torch.tensor([100.0 + i, 101.0 + i]), 100 + i, i % 2,
                      i % 2 if test_clean else None)
        for i in range(test_size)
    )
    return {
        "train": ScratchSplit("custom", "train", train, 2),
        "test": ScratchSplit("custom", "test", test, 2),
    }


def _plan(source, **overrides):
    plan = {
        "seed": 7,
        "data": {"name": "custom", "source": source},
        "semantics": {"num_classes": 2},
        "split": {"validation_size": 0, "strategy": "random", "seed": 9},
        "labels": {"train": "observed", "validation": "clean", "test": "clean", "trusted": "clean"},
        "noise": {"name": "none", "rate": 0.0, "seed": 3},
        "preprocessing": {"name": "tensor_only"},
        "views": ["weak", "strong"],
        "roles": ["train", "train_eval", "test"],
        "loader": {"batch_size": 2, "num_workers": 0},
    }
    for key, value in overrides.items():
        plan[key] = value
    return plan


class ScratchDataRuntimeTest(unittest.TestCase):
    def test_test_role_uses_explicit_test_source_and_canonical_batch(self):
        prepared, _ = materialize_plan(_plan(_source()), {})
        self.assertEqual(prepared.dataset_for("test").indices, [100, 101, 102])
        self.assertNotEqual(set(prepared.dataset_for("train").indices), set(prepared.dataset_for("test").indices))
        batch = next(iter(prepared.loader("train", batch_size=2, shuffle=False)))
        self.assertEqual(set(batch), {"inputs", "targets", "indices", "clean_targets", "views", "strong_input"})
        self.assertTrue(torch.equal(batch["targets"], batch["clean_targets"]))

    def test_split_preserves_source_indices_without_offset(self):
        plan = _plan(_source(), split={"validation_size": 3, "strategy": "random", "seed": 9},
                     roles=["train", "noisy_validation", "test"])
        prepared, _ = materialize_plan(plan, {})
        train_ids = set(prepared.dataset_for("train").indices)
        val_ids = set(prepared.dataset_for("noisy_validation").indices)
        self.assertEqual(train_ids | val_ids, set(range(8)))
        self.assertEqual(train_ids & val_ids, set())
        self.assertTrue(all(index < 8 for index in train_ids | val_ids))

    def test_noise_changes_observed_only_and_train_eval_stays_observed(self):
        plan = _plan(_source(), noise={"name": "pairflip", "rate": 1.0, "seed": 2},
                     roles=["train", "train_eval", "test"])
        prepared, manifest = materialize_plan(plan, {})
        train = prepared.dataset_for("train")
        train_eval = prepared.dataset_for("train_eval")
        self.assertTrue(all(row["targets"] == 1 - row["clean_targets"] for row in train))
        self.assertEqual([row["targets"] for row in train], [row["targets"] for row in train_eval])
        self.assertEqual(manifest.clean_targets.tolist(), [row["clean_targets"] for row in train])

    def test_trusted_subset_is_clean_and_excluded_from_learning_train(self):
        plan = _plan(_source(train_size=10),
                     data={"name": "custom", "source": _source(train_size=10), "num_clean": 3, "trusted_seed": 1234},
                     roles=["train", "trusted_validation", "test"],
                     noise={"name": "symmetric", "rate": 0.5, "seed": 4})
        prepared, _ = materialize_plan(plan, {})
        trusted_ids = set(prepared.dataset_for("trusted_validation").indices)
        train_ids = set(prepared.dataset_for("train").indices)
        self.assertEqual(len(trusted_ids), 3)
        self.assertTrue(trusted_ids.isdisjoint(train_ids))
        self.assertTrue(all(row["clean_targets"] is not None for row in prepared.dataset_for("trusted_validation")))

    def test_missing_clean_labels_cannot_drive_synthetic_noise_or_clean_role(self):
        with self.assertRaisesRegex(ValueError, "complete clean_target"):
            materialize_plan(_plan(_source(train_clean=False),
                                   noise={"name": "symmetric", "rate": 0.2, "seed": 1}), {})
        with self.assertRaisesRegex(ValueError, "clean_validation"):
            materialize_plan(_plan(_source(train_clean=False, test_clean=False),
                                   roles=["train", "clean_validation", "test"]), {})

    def test_partial_clean_batch_returns_none_instead_of_filling_a_sentinel(self):
        rows = [
            {"inputs": torch.tensor([1.0]), "targets": 1, "indices": 4, "clean_targets": 1},
            {"inputs": torch.tensor([2.0]), "targets": 0, "indices": 8, "clean_targets": None},
        ]
        batch = collate_scratch_batch(rows)
        self.assertIsNone(batch["clean_targets"])
        self.assertFalse(torch.any(batch["targets"] < 0))

    def test_external_manifest_can_supply_verified_clean_targets(self):
        source = _source(train_clean=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "noise.pt"
            torch.save({"clean_label_train": torch.tensor([i % 2 for i in range(8)]),
                        "noise_label_train": torch.tensor([(i + 1) % 2 for i in range(8)])}, path)
            plan = _plan(source, data={"name": "custom", "source": source, "num_clean": 2, "trusted_seed": 1234},
                         noise={"name": "external", "path": str(path)},
                         roles=["train", "trusted_validation", "test"])
            prepared, _ = materialize_plan(plan, {})
            self.assertTrue(all(row["clean_targets"] is not None for row in prepared.dataset_for("trusted_validation")))
            self.assertTrue(all(row["targets"] == 1 - row["clean_targets"] for row in prepared.dataset_for("train")))


if __name__ == "__main__":
    unittest.main()
