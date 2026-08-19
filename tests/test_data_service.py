from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lnl_toolbox.data import DataRequirements, DataRole, DataSpec
from lnl_toolbox.training.checkpoint import atomic_save, read_checkpoint
from lnl_toolbox.training.data_service import (
    DATASETS,
    prepare_experiment_data,
)


def _config() -> dict:
    return {
        "seed": 9,
        "data": {
            "name": "synthetic_multiclass",
            "num_classes": 3,
            "dimension": 4,
            "train_size": 30,
            "validation_size": 12,
            "test_size": 12,
        },
        "noise": {"name": "clean", "rate": 0.0, "seed": 9},
        "loader": {"batch_size": 6, "num_workers": 0, "drop_last": False},
    }


class DataServiceTest(unittest.TestCase):
    def test_registry_alias_and_unknown_dataset(self) -> None:
        self.assertEqual(DATASETS.get("cifar-10").name, "cifar10")
        self.assertEqual(DATASETS.get("fashionmnist").name, "fashion_mnist")
        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            DATASETS.get("cifar11")

    def test_spec_preserves_legacy_options(self) -> None:
        spec = DataSpec.from_mapping({"name": "cifar-10", "root": "data", "num_val": 7})
        self.assertEqual(spec.name, "cifar_10")
        self.assertEqual(spec.options["num_val"], 7)

    def test_roles_batch_schema_indices_views_and_loader_seed(self) -> None:
        requirements = DataRequirements(
            roles=frozenset({
                DataRole.TRAIN,
                DataRole.TRAIN_EVAL,
                DataRole.CLEAN_VALIDATION,
                DataRole.TEST,
            }),
            views=("weak", "strong"),
        )
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_experiment_data(
                _config(), requirements=requirements, run_dir=directory, seed=9
            )
            batch = next(iter(prepared.loader(DataRole.TRAIN, epoch=2)))
            self.assertEqual(
                set(batch), {"input", "target", "index", "views", "strong_input"}
            )
            self.assertNotIn("clean_target", batch)
            self.assertEqual(set(batch["views"]), {"weak", "strong"})

            first = torch.cat([
                value["index"] for value in prepared.loader(DataRole.TRAIN, epoch=4)
            ])
            repeated = torch.cat([
                value["index"] for value in prepared.loader(DataRole.TRAIN, epoch=4)
            ])
            another = torch.cat([
                value["index"] for value in prepared.loader(DataRole.TRAIN, epoch=5)
            ])
            self.assertTrue(torch.equal(first, repeated))
            self.assertFalse(torch.equal(first, another))
            self.assertEqual(set(first.tolist()), set(prepared.train_indices.tolist()))

            chosen = prepared.train_indices[::2]
            probabilities = {int(index): float(offset) for offset, index in enumerate(chosen)}
            dynamic = prepared.dynamic_dataset(
                chosen, overlays={"clean_probability": probabilities}
            )
            self.assertEqual(dynamic.indices.tolist(), chosen.tolist())
            self.assertIn("clean_probability", dynamic[0])

    def test_manifest_checkpoint_round_trip_and_tamper_failure(self) -> None:
        requirements = DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.TEST}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = prepare_experiment_data(
                _config(), requirements=requirements, run_dir=root, seed=9
            )
            atomic_save({"model": {}}, root / "last.pt")
            payload = read_checkpoint(root / "last.pt")
            self.assertEqual(payload["data"], prepared.state_dict())
            prepare_experiment_data(
                _config(), requirements=requirements, run_dir=root, seed=9,
                checkpoint_payload=payload,
            )

            manifest_path = root / "data_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["loader"]["batch_size"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                read_checkpoint(root / "last.pt")

    def test_checkpoint_state_rejects_different_data(self) -> None:
        requirements = DataRequirements(roles=frozenset({DataRole.TRAIN, DataRole.TEST}))
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            original = prepare_experiment_data(
                _config(), requirements=requirements, run_dir=first, seed=9
            )
            changed = _config()
            changed["data"]["train_size"] = 33
            current = prepare_experiment_data(
                changed, requirements=requirements, run_dir=second, seed=9
            )
            with self.assertRaisesRegex(ValueError, "data identity mismatch"):
                current.load_state_dict(original.state_dict())

    def test_all_experiment_runners_use_only_unified_entry(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "lnl_toolbox" / "training"
        runners = (
            "experiment.py", "multi_model_experiment.py", "binary_experiment.py",
            "importance_reweighting_experiment.py", "cwd_experiment.py",
            "dual_t_experiment.py", "dual_t_evidence_experiment.py",
            "instance_transition_experiment.py", "t_revision_experiment.py",
            "volminnet_experiment.py", "upm_experiment.py", "pcse_experiment.py",
            "mc_ldce_experiment.py", "cal_experiment.py", "volmin_experiment.py",
            "coteaching_experiment.py", "cnlcu_experiment.py", "lend_experiment.py",
            "fine_experiment.py", "dld_experiment.py", "ca2c_experiment.py",
            "dividemix_experiment.py", "l2rw_experiment.py",
        )
        forbidden = (
            "from lnl_toolbox.data.cifar",
            "from lnl_toolbox.data.torch_cifar",
            "DataLoader(",
            "prepare_noisy_classification(",
        )
        for filename in runners:
            source = (root / filename).read_text(encoding="utf-8")
            self.assertIn("prepare_experiment_data", source, filename)
            for pattern in forbidden:
                self.assertNotIn(pattern, source, f"{filename}: {pattern}")


if __name__ == "__main__":
    unittest.main()
