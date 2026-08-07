from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset

from lnl_toolbox.data.trusted import TrustedSupervisionManifest, TrustedValidationProvider


class _Dataset(Dataset):
    def __len__(self): return 2
    def __getitem__(self, item):
        return {"input": torch.tensor([float(item)]), "target": 99, "index": (8, 3)[item]}


class TrustedSupervisionTest(unittest.TestCase):
    def test_manifest_roundtrip_and_provider_replaces_only_explicit_targets(self) -> None:
        manifest = TrustedSupervisionManifest(
            np.array([3, 8]), np.array([1, 0]), "fixture", "trusted_validation",
            "audited_manifest", True, {"reviewer": "unit-test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trusted.npz"
            manifest.save(path)
            loaded = TrustedSupervisionManifest.load(path)
        provider = TrustedValidationProvider(_Dataset(), loaded)
        batch = next(iter(provider.loader(batch_size=2, shuffle=False, seed=1)))
        self.assertEqual(batch["target"].tolist(), [0, 1])
        self.assertEqual(provider.fingerprint, manifest.fingerprint)

    def test_ordinary_validation_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicitly audited"):
            TrustedSupervisionManifest(
                np.array([0]), np.array([0]), "fixture", "trusted_validation",
                "validation", True,
            )

    def test_test_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "trusted_validation"):
            TrustedSupervisionManifest(
                np.array([0]), np.array([0]), "fixture", "test",
                "audited_manifest", True,
            )


if __name__ == "__main__": unittest.main()
