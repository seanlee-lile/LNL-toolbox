from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts.check_paper_registration import check_repository


ROOT = Path(__file__).resolve().parents[1]


class PaperRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.relative = "configs/experiment/demo_reproduction.yaml"
        self.config = {
            "schema_version": 1, "kind": "experiment", "seed": 1,
            "data": {"name": "cifar10"}, "loss": {"name": "ce"},
            "execution": {"runner": "supervised"},
        }
        self.manifest = {"recipes": [self.relative], "public": [], "conditional": []}
        self.papers = [{"id": "demo", "configs": [{"recipe_id": "demo-reproduction", "profile": "reproduction"}]}]
        self.packaged = True

    def write_fixture(self):
        files = {
            "src/lnl_toolbox/cli/data/recipe_catalog.json": json.dumps(self.manifest),
            "src/lnl_toolbox/paper_catalog.json": json.dumps(self.papers),
            self.relative: yaml.safe_dump(self.config),
            "pyproject.toml": '[tool.setuptools.data-files]\n"share/lnl-toolbox/configs/experiment" = ' + json.dumps([self.relative] if self.packaged else []) + '\n',
        }
        for relative, text in files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def codes(self):
        self.write_fixture()
        return {issue.code for issue in check_repository(self.root)}

    def test_current_all_papers_have_consistent_registration_without_preflight(self):
        with patch("lnl_toolbox.training.service.ExperimentService.preflight", side_effect=AssertionError("must not preflight")), \
             patch("lnl_toolbox.training.service.ExperimentService.prerequisite_readiness", side_effect=AssertionError("must not consume artifacts")):
            self.assertEqual(check_repository(ROOT), [])
        papers = json.loads((ROOT / "src/lnl_toolbox/paper_catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(len(papers), 26)
        self.assertEqual(sum(len(paper["configs"]) for paper in papers), 56)

    def test_minimal_valid_chain(self):
        self.assertEqual(self.codes(), set())

    def test_unknown_public_and_conditional_references(self):
        self.manifest.update(public=[{"path": "missing.yaml"}], conditional=["absent.yaml"])
        self.assertIn("unknown_catalog_reference", self.codes())

    def test_missing_yaml(self):
        self.manifest["recipes"] = ["configs/experiment/missing.yaml"]
        self.assertIn("recipe_path_invalid", self.codes())

    def test_unknown_paper_recipe(self):
        self.papers[0]["configs"][0]["recipe_id"] = "missing"
        self.assertIn("paper_recipe_missing", self.codes())

    def test_unknown_runner(self):
        self.config["execution"]["runner"] = "does_not_exist"
        self.assertIn("recipe_resolution_failed", self.codes())

    def test_missing_requirements(self):
        self.write_fixture()
        with patch("lnl_toolbox.training.runners.RunnerSpec.requirements", return_value=None):
            self.assertIn("requirements_unavailable", {i.code for i in check_repository(self.root)})

    def test_formal_recipe_must_be_packaged_in_correct_destination(self):
        self.packaged = False
        self.assertIn("formal_recipe_not_packaged", self.codes())

    def test_duplicate_registration(self):
        self.manifest["recipes"].append(self.relative)
        self.papers.append(deepcopy(self.papers[0]))
        self.assertTrue({"duplicate_recipe_path", "duplicate_paper_id"} <= self.codes())

    def test_metadata_only_paper_is_allowed(self):
        self.papers[0]["configs"] = []
        self.assertEqual(self.codes(), set())

    def test_formal_recipe_packaging_checked_without_paper_reference(self):
        self.papers[0]["configs"] = []
        self.packaged = False
        self.assertIn("formal_recipe_not_packaged", self.codes())

    def test_duplicate_filename_derived_recipe_ids_are_rejected(self):
        self.write_fixture()
        other = "configs/reproduction/demo_reproduction.yaml"
        path = self.root / other
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        self.manifest["recipes"].append(other)
        self.assertIn("duplicate_recipe_id", self.codes())

    def test_recipe_path_cannot_escape_repository(self):
        self.manifest["recipes"] = ["../outside.yaml"]
        self.assertIn("recipe_path_invalid", self.codes())

    def test_malformed_catalog_is_reported(self):
        self.write_fixture()
        path = self.root / "src/lnl_toolbox/paper_catalog.json"
        path.write_text("{broken", encoding="utf-8")
        self.assertIn("catalog_unreadable", {i.code for i in check_repository(self.root)})


if __name__ == "__main__":
    unittest.main()
