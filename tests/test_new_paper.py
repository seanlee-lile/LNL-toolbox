from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from lnl_toolbox.catalog import load_yaml
from scripts.check_paper_registration import check_draft, main as check_main
from scripts.new_paper import EXTENSION_TYPES, create_draft, main


ROOT = Path(__file__).resolve().parents[1]


class NewPaperDraftTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "drafts"

    def draft(self, **kwargs):
        return create_draft("new-paper-demo", self.output, **kwargs)

    def test_default_creates_linked_materials_without_fake_implementation(self):
        path = self.draft()
        draft = json.loads((path / "draft.json").read_text(encoding="utf-8"))
        paper = json.loads((path / draft["paper_file"]).read_text(encoding="utf-8"))
        config = yaml.safe_load((path / draft["recipe_file"]).read_text(encoding="utf-8"))
        self.assertFalse(draft["registered"])
        self.assertTrue(draft["todo"])
        self.assertEqual(config, {"schema_version": 1, "kind": "fragment"})
        self.assertEqual(paper["configs"], [])
        self.assertEqual(paper["availability"], "unavailable")
        self.assertEqual(paper["reproduction_status"], "not_run")
        self.assertEqual({i.code for i in check_draft(path)}, {"draft_incomplete"})

    def test_each_extension_type_uses_same_safe_draft_boundary(self):
        for index, extension in enumerate(EXTENSION_TYPES):
            with self.subTest(extension=extension):
                path = create_draft(f"paper-{index}", self.output, extension_type=extension)
                self.assertEqual({i.code for i in check_draft(path)}, {"draft_incomplete"})
                config = yaml.safe_load(next((path / "configs/experiment").glob("*.yaml")).read_text())
                self.assertNotIn("method", config)
                self.assertNotIn("prerequisites", config)

    def test_explicit_base_preserves_all_training_values_and_requires_review(self):
        base = ROOT / "configs/experiment/cifar10_clean_smoke.yaml"
        path = self.draft(base_config=base, title="New Robust Loss Paper")
        manifest = json.loads((path / "draft.json").read_text())
        self.assertEqual(yaml.safe_load((path / manifest["recipe_file"]).read_text()), load_yaml(base))
        paper = json.loads((path / "paper.json").read_text())
        recipe_id = Path(manifest["recipe_file"]).stem.replace("_", "-")
        self.assertEqual(paper["configs"][0]["recipe_id"], recipe_id)
        self.assertEqual(paper["configs"][0]["profile"], "experiment")
        self.assertEqual(paper["title"], "New Robust Loss Paper")
        self.assertEqual({i.code for i in check_draft(path)}, {"draft_incomplete"})

    def test_duplicate_refusal_preserves_all_existing_bytes(self):
        path = self.draft()
        before = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.draft()
        self.assertEqual(before, {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()})

    def test_invalid_ids_create_nothing(self):
        for name in ("", "../outside", "Bad-ID", "with_underscore", "a/b", "con", "a" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                create_draft(name, self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_type_creates_nothing(self):
        with self.assertRaises(ValueError):
            self.draft(extension_type="UNKNOWN")
        self.assertFalse(self.output.exists())

    def test_parent_traversal_and_file_output_are_rejected(self):
        with self.assertRaises(ValueError):
            create_draft("demo", self.output / ".." / "escape")
        file = Path(self.temporary.name) / "file"
        file.write_text("keep")
        with self.assertRaises(ValueError):
            create_draft("demo", file)
        self.assertEqual(file.read_text(), "keep")

    def test_symlink_output_is_refused_without_writing(self):
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda p: p == self.output or original(p)):
            with self.assertRaisesRegex(ValueError, "symlinks"):
                self.draft()
        self.assertFalse(self.output.exists())

    def test_bad_base_config_creates_nothing(self):
        base = Path(self.temporary.name) / "bad.yaml"
        base.write_text("schema_version: 1\nkind: experiment\ndata: {name: cifar10}\nexecution: {runner: nonexistent}\n")
        with self.assertRaises(ValueError):
            self.draft(base_config=base)
        self.assertFalse(self.output.exists())

    def test_live_catalogs_and_packaging_are_never_written(self):
        paths = [ROOT / "src/lnl_toolbox/paper_catalog.json", ROOT / "src/lnl_toolbox/cli/data/recipe_catalog.json", ROOT / "pyproject.toml"]
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        self.draft()
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])

    def test_checker_detects_tampered_draft_path(self):
        path = self.draft()
        metadata = json.loads((path / "draft.json").read_text())
        metadata["recipe_file"] = "../outside.yaml"
        (path / "draft.json").write_text(json.dumps(metadata))
        self.assertIn("draft_path_invalid", {i.code for i in check_draft(path)})

    def test_checker_detects_wrong_recipe_reference_and_reproduction_claim(self):
        path = self.draft(base_config=ROOT / "configs/experiment/cifar10_clean_smoke.yaml")
        paper_path = path / "paper.json"
        paper = json.loads(paper_path.read_text())
        paper["configs"][0]["recipe_id"] = "another-recipe"
        paper["reproduction_status"] = "verified"
        paper_path.write_text(json.dumps(paper))
        self.assertTrue({"paper_recipe_missing", "draft_reproduction_claim"} <= {i.code for i in check_draft(path)})

    def test_cli_reports_creation_not_registration_and_checker_reports_todos(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["new-paper-demo", "--output-dir", str(self.output)]), 0)
            self.assertEqual(check_main(["--draft", str(self.output / "new-paper-demo")]), 1)
        self.assertIn("NOT REGISTERED", output.getvalue())
        self.assertIn("draft_incomplete", output.getvalue())


if __name__ == "__main__":
    unittest.main()
