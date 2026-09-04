from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from lnl_toolbox.scratch import load_recipe, validate_recipe
from lnl_toolbox.scratch.web.server import _paper_examples, _recipe_path, _template_catalog
from web.command_console import _scratch_examples_payload, _scratch_recipe_path, _scratch_template_payload


PAPER_ROOT = Path(__file__).resolve().parents[1] / "recipes" / "papers"
FORMULA_READY = {"gce", "coteaching"}


class ScratchWebTemplateTest(unittest.TestCase):
    def _assert_catalog(self, items: list[dict[str, str]]) -> None:
        self.assertEqual(len(items), 26)
        ids = [item["id"] for item in items]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {path.stem for path in PAPER_ROOT.glob("*.yaml")})
        self.assertNotIn("legacy-scratch", {item["status"] for item in items})
        for item in items:
            self.assertEqual(item["path"], f"papers/{item['id']}.yaml")
            self.assertTrue((PAPER_ROOT / f"{item['id']}.yaml").is_file())
            expected = "formula-ready" if item["id"] in FORMULA_READY else "template-ready"
            self.assertEqual(item["status"], expected)

    def test_standalone_template_and_example_catalogs_are_full(self) -> None:
        catalog = _template_catalog()
        self._assert_catalog(catalog)
        self.assertEqual(_paper_examples(), catalog)

    def test_mounted_scratch_template_and_example_catalogs_are_full(self) -> None:
        catalog = _scratch_template_payload()
        self._assert_catalog(catalog)
        self.assertEqual(_scratch_examples_payload(), catalog)

    def test_every_paper_recipe_still_validates(self) -> None:
        for path in sorted(PAPER_ROOT.glob("*.yaml")):
            validate_recipe(load_recipe(path))

    def test_packaged_template_loads_when_user_recipe_root_is_locked(self) -> None:
        # Opening a paper template must not fail just because the optional
        # per-user recipe directory cannot be created on this machine.
        with mock.patch("lnl_toolbox.scratch.web.server.recipe_workspace_root", side_effect=PermissionError("locked")):
            self.assertEqual(_recipe_path("papers/gce.yaml"), PAPER_ROOT / "gce.yaml")
        with mock.patch("lnl_toolbox.scratch.recipe.recipe_workspace_root", side_effect=PermissionError("locked")):
            self.assertEqual(_scratch_recipe_path("papers/gce.yaml"), PAPER_ROOT / "gce.yaml")


if __name__ == "__main__":
    unittest.main()
