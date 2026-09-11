import tempfile
import unittest
from pathlib import Path

from lnl_toolbox.scratch.formula.storage import delete_formula, export_formula, import_formula, list_formula_files, load_formula, rename_formula, save_formula


def _formula(formula_id="user/storage_formula"):
    return {
        "id": formula_id,
        "name": "Storage Formula",
        "inputs": {"x": {}},
        "steps": [{"id": "out", "block": "detach", "bindings": {"input": "x"}}],
        "outputs": {"value": {"source": "out"}},
    }


class FormulaStorageTest(unittest.TestCase):
    def test_yaml_roundtrip_rename_export_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = save_formula(_formula(), root)
            self.assertTrue(path.is_file())
            self.assertEqual(load_formula("user/storage_formula", root).id, "user/storage_formula")
            exported = export_formula("user/storage_formula", root=root)
            self.assertIn("steps:", exported)
            rename_formula("user/storage_formula", "user/renamed_formula", root)
            self.assertEqual(load_formula("user/renamed_formula", root).id, "user/renamed_formula")
            imported = import_formula(exported, root)
            self.assertEqual(imported.id, "user/storage_formula")
            delete_formula("user/storage_formula", root)
            self.assertFalse(any(path.name == "storage_formula.yaml" for path in list_formula_files(root)))

