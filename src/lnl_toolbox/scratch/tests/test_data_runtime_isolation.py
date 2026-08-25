import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ScratchDataRuntimeIsolationTest(unittest.TestCase):
    def test_data_runtime_has_no_legacy_runtime_imports(self):
        targets = [ROOT / "scratch" / "data_runtime.py", ROOT / "scratch" / "blocks" / "data.py"]
        forbidden = ("lnl_toolbox.data", "lnl_toolbox.training", "lnl_toolbox.noise")
        for path in targets:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(any(name.startswith(forbidden) for name in imports),
                             f"{path} imports legacy data runtime: {imports}")

    def test_old_data_prepare_blocks_are_not_reintroduced(self):
        text = (ROOT / "scratch" / "blocks" / "data.py").read_text(encoding="utf-8")
        self.assertNotIn("def prepare_", text)
        self.assertNotIn('id="prepare_', text)


if __name__ == "__main__":
    unittest.main()
