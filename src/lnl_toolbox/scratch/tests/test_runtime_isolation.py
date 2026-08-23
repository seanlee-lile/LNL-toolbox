from __future__ import annotations

import ast
from pathlib import Path
import unittest


class ScratchRuntimeIsolationTest(unittest.TestCase):
    def test_runtime_imports_only_scratch_from_lnl_toolbox(self) -> None:
        root = Path(__file__).resolve().parents[1]
        violations = []
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.startswith("lnl_toolbox.") and not name.startswith("lnl_toolbox.scratch"):
                        violations.append(f"{path.name}:{node.lineno}:{name}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
