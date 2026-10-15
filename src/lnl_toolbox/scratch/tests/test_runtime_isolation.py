from __future__ import annotations

import ast
from pathlib import Path
import unittest


class ScratchRuntimeIsolationTest(unittest.TestCase):
    def test_runtime_reuses_only_explicit_artifact_and_model_interfaces(self) -> None:
        root = Path(__file__).resolve().parents[1]
        # Artifact/model interfaces and detached mathematical operations are
        # reusable, not algorithm/trainer delegation. Restrict each exception by file,
        # module, symbol, and import form; all other legacy imports still fail.
        shared_interfaces = {
            ("native_stats.py", "lnl_toolbox.training.mentor_artifacts", ("MentorArtifact",)),
            ("native_stats.py", "lnl_toolbox.models.mentornet", ("build_mentor_model",)),
            ("blocks/graph.py", "lnl_toolbox.algorithms.lend.dilution", ("dilute_labels",)),
            ("blocks/parameter.py", "lnl_toolbox.algorithms.cdr", ("critical_parameter_masks", "official_code_parameter_masks")),
        }
        seen = set()
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
                        if isinstance(node, ast.ImportFrom):
                            interface = (path.relative_to(root).as_posix(), name,
                                         tuple(alias.name for alias in node.names))
                            if interface in shared_interfaces:
                                seen.add(interface)
                                continue
                        violations.append(f"{path.name}:{node.lineno}:{name}")
        self.assertEqual(violations, [])
        self.assertEqual(seen, shared_interfaces)


if __name__ == "__main__":
    unittest.main()
