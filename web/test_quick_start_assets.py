from __future__ import annotations

from pathlib import Path
import unittest

from web.command_console import STATIC_ASSETS


ROOT = Path(__file__).resolve().parents[1]


class QuickStartAssetsTests(unittest.TestCase):
    def test_assets_are_served_from_explicit_allowlist(self) -> None:
        self.assertEqual(set(STATIC_ASSETS), {"/assets/quick_start.js", "/assets/quick_start.css", "/assets/run_output.js"})
        self.assertTrue(all(path.is_file() for path, _content_type in STATIC_ASSETS.values()))

    def test_parent_traversal_is_not_an_asset(self) -> None:
        self.assertNotIn("/assets/../index.html", STATIC_ASSETS)

    def test_index_loads_external_quick_start_resources(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn('/assets/quick_start.css', html)
        self.assertIn('/assets/quick_start.js', html)
        self.assertIn('/assets/run_output.js', html)
        self.assertIn('window.quickStartController', html)
        self.assertIn('id="training-output"', html)

    def test_existing_modules_remain(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        for module in ('quickstart', 'beginner', 'yaml', 'data', 'sweep', 'results', 'papers', 'advanced'):
            self.assertIn('id: "' + module + '"', html)

    def test_registered_dataset_and_training_guard_are_wired(self) -> None:
        script = (ROOT / "web" / "assets" / "quick_start.js").read_text(encoding="utf-8")
        self.assertIn("esc(item.location)", script)
        self.assertIn('payload.status === "already_registered"', script)
        self.assertIn("qs-required-input", script)
        self.assertIn("window.confirm", script)
        self.assertIn("loadingMarkup", script)
        self.assertIn("请勿重复点击", script)
        self.assertIn("finally(endLoading)", script)
        self.assertIn("item.key === state.noiseSelection.key", script)
        self.assertIn('qs-rate")?.addEventListener("change", updateNoise)', script)
        self.assertIn('qs-seed")?.addEventListener("change", updateNoise)', script)
        self.assertIn("训练轮次", (ROOT / "src" / "lnl_toolbox" / "quickstart" / "service.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
