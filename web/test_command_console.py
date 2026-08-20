import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))
import command_console  # noqa: E402


class CommandConsoleTest(unittest.TestCase):
    def test_command_catalog_is_fixed_and_displayable(self):
        self.assertGreaterEqual(len(command_console.COMMANDS), 8)
        for spec in command_console.COMMANDS.values():
            self.assertTrue(spec.display_command.startswith("lnl "))
            self.assertNotIn("|", spec.args)
            self.assertNotIn(";", spec.args)

    def test_recipe_menu_covers_built_in_recipes(self):
        recipes = command_console._recipe_payload()
        self.assertGreaterEqual(len(recipes), 26)
        recipe_ids = {item["id"] for item in recipes}
        self.assertIn("cifar10-clean-smoke", recipe_ids)
        self.assertIn("fine-cifar100n-reproduction", recipe_ids)

    def test_sweep_shortcut_is_available(self):
        command = command_console.COMMANDS["sweep-smoke"].display_command
        self.assertIn("lnl sweep", command)
        self.assertIn("--seeds 1 2 3", command)

    def test_paper_menu_exposes_recipe_variants(self):
        papers = command_console._paper_payload()
        self.assertGreaterEqual(len(papers), 1)
        self.assertTrue(all(item["configs"] for item in papers))
        self.assertTrue(all("id" in item and "title" in item for item in papers))

    def test_recipe_yaml_can_be_loaded_and_saved_as_project_local_file(self):
        config = command_console._config_payload("cifar10-clean-smoke")
        self.assertIn("data:", config["content"])
        schema = command_console._config_schema("cifar10-clean-smoke")
        self.assertIn("data.name", {field["path"] for field in schema["fields"]})
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            destination = Path(directory) / "edited.yaml"
            saved = command_console._save_config(
                {
                    "path": str(destination),
                    "recipe": "cifar10-clean-smoke",
                    "patches": [{"path": "data.name", "value": "cifar10"}],
                }
            )
            self.assertEqual(Path(saved["path"]).name, "edited.yaml")
            self.assertTrue(destination.is_file())

    def test_schema_hides_component_wiring(self):
        schema = command_console._config_schema("fine-cifar100n-smoke")
        paths = {field["path"] for field in schema["fields"]}
        self.assertIn("fine.warmup_epochs", paths)
        self.assertNotIn("model.name", paths)
        self.assertNotIn("execution.runner", paths)

        cdr_schema = command_console._config_schema("cifar10-symmetric-cdr-smoke")
        cdr_paths = {field["path"] for field in cdr_schema["fields"]}
        self.assertIn("parameter_update.noise_rate", cdr_paths)
        self.assertNotIn("parameter_update.name", cdr_paths)

    def test_safe_save_rejects_component_changes(self):
        with tempfile.TemporaryDirectory(dir=command_console.ROOT) as directory:
            with self.assertRaises(ValueError):
                command_console._save_config(
                    {
                        "path": str(Path(directory) / "edited.yaml"),
                        "recipe": "fine-cifar100n-smoke",
                        "patches": [{"path": "model.name", "value": "resnet50"}],
                    }
                )

    def test_builtin_recipe_cannot_be_overwritten(self):
        config = command_console._config_payload("cifar10-clean-smoke")
        with self.assertRaises(ValueError):
            command_console._save_config(
                {
                    "path": config["path"],
                    "recipe": "cifar10-clean-smoke",
                    "patches": [],
                    "overwrite": True,
                }
            )

    def test_unknown_command_is_rejected(self):
        with self.assertRaises(KeyError):
            command_console.build_command("not-allowed")

    def test_free_command_is_parsed_without_shell(self):
        with mock.patch.object(command_console.shutil, "which", return_value="lnl"):
            command = command_console.parse_free_command(
                'lnl run --recipe "cifar10-clean-smoke" --epochs 1'
            )
        self.assertEqual(command[0], "lnl")
        self.assertIn("--epochs", command)
        self.assertIn("1", command)

    def test_free_command_rejects_shell_syntax(self):
        for raw in ("python evil.py", "lnl doctor | more", "lnl doctor > out.txt"):
            with self.assertRaises(ValueError):
                command_console.parse_free_command(raw)

    def test_fallback_command_uses_current_python(self):
        with mock.patch.object(command_console.shutil, "which", return_value=None):
            self.assertEqual(
                command_console.resolve_lnl_command(),
                [sys.executable, "-m", "lnl_toolbox.cli.main"],
            )

    def test_command_builder_does_not_use_shell(self):
        with mock.patch.object(command_console.shutil, "which", return_value="lnl"):
            command = command_console.build_command("train-one")
        self.assertEqual(command[:1], ["lnl"])
        self.assertIn("--epochs", command)
        self.assertIn("1", command)

    def test_job_payload_is_json_serializable(self):
        job = command_console.Job(
            job_id="abc",
            key="help",
            command=["lnl", "--help"],
            display_command="lnl --help",
            lines=["ok"],
            returncode=0,
        )
        payload = command_console._job_payload(job)
        self.assertEqual(payload["returncode"], 0)
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
