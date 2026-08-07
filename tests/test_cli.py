from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from lnl_toolbox.cli import PromptSession, TrainingSelection
import lnl_toolbox.cli as cli_shared
from lnl_toolbox.cli import clean_train as clean_cli
from lnl_toolbox.cli import inspect_data as inspect_cli
from lnl_toolbox.cli import make_noise as noise_cli
from lnl_toolbox.cli import train as train_cli


def scripted_session(values: list[str]) -> tuple[PromptSession, list[str]]:
    answers = iter(values)
    output: list[str] = []
    return PromptSession(read=lambda _: next(answers), write=output.append), output


class PromptSessionTest(unittest.TestCase):
    def test_clean_template_discovery_excludes_noisy_configs(self) -> None:
        names = {path.name for path, _ in cli_shared._experiment_templates(clean=True)}
        self.assertNotIn("cifar10_symmetric_ce_smoke.yaml", names)

    def test_choice_and_number_retry(self) -> None:
        session, output = scripted_session(["invalid", "2", "bad", "0", "3"])
        self.assertEqual(
            session.choose("choice", [("First", "first"), ("Second", "second")]),
            "second",
        )
        self.assertEqual(session.integer("count", minimum=1), 3)
        self.assertTrue(any("选项无效" in item for item in output))
        self.assertTrue(any("请输入整数" in item for item in output))

    def test_loss_wizard_builds_gce_and_nested_apl(self) -> None:
        gce_session, _ = scripted_session(["gce", "0.6"])
        self.assertEqual(cli_shared._prompt_loss(gce_session, {"name": "ce"}),
                         {"name": "gce", "q": 0.6})

        apl_session, _ = scripted_session(["apl", "mae", "", "2", "0.5", ""])
        self.assertEqual(
            cli_shared._prompt_loss(apl_session, {"name": "ce"}),
            {
                "name": "apl", "alpha": 2.0, "beta": 0.5,
                "active": {"name": "nce", "eps": 1e-8},
                "passive": {"name": "mae", "scale": 2.0},
            },
        )

    def test_apl_wizard_retries_zero_weights(self) -> None:
        session, _ = scripted_session(["apl", "rce", "", "0", "2", "0", "0.5", ""])
        self.assertEqual(
            cli_shared._prompt_loss(session, {"name": "ce"}),
            {
                "name": "apl", "alpha": 2.0, "beta": 0.5,
                "active": {"name": "nce", "eps": 1e-8},
                "passive": {"name": "rce", "log_zero": -4.0},
            },
        )

    def test_clean_scheduler_wizard_builds_multistep(self) -> None:
        session, _ = scripted_session(["multistep", "5,10", "0.2"])
        self.assertEqual(
            cli_shared._prompt_scheduler(session, {"name": "none"}, 20),
            {"name": "multistep", "milestones": [5, 10], "gamma": 0.2},
        )

    def test_noise_wizard_builds_generated_and_external_modes(self) -> None:
        generated, _ = scripted_session(["2", "", "2", "0.3", "9"])
        self.assertEqual(
            cli_shared._prompt_noise(generated, None, 1),
            {
                "name": "pairflip",
                "rate": 0.3,
                "seed": 9,
                "manifest_filename": "noise_manifest.npz",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "source.npz"
            manifest.touch()
            external, _ = scripted_session(["3", "", str(manifest)])
            self.assertEqual(
                cli_shared._prompt_noise(external, None, 1),
                {
                    "manifest": str(manifest),
                    "manifest_filename": "noise_manifest.npz",
                },
            )

    def test_training_wizard_keeps_template_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = {
                "seed": 7,
                "data": {"name": "cifar10", "root": str(root)},
                "loader": {"batch_size": 16},
                "model": {"name": "tiny_cnn", "width": 8},
                "loss": {"name": "ce"},
                "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
                "trainer": {"epochs": 2, "device": "cpu"},
            }
            session, _ = scripted_session([
                "", "", "", "", "gce", "0.5", "", "", "", "", "", "", "", "", "", "y",
            ])
            with patch.object(cli_shared, "_choose_template", return_value=(root / "base.yaml", template)):
                selection = cli_shared.prompt_training_selection(session, clean=False)
            self.assertIsNotNone(selection)
            assert selection is not None
            self.assertEqual(selection.config["loss"], {"name": "gce", "q": 0.5})
            self.assertNotIn("noise", selection.config)
            self.assertEqual(template["loss"], {"name": "ce"})


class CliEntryPointTest(unittest.TestCase):
    def test_train_help_lists_t_revision(self) -> None:
        help_text = train_cli.build_parser().format_help()
        self.assertIn("t_revision", help_text)

    def test_missing_training_config_has_clear_error(self) -> None:
        missing = Path("definitely-missing-t-revision-config.yaml")
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            train_cli.main(["--config", str(missing)])

    def test_epochs_override_targets_t_revision_final_stage(self) -> None:
        config = {
            "method": "t_revision",
            "trainer": {"device": "cpu"},
            "t_revision": {"revision": {"epochs": 2}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch.object(train_cli, "run_experiment") as run:
                self.assertEqual(
                    train_cli.main(["--config", str(path), "--epochs", "5"]),
                    0,
                )
        called = run.call_args.args[0]
        self.assertEqual(called["t_revision"]["revision"]["epochs"], 5)
        self.assertNotIn("epochs", called["trainer"])

    def test_epochs_override_rejects_ambiguous_staged_method(self) -> None:
        config = {
            "method": "dual_t",
            "posterior_stage": {"epochs": 1},
            "final_stage": {"epochs": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ambiguous.*dual_t"):
                train_cli.main(["--config", str(path), "--epochs", "5"])

    def test_all_no_argument_entrypoints_dispatch_interactively(self) -> None:
        session, _ = scripted_session([])
        selection = TrainingSelection({"trainer": {"epochs": 1}}, Path("template.yaml"))
        with patch.object(train_cli, "prompt_training_selection", return_value=selection), \
             patch.object(train_cli, "run_experiment") as run:
            self.assertEqual(train_cli.main([], session), 0)
            run.assert_called_once_with(selection.config, None, None)

        seed_selection = TrainingSelection(
            {"output_root": "artifacts/runs"}, Path("template.yaml"),
            Path("suite"), seeds=[1, 2],
        )
        with patch.object(clean_cli, "prompt_training_selection", return_value=seed_selection), \
             patch.object(clean_cli, "run_seed_suite") as run_suite:
            self.assertEqual(clean_cli.main([], session), 0)
            run_suite.assert_called_once_with(seed_selection.config, [1, 2], Path("suite"))

        with patch.object(inspect_cli, "_interactive", return_value=("cifar10", Path("data"), "all")), \
             patch.object(inspect_cli, "_execute") as inspect:
            self.assertEqual(inspect_cli.main([], session), 0)
            inspect.assert_called_once_with("cifar10", Path("data"), "all")

        noise_values = (Path("labels.npy"), Path("noise.npz"), "symmetric", 0.2, 10, 1, "cifar10")
        with patch.object(noise_cli, "_interactive", return_value=noise_values), \
             patch.object(noise_cli, "_execute") as generate:
            self.assertEqual(noise_cli.main([], session), 0)
            generate.assert_called_once_with(*noise_values)

    def test_argument_mode_bypasses_terminal_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump({"trainer": {"epochs": 2}}), encoding="utf-8")
            session = PromptSession(read=lambda _: self.fail("stdin should not be read"))
            with patch.object(train_cli, "run_experiment") as run:
                result = train_cli.main(
                    ["--config", str(config_path), "--epochs", "3", "--output-dir", "out"],
                    session,
                )
            self.assertEqual(result, 0)
            called_config, output_dir, resume = run.call_args.args
            self.assertEqual(called_config["trainer"]["epochs"], 3)
            self.assertEqual(output_dir, Path("out"))
            self.assertIsNone(resume)

    def test_cancelled_prompt_returns_130(self) -> None:
        def cancel(_: str) -> str:
            raise EOFError

        output: list[str] = []
        result = inspect_cli.main([], PromptSession(read=cancel, write=output.append))
        self.assertEqual(result, 130)
        self.assertTrue(any("已取消" in item for item in output))

    def test_clean_argument_mode_rejects_resume_with_seed_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump({"trainer": {"epochs": 2}}), encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                clean_cli.main([
                    "--config", str(config_path), "--resume", "last.pt", "--seeds", "1", "2",
                ])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
