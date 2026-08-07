import io
import tempfile
import unittest
from pathlib import Path

from lnl_toolbox.training.progress import (
    TerminalTrainingProgress,
    write_training_curves_svg,
)


class TrainingProgressTest(unittest.TestCase):
    def test_terminal_progress_reports_interval_and_final_batch(self) -> None:
        stream = io.StringIO()
        progress = TerminalTrainingProgress(
            epoch=2,
            total_epochs=5,
            total_batches=5,
            update_interval=2,
            stream=stream,
            force=True,
        )
        for batch in range(1, 6):
            progress.update(batch, loss=1.0 / batch, accuracy=batch / 10.0)
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("Epoch 002/005", lines[0])
        self.assertIn("0002/0005", lines[0])
        self.assertIn("0005/0005", lines[-1])

    def test_disabled_terminal_progress_is_silent(self) -> None:
        stream = io.StringIO()
        progress = TerminalTrainingProgress(
            epoch=1,
            total_epochs=1,
            total_batches=1,
            enabled=False,
            stream=stream,
        )
        progress.update(1, loss=1.0, accuracy=0.5)
        self.assertEqual(stream.getvalue(), "")

    def test_non_interactive_terminal_progress_is_silent(self) -> None:
        stream = io.StringIO()
        progress = TerminalTrainingProgress(
            epoch=1,
            total_epochs=1,
            total_batches=1,
            stream=stream,
        )
        progress.update(1, loss=1.0, accuracy=0.5)
        self.assertEqual(stream.getvalue(), "")

    def test_svg_contains_all_curves_and_is_overwritten(self) -> None:
        rows = [
            {
                "train_loss": 2.0,
                "validation_loss": 2.1,
                "train_accuracy": 0.2,
                "validation_accuracy": 0.18,
                "learning_rate": 0.01,
            },
            {
                "train_loss": 1.5,
                "validation_loss": 1.7,
                "train_accuracy": 0.4,
                "validation_accuracy": 0.35,
                "learning_rate": 0.001,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_curves.svg"
            returned = write_training_curves_svg(rows[:1], path)
            write_training_curves_svg(rows, path)
            text = path.read_text(encoding="utf-8")
        self.assertEqual(returned, path)
        self.assertIn("<svg", text)
        self.assertIn("Training progress", text)
        self.assertIn("epochs: 2", text)
        self.assertGreaterEqual(text.count("<polyline"), 5)

    def test_progress_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            TerminalTrainingProgress(
                epoch=0, total_epochs=1, total_batches=1
            )
        with self.assertRaises(ValueError):
            write_training_curves_svg([], Path("unused.svg"))


if __name__ == "__main__":
    unittest.main()
