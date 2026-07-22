from __future__ import annotations

import unittest

import torch

from lnl_toolbox.selectors import (
    AllSelector,
    ConstantKeepRateSchedule,
    LinearKeepRateSchedule,
    SelectionInput,
    SelectionResult,
    SmallLossSelector,
    validate_selection_result,
)


class SelectorTest(unittest.TestCase):
    def test_constant_schedule_returns_the_same_rate_at_every_epoch(self) -> None:
        schedule = ConstantKeepRateSchedule(0.8)
        self.assertEqual(schedule.rate_at(0), 0.8)
        self.assertEqual(schedule.rate_at(5), 0.8)

    def test_linear_schedule_uses_zero_based_start_middle_end_and_plateau(self) -> None:
        schedule = LinearKeepRateSchedule(start=1.0, end=0.6, warmup_epochs=10)
        self.assertEqual(schedule.rate_at(0), 1.0)
        self.assertEqual(schedule.rate_at(5), 0.8)
        self.assertEqual(schedule.rate_at(10), 0.6)
        self.assertEqual(schedule.rate_at(12), 0.6)

    def test_schedule_rejects_invalid_parameters_and_epochs(self) -> None:
        for kwargs in (
            {"start": 0.0, "end": 0.6, "warmup_epochs": 10},
            {"start": 1.0, "end": 1.1, "warmup_epochs": 10},
            {"start": 1.0, "end": float("nan"), "warmup_epochs": 10},
            {"start": 1.0, "end": 0.6, "warmup_epochs": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LinearKeepRateSchedule(**kwargs)
        with self.assertRaises(TypeError):
            LinearKeepRateSchedule(start=1.0, end=0.6, warmup_epochs=1.5)
        schedule = ConstantKeepRateSchedule(0.8)
        with self.assertRaises(TypeError):
            schedule.rate_at("1")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            schedule.rate_at(True)
        with self.assertRaises(ValueError):
            schedule.rate_at(-1)

    def test_all_selector_keeps_every_sample(self) -> None:
        result = AllSelector().select(
            SelectionInput(
                scores=torch.tensor([0.8, 0.2, 0.4]),
                sample_indices=torch.tensor([12, 3, 9]),
            )
        )

        self.assertEqual(result.selected_mask.tolist(), [True, True, True])
        self.assertEqual(result.metrics["selected_samples"], 3.0)
        self.assertEqual(result.metrics["selected_ratio"], 1.0)

    def test_small_loss_keeps_fixed_rate(self) -> None:
        result = SmallLossSelector(keep_rate=0.5).select(
            SelectionInput(
                scores=torch.tensor([0.8, 0.2, 0.4, 0.1]),
                sample_indices=torch.tensor([12, 3, 9, 20]),
            )
        )

        self.assertEqual(result.selected_mask.tolist(), [False, True, False, True])
        self.assertEqual(result.metrics["selected_samples"], 2.0)
        self.assertEqual(result.metrics["selected_ratio"], 0.5)
        self.assertEqual(result.metrics["keep_rate"], 0.5)

    def test_small_loss_accepts_constant_mapping_and_missing_epoch_defaults_to_zero(self) -> None:
        selector = SmallLossSelector({"name": "constant", "value": 0.5})
        missing_epoch = selector.select(
            SelectionInput(
                scores=torch.arange(4, dtype=torch.float32),
                sample_indices=torch.arange(4),
            )
        )
        explicit_epoch_zero = selector.select(
            SelectionInput(
                scores=torch.arange(4, dtype=torch.float32),
                sample_indices=torch.arange(4),
                metadata={"epoch": 0},
            )
        )
        self.assertTrue(torch.equal(
            missing_epoch.selected_mask, explicit_epoch_zero.selected_mask
        ))
        self.assertEqual(missing_epoch.metrics["keep_rate"], 0.5)

    def test_small_loss_linear_schedule_selects_expected_count_by_epoch(self) -> None:
        selector = SmallLossSelector({
            "name": "linear", "start": 1.0, "end": 0.6, "warmup_epochs": 2,
        })
        expected = {0: (10, 1.0), 1: (8, 0.8), 2: (6, 0.6), 3: (6, 0.6)}
        for epoch, (count, rate) in expected.items():
            with self.subTest(epoch=epoch):
                result = selector.select(SelectionInput(
                    scores=torch.arange(10, dtype=torch.float32),
                    sample_indices=torch.arange(10),
                    metadata={"epoch": epoch},
                ))
                self.assertEqual(int(result.selected_mask.sum().item()), count)
                self.assertAlmostEqual(result.metrics["keep_rate"], rate)

    def test_small_loss_rejects_uncertain_epoch_metadata(self) -> None:
        selector = SmallLossSelector(0.5)
        for epoch, error in ((None, TypeError), (1.5, TypeError), (True, TypeError), (-1, ValueError)):
            with self.subTest(epoch=epoch), self.assertRaises(error):
                selector.select(SelectionInput(
                    scores=torch.tensor([0.1, 0.2]),
                    sample_indices=torch.tensor([1, 2]),
                    metadata={"epoch": epoch},
                ))

    def test_small_loss_rounds_up_and_keeps_at_least_one(self) -> None:
        result = SmallLossSelector(keep_rate=0.01).select(
            SelectionInput(
                scores=torch.tensor([0.3, 0.1, 0.2]),
                sample_indices=torch.tensor([1, 2, 3]),
            )
        )

        self.assertEqual(result.selected_mask.tolist(), [False, True, False])

    def test_equal_scores_use_stable_global_index_tie_break(self) -> None:
        selector = SmallLossSelector(keep_rate=0.5)
        first = selector.select(
            SelectionInput(
                scores=torch.ones(4),
                sample_indices=torch.tensor([40, 10, 30, 20]),
            )
        )
        second = selector.select(
            SelectionInput(
                scores=torch.ones(4),
                sample_indices=torch.tensor([20, 40, 10, 30]),
            )
        )

        first_ids = {
            value
            for value, keep in zip([40, 10, 30, 20], first.selected_mask.tolist())
            if keep
        }
        second_ids = {
            value
            for value, keep in zip([20, 40, 10, 30], second.selected_mask.tolist())
            if keep
        }
        self.assertEqual(first_ids, {10, 20})
        self.assertEqual(second_ids, {10, 20})

    def test_selector_rejects_scores_attached_to_autograd(self) -> None:
        with self.assertRaisesRegex(ValueError, "detached"):
            AllSelector().select(
                SelectionInput(
                    scores=torch.tensor([0.1, 0.2], requires_grad=True),
                    sample_indices=torch.tensor([1, 2]),
                )
            )

    def test_selector_rejects_duplicate_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            SmallLossSelector(0.5).select(
                SelectionInput(
                    scores=torch.tensor([0.1, 0.2]),
                    sample_indices=torch.tensor([1, 1]),
                )
            )

    def test_keep_rate_validation(self) -> None:
        for value in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SmallLossSelector(value)

    def test_result_requires_boolean_aligned_nonempty_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "torch.bool"):
            validate_selection_result(
                SelectionResult(selected_mask=torch.tensor([1, 0])),
                batch_size=2,
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_selection_result(
                SelectionResult(selected_mask=torch.zeros(2, dtype=torch.bool)),
                batch_size=2,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
