import copy
import math
import unittest
from unittest.mock import patch

import torch

from lnl_toolbox.algorithms.cnlcu import (
    HardRobustLossEstimator, PeerLossHistory, cnlcu_hard_score,
    cnlcu_soft_score, lof_retained_mask, soft_influence, soft_robust_mean,
)


class CNLCUSoftEstimatorTest(unittest.TestCase):
    def test_influence_and_eq3_match_hand_calculation(self):
        losses = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        observed = torch.tensor([[True, True], [True, False]])
        transformed = soft_influence(losses)
        torch.testing.assert_close(transformed[0], torch.log(torch.tensor([1.0, 2.5])))
        means, lengths = soft_robust_mean(losses, observed)
        self.assertEqual(lengths.tolist(), [2, 1])
        self.assertAlmostEqual(means[0].item(), math.log(2.5) / 2, places=6)
        self.assertAlmostEqual(means[1].item(), math.log(5.0), places=6)

    def test_eq7_matches_hand_calculation_without_relu(self):
        mean = torch.tensor([0.1])
        t = torch.tensor([2])
        count = torch.tensor([1])
        score, bonus = cnlcu_soft_score(mean, t, count, 0.2)
        expected = 0.2 * (2 + 0.2 * math.log(4) / 4) / 0.8
        self.assertAlmostEqual(bonus.item(), expected, places=6)
        self.assertAlmostEqual(score.item(), 0.1 - expected, places=6)
        self.assertLess(score.item(), 0.0)

    def test_less_selected_sample_gets_larger_bonus(self):
        mean = torch.tensor([1.0, 1.0])
        score, bonus = cnlcu_soft_score(mean, torch.tensor([3, 3]), torch.tensor([1, 5]), 0.1)
        self.assertGreater(bonus[0], bonus[1])
        self.assertLess(score[0], score[1])

    def test_invalid_values_fail(self):
        for sigma in (0.0, 1.0, math.nan):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                cnlcu_soft_score(torch.tensor([1.0]), torch.tensor([1]), torch.tensor([1]), sigma)
        with self.assertRaisesRegex(ValueError, "denominator"):
            cnlcu_soft_score(
                torch.tensor([1.0]), torch.tensor([1]), torch.tensor([0.05]), 0.1
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            soft_influence(torch.tensor([math.inf]))
        with self.assertRaisesRegex(ValueError, "at least one"):
            soft_robust_mean(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))


class CNLCUHistoryTest(unittest.TestCase):
    def test_sparse_mapping_append_permutation_and_window_reset(self):
        history = PeerLossHistory(torch.tensor([40, 10, 90]), 2, "a")
        history.prepare_epoch(0)
        rows = history.append(torch.tensor([90, 10]), torch.tensor([0.9, 0.1]))
        values, observed, counts = history.lookup_rows(rows)
        torch.testing.assert_close(values[:, 0], torch.tensor([0.9, 0.1]))
        self.assertTrue(bool(observed[:, 0].all()))
        history.increment_selected(rows, torch.tensor([False, True]))
        self.assertEqual(history.lookup_rows(rows)[2].tolist(), [0, 1])
        history.prepare_epoch(1)
        history.append(torch.tensor([10, 90]), torch.tensor([0.2, 0.8]))
        self.assertEqual(history.lookup_rows(history.resolve(torch.tensor([90])))[1].sum().item(), 2)
        history.prepare_epoch(2)
        self.assertFalse(bool(history.observed.any()))
        self.assertEqual(history.selected_count.sum().item(), 0)

    def test_selection_count_resets_with_each_epoch_window(self):
        history = PeerLossHistory(torch.tensor([10, 20]), 2, "a")
        history.prepare_epoch(0)
        rows = history.append(torch.tensor([10, 20]), torch.tensor([0.1, 0.2]))
        history.increment_selected(rows, torch.tensor([True, False]))
        history.prepare_epoch(1)
        self.assertEqual(history.selected_count.tolist(), [1, 0])
        history.prepare_epoch(2)
        self.assertEqual(history.window_start_epoch, 2)
        self.assertEqual(history.selected_count.tolist(), [0, 0])

    def test_duplicate_missing_and_double_observation_fail(self):
        history = PeerLossHistory(torch.tensor([10, 20]), 2, "a")
        history.prepare_epoch(0)
        with self.assertRaisesRegex(ValueError, "unique"):
            history.resolve(torch.tensor([10, 10]))
        with self.assertRaises(KeyError):
            history.resolve(torch.tensor([30]))
        history.append(torch.tensor([10]), torch.tensor([0.1]))
        with self.assertRaisesRegex(ValueError, "twice"):
            history.append(torch.tensor([10]), torch.tensor([0.2]))

    def test_state_roundtrip_and_mapping_drift_rejected(self):
        first = PeerLossHistory(torch.tensor([7, 20]), 3, "a")
        first.prepare_epoch(0)
        rows = first.append(torch.tensor([20]), torch.tensor([0.5]))
        first.increment_selected(rows, torch.tensor([True]))
        state = copy.deepcopy(first.state_dict())
        restored = PeerLossHistory(torch.tensor([20, 7]), 3, "a")
        restored.load_state_dict(state)
        self.assertEqual(restored.selected_count.tolist(), first.selected_count.tolist())
        with self.assertRaisesRegex(ValueError, "mapping"):
            PeerLossHistory(torch.tensor([7, 21]), 3, "a").load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "identity"):
            PeerLossHistory(torch.tensor([7, 20]), 3, "b").load_state_dict(state)
        negative = copy.deepcopy(state)
        negative["selected_count"][0] = -1
        with self.assertRaisesRegex(ValueError, "counts"):
            PeerLossHistory(torch.tensor([7, 20]), 3, "a").load_state_dict(negative)
        bad_cursor = copy.deepcopy(state)
        bad_cursor["active_epoch"] = 4
        with self.assertRaisesRegex(ValueError, "cursor"):
            PeerLossHistory(torch.tensor([7, 20]), 3, "a").load_state_dict(bad_cursor)


class CNLCUHardEstimatorTest(unittest.TestCase):
    def test_short_history_keeps_every_observation(self):
        history = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
        observed = torch.tensor([[True, True, False], [True, False, False]])
        retained = lof_retained_mask(
            history, observed, n_neighbors=2, contamination=0.1,
            minimum_observations=3,
        )
        torch.testing.assert_close(retained, observed)
        estimate = HardRobustLossEstimator(
            n_neighbors=2, contamination=0.1, minimum_observations=3,
        ).estimate(history, observed)
        torch.testing.assert_close(estimate.robust_mean, torch.tensor([1.5, 3.0], dtype=torch.float64))
        self.assertEqual(estimate.outlier_count.tolist(), [0, 0])

    def test_corrected_lof_removes_minus_one_and_uses_retained_denominator(self):
        class FakeLOF:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit_predict(self, samples):
                self.samples = samples
                return [-1, 1, 1]

        history = torch.tensor([[100.0, 1.0, 2.0, 999.0]])
        observed = torch.tensor([[True, True, True, False]])
        with patch(
            "lnl_toolbox.algorithms.cnlcu.outliers._load_local_outlier_factor",
            return_value=FakeLOF,
        ):
            estimate = HardRobustLossEstimator(
                n_neighbors=2, contamination=0.1, minimum_observations=3,
            ).estimate(history, observed)
        self.assertEqual(estimate.retained_mask.tolist(), [[False, True, True, False]])
        self.assertEqual(estimate.observation_count.tolist(), [3])
        self.assertEqual(estimate.outlier_count.tolist(), [1])
        self.assertEqual(estimate.retained_count.tolist(), [2])
        self.assertEqual(estimate.robust_mean.tolist(), [1.5])
        self.assertNotEqual(estimate.robust_mean.item(), (1.0 + 2.0) / 3.0)

    def test_actual_lof_is_deterministic_and_removes_extreme_value(self):
        history = torch.tensor([[1.0] * 9 + [100.0], [2.0] * 10])
        observed = torch.ones_like(history, dtype=torch.bool)
        kwargs = dict(n_neighbors=2, contamination=0.1, minimum_observations=5)
        first = lof_retained_mask(history, observed, **kwargs)
        second = lof_retained_mask(history, observed, **kwargs)
        torch.testing.assert_close(first, second)
        self.assertFalse(first[0, -1].item())
        self.assertTrue(bool(first[1].any()))

    def test_padding_permutations_and_invalid_detector_results(self):
        history = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 0.0]])
        observed = torch.tensor([[True, True, True], [True, True, False]])
        estimator = HardRobustLossEstimator(
            n_neighbors=2, contamination=0.1, minimum_observations=4,
        )
        first = estimator.estimate(history, observed)
        permutation = torch.tensor([1, 0])
        second = estimator.estimate(history[permutation], observed[permutation])
        torch.testing.assert_close(first.robust_mean[permutation], second.robust_mean)
        self.assertFalse(bool((first.retained_mask & ~observed).any()))

        class RejectAll:
            def __init__(self, **kwargs): pass
            def fit_predict(self, samples): return [-1] * len(samples)

        with patch(
            "lnl_toolbox.algorithms.cnlcu.outliers._load_local_outlier_factor",
            return_value=RejectAll,
        ), self.assertRaisesRegex(RuntimeError, "every observation"):
            lof_retained_mask(
                history[:1], observed[:1], n_neighbors=2,
                contamination=0.1, minimum_observations=3,
            )

    def test_missing_sklearn_and_invalid_parameters_fail(self):
        history = torch.ones(1, 3)
        observed = torch.ones(1, 3, dtype=torch.bool)
        with patch(
            "lnl_toolbox.algorithms.cnlcu.outliers._load_local_outlier_factor",
            side_effect=ImportError("optional training dependency [train]"),
        ), self.assertRaisesRegex(ImportError, "optional training dependency"):
            lof_retained_mask(
                history, observed, n_neighbors=2, contamination=0.1,
                minimum_observations=3,
            )
        for kwargs in (
            dict(n_neighbors=0, contamination=0.1, minimum_observations=3),
            dict(n_neighbors=2, contamination=0.5, minimum_observations=3),
            dict(n_neighbors=2, contamination=0.1, minimum_observations=2),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                lof_retained_mask(history, observed, **kwargs)

    def test_eq8_matches_hand_calculation_and_allows_negative_score(self):
        mean = torch.tensor([0.1], dtype=torch.float64)
        t = torch.tensor([4])
        outliers = torch.tensor([1])
        retained = torch.tensor([3])
        count = torch.tensor([2])
        score, bonus = cnlcu_hard_score(
            mean, t, outliers, retained, count, 0.01, 2.0,
        )
        factor = (
            2 * math.sqrt(0.02) * 2.0 * (4 + math.sqrt(2))
            / (3 * math.sqrt(4))
        )
        expected = factor * math.sqrt(math.log(16) / 2)
        self.assertAlmostEqual(bonus.item(), expected, places=12)
        self.assertAlmostEqual(score.item(), 0.1 - expected, places=12)
        self.assertLess(score.item(), 0.0)

    def test_eq8_zero_outliers_count_effect_and_invalid_inputs(self):
        mean = torch.tensor([1.0, 1.0], dtype=torch.float64)
        score, bonus = cnlcu_hard_score(
            mean, torch.tensor([2, 2]), torch.tensor([0, 0]),
            torch.tensor([2, 2]), torch.tensor([1, 4]), 0.01, 2.0,
        )
        self.assertGreater(bonus[0], bonus[1])
        self.assertLess(score[0], score[1])
        for tau, bound in ((0.0, 2.0), (0.01, 0.0), (math.nan, 2.0)):
            with self.subTest(tau=tau, bound=bound), self.assertRaises(ValueError):
                cnlcu_hard_score(
                    mean[:1], torch.tensor([1]), torch.tensor([0]),
                    torch.tensor([1]), torch.tensor([1]), tau, bound,
                )
        with self.assertRaisesRegex(ValueError, "outlier_count"):
            cnlcu_hard_score(
                mean[:1], torch.tensor([1]), torch.tensor([1]),
                torch.tensor([0]), torch.tensor([1]), 0.01, 2.0,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            cnlcu_hard_score(
                torch.tensor([math.inf]), torch.tensor([1]), torch.tensor([0]),
                torch.tensor([1]), torch.tensor([1]), 0.01, 2.0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_eq8_cpu_cuda_agree(self):
        arguments = (
            torch.tensor([0.5, 1.0], dtype=torch.float64),
            torch.tensor([3, 4]), torch.tensor([0, 1]),
            torch.tensor([3, 3]), torch.tensor([1, 2]), 0.01, 2.0,
        )
        cpu = cnlcu_hard_score(*arguments)
        cuda = cnlcu_hard_score(
            *(value.cuda() if torch.is_tensor(value) else value for value in arguments)
        )
        torch.testing.assert_close(cpu[0], cuda[0].cpu())
        torch.testing.assert_close(cpu[1], cuda[1].cpu())


if __name__ == "__main__":
    unittest.main()
