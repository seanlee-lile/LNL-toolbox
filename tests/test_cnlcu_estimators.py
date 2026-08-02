import copy
import math
import unittest

import torch

from lnl_toolbox.algorithms.cnlcu import (
    PeerLossHistory, cnlcu_soft_score, soft_influence, soft_robust_mean,
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
        self.assertEqual(history.selected_count.sum().item(), 1)

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


if __name__ == "__main__":
    unittest.main()
