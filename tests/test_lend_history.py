import unittest

import torch

from lnl_toolbox.algorithms.lend.history import LENDLabelHistory


class LENDHistoryTest(unittest.TestCase):
    def test_arbitrary_indices_first_observation_and_equation_five(self):
        history = LENDLabelHistory(torch.tensor([40, 10, 90]), 2)
        indices = torch.tensor([90, 10])
        current = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
        first = history.propose(indices, current, epoch=0, beta=0.9)
        torch.testing.assert_close(first.values, current)
        history.commit(first)
        next_value = torch.tensor([[0.6, 0.4], [0.1, 0.9]])
        second = history.propose(indices.flip(0), next_value, epoch=1, beta=0.75)
        expected = 0.25 * next_value + 0.75 * current.flip(0)
        torch.testing.assert_close(second.values, expected)
        history.commit(second)
        untouched = torch.searchsorted(history.canonical_sample_indices, torch.tensor([40]))
        self.assertFalse(history.initialized[untouched].item())

    def test_duplicate_same_epoch_missing_index_and_mapping_drift_fail(self):
        history = LENDLabelHistory(torch.tensor([2, 7]), 2)
        proposal = history.propose(torch.tensor([2]), torch.tensor([[1., 0.]]), epoch=0, beta=.9)
        history.commit(proposal)
        with self.assertRaisesRegex(ValueError, "twice"):
            history.propose(torch.tensor([2]), torch.tensor([[1., 0.]]), epoch=0, beta=.9)
        with self.assertRaisesRegex(ValueError, "missing"):
            history.propose(torch.tensor([8]), torch.tensor([[1., 0.]]), epoch=1, beta=.9)
        with self.assertRaisesRegex(ValueError, "beta"):
            history.propose(torch.tensor([7]), torch.tensor([[1., 0.]]), epoch=1, beta=1.1)
        state = history.state_dict()
        other = LENDLabelHistory(torch.tensor([2, 8]), 2)
        with self.assertRaisesRegex(ValueError, "mapping changed"):
            other.load_state_dict(state)

    def test_checkpoint_roundtrip_is_deep(self):
        history = LENDLabelHistory(torch.tensor([5, 1]), 3)
        proposal = history.propose(torch.tensor([5]), torch.tensor([[.2, .3, .4]]), epoch=0, beta=.9)
        history.commit(proposal)
        state = history.state_dict()
        restored = LENDLabelHistory(torch.tensor([1, 5]), 3)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.values, history.values))
        state["values"].zero_()
        self.assertFalse(torch.equal(restored.values, state["values"]))


if __name__ == "__main__": unittest.main()
