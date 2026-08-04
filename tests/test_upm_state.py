import copy
import unittest

import torch

from lnl_toolbox.algorithms.upm import (
    ConfusingProbabilityState,
    UPMPhase,
    UPMState,
)


class UPMConfusingStateTest(unittest.TestCase):
    def test_noncontiguous_permutation_and_only_batch_updates(self) -> None:
        state = ConfusingProbabilityState(torch.tensor([30, 2, 11, 7]), 0.01)
        before = state.eta.clone()
        indices = torch.tensor([11, 2])
        state.update(
            indices,
            torch.tensor([[0.2, 0.8], [0.7, 0.3]]),
            torch.tensor([1, 0]), torch.tensor([0.6, 0.7]),
            learning_rate=0.1, epsilon=1e-4,
        )
        rows = state.resolve_rows(indices)
        untouched = torch.ones(4, dtype=torch.bool)
        untouched[rows] = False
        torch.testing.assert_close(state.eta[untouched], before[untouched], rtol=0, atol=0)
        self.assertTrue(bool((state.update_count[rows] == 1).all()))
        self.assertTrue(bool((state.update_count[untouched] == 0).all()))

    def test_duplicate_and_missing_indices_fail(self) -> None:
        state = ConfusingProbabilityState(torch.tensor([1, 5]), 0.1)
        with self.assertRaisesRegex(ValueError, "unique"):
            state.resolve_rows(torch.tensor([1, 1]))
        with self.assertRaises(KeyError):
            state.resolve_rows(torch.tensor([9]))

    def test_roundtrip_and_mapping_drift(self) -> None:
        state = ConfusingProbabilityState(torch.tensor([8, 2, 5]), 0.2)
        saved = state.state_dict()
        restored = ConfusingProbabilityState(torch.tensor([2, 5, 8]), 0.2)
        restored.load_state_dict(saved)
        torch.testing.assert_close(restored.eta, state.eta)
        changed = copy.deepcopy(saved)
        changed["canonical_sample_indices"][0] = 3
        with self.assertRaisesRegex(ValueError, "mapping"):
            restored.load_state_dict(changed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_state(self) -> None:
        state = ConfusingProbabilityState(torch.tensor([1, 4]), 0.1, device="cuda")
        self.assertEqual(state.eta.device.type, "cuda")
        saved = state.state_dict()
        self.assertEqual(saved["eta"].device.type, "cpu")


class UPMPhaseStateTest(unittest.TestCase):
    def test_illegal_phase_transition_fails(self) -> None:
        state = UPMState()
        with self.assertRaisesRegex(ValueError, "illegal"):
            state.advance(UPMPhase.PSI_READY)


if __name__ == "__main__":
    unittest.main()
