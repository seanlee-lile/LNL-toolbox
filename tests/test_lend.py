from __future__ import annotations

import unittest

import torch

from lnl_toolbox.selectors.lend import LENDSelector, dilute_labels


class LENDMathTest(unittest.TestCase):
    def test_batch_local_dilution_is_a_distribution(self) -> None:
        features = torch.eye(4)
        targets = torch.tensor([0, 1, 2, 1])
        diluted = dilute_labels(features, targets, neighbors=2, num_classes=3, diffusion_steps=2)
        self.assertEqual(tuple(diluted.shape), (4, 3))
        torch.testing.assert_close(diluted.sum(1), torch.ones(4))

    def test_selector_returns_complementary_mask(self) -> None:
        result = LENDSelector(neighbors=2, num_classes=3).select(features=torch.eye(4), noisy_targets=torch.tensor([0, 1, 2, 1]))
        torch.testing.assert_close(result.rejected_mask, ~result.selected_mask)


if __name__ == "__main__":
    unittest.main()
