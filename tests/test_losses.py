import unittest

import numpy as np

from lnl_toolbox.losses import cross_entropy, generalized_cross_entropy


class LossTest(unittest.TestCase):
    def test_gce_approaches_ce(self) -> None:
        probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
        targets = np.array([0, 1])
        np.testing.assert_allclose(
            generalized_cross_entropy(probabilities, targets, q=1e-7),
            cross_entropy(probabilities, targets),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()

