import unittest

import numpy as np

from lnl_toolbox.algorithms import coteaching_exchange, remember_rate
from lnl_toolbox.algorithms.coteaching import (
    coteaching_exchange as package_coteaching_exchange,
)
from lnl_toolbox.algorithms.coteaching import remember_rate as package_remember_rate


class CoTeachingTest(unittest.TestCase):
    def test_peer_exchange(self) -> None:
        losses_a = np.array([0.1, 0.8, 0.2, 0.7])
        losses_b = np.array([0.9, 0.1, 0.8, 0.2])
        update_a, update_b = coteaching_exchange(losses_a, losses_b, keep_rate=0.5)
        np.testing.assert_array_equal(update_a, np.array([1, 3]))
        np.testing.assert_array_equal(update_b, np.array([0, 2]))
        self.assertAlmostEqual(remember_rate(10, 0.4, 10), 0.6)

    def test_legacy_package_imports_preserve_behavior(self) -> None:
        losses_a = np.array([0.1, 0.8, 0.2, 0.7])
        losses_b = np.array([0.9, 0.1, 0.8, 0.2])
        update_a, update_b = package_coteaching_exchange(
            losses_a, losses_b, keep_rate=0.5
        )
        np.testing.assert_array_equal(update_a, np.array([1, 3]))
        np.testing.assert_array_equal(update_b, np.array([0, 2]))
        self.assertAlmostEqual(package_remember_rate(10, 0.4, 10), 0.6)


if __name__ == "__main__":
    unittest.main()
