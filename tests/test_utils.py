"""Tests for learning-rate schedules and small utilities."""

import unittest

from mimillm.utils import learning_rate_at


class UtilityTests(unittest.TestCase):
    def test_warmup_reaches_the_base_learning_rate(self) -> None:
        self.assertEqual(learning_rate_at(5, 100, 0.01, 10), 0.005)
        self.assertEqual(learning_rate_at(10, 100, 0.01, 10), 0.01)

    def test_cosine_schedule_reaches_the_configured_floor(self) -> None:
        middle = learning_rate_at(55, 100, 0.01, 10, schedule="cosine", min_ratio=0.1)
        self.assertAlmostEqual(middle, 0.0055)
        self.assertAlmostEqual(
            learning_rate_at(100, 100, 0.01, 10, schedule="cosine", min_ratio=0.1),
            0.001,
        )

    def test_linear_and_constant_schedules(self) -> None:
        self.assertAlmostEqual(
            learning_rate_at(55, 100, 0.01, 10, schedule="linear", min_ratio=0.1),
            0.0055,
        )
        self.assertEqual(
            learning_rate_at(100, 100, 0.01, 10, schedule="constant"),
            0.01,
        )


if __name__ == "__main__":
    unittest.main()
