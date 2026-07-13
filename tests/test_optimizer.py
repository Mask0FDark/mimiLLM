"""Проверки оптимизаторов на простой выпуклой функции."""

import unittest

from minillm.optim import AdamW, SGD
from minillm.parameter import Parameter


class OptimizerTests(unittest.TestCase):
    def test_sgd_decreases_quadratic(self) -> None:
        value = Parameter([5.0], ())
        optimizer = SGD([value], learning_rate=0.1)
        initial = (value * value).item()
        for _ in range(30):
            loss = value * value
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.assertLess((value * value).item(), initial * 1e-4)

    def test_gradient_clipping(self) -> None:
        value = Parameter([10.0], ())
        (value * value).backward()
        original = SGD([value], 0.1).clip_grad_norm(1.0)
        self.assertAlmostEqual(original, 20.0, places=5)
        self.assertAlmostEqual(value.grad[0], 1.0, places=5)  # type: ignore[index]

    def test_adamw_decreases_quadratic_and_restores_state(self) -> None:
        value = Parameter([5.0], ())
        optimizer = AdamW([value], learning_rate=0.2, weight_decay=0.0)
        initial = (value * value).item()
        for _ in range(40):
            (value * value).backward()
            optimizer.step()
            optimizer.zero_grad()
        self.assertLess((value * value).item(), initial * 0.02)
        state = optimizer.state_dict()
        restored = AdamW([Parameter([0.0], ())], learning_rate=0.01)
        restored.load_state_dict(state)
        self.assertEqual(restored.step_count, optimizer.step_count)
        self.assertEqual(restored.first_moments, optimizer.first_moments)


if __name__ == "__main__":
    unittest.main()
