"""Проверки оптимизаторов на простой выпуклой функции."""

import unittest

from minillm.optim import SGD
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


if __name__ == "__main__":
    unittest.main()
