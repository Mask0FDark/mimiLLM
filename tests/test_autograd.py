"""Численные и структурные проверки динамического autograd."""

import random
import unittest

from mimillm.autograd import gradcheck
from mimillm.layers import Linear
from mimillm.tensor import Tensor, is_grad_enabled, no_grad


class AutogradTests(unittest.TestCase):
    def test_no_grad_disables_graph_and_restores_state(self) -> None:
        value = Tensor([2.0], requires_grad=True)
        self.assertTrue(is_grad_enabled())
        with no_grad():
            self.assertFalse(is_grad_enabled())
            output = value * value + 1.0
            self.assertFalse(output.requires_grad)
            self.assertEqual(output.parents, ())
        self.assertTrue(is_grad_enabled())
        tracked = value * value
        self.assertTrue(tracked.requires_grad)

    def assertGradcheck(self, function, tensors, tolerance: float = 4e-3) -> None:
        passed, error = gradcheck(function, tensors, tolerance=tolerance)
        self.assertTrue(passed, f"максимальная ошибка градиента {error}")

    def test_addition_and_reused_tensor(self) -> None:
        value = Tensor([2.0], (), requires_grad=True)
        result = value * value + value
        result.backward()
        self.assertAlmostEqual(value.grad[0], 5.0, places=5)  # type: ignore[index]

    def test_multiplication_mean_and_relu_gradcheck(self) -> None:
        left = Tensor([-1.2, 0.7, 2.0], (3,), requires_grad=True)
        right = Tensor([0.5, -0.3, 1.1], (3,), requires_grad=True)
        self.assertGradcheck(lambda: (left * right).relu().mean(), [left, right])

    def test_matmul_gradcheck(self) -> None:
        left = Tensor([0.2, -0.4, 0.7, 1.1], (2, 2), requires_grad=True)
        right = Tensor([0.5, 0.3, -0.2, 0.8], (2, 2), requires_grad=True)
        self.assertGradcheck(lambda: (left @ right).mean(), [left, right])

    def test_cross_entropy_gradcheck(self) -> None:
        logits = Tensor([0.2, -0.1, 0.7, 0.8, 0.1, -0.2], (2, 3), requires_grad=True)
        self.assertGradcheck(lambda: logits.cross_entropy([2, 0]), [logits], tolerance=2e-3)

    def test_weighted_cross_entropy_gradcheck(self) -> None:
        logits = Tensor([0.2, -0.1, 0.7, 0.8, 0.1, -0.2], (2, 3), requires_grad=True)
        self.assertGradcheck(
            lambda: logits.cross_entropy([2, 0], weights=[0.25, 1.0]),
            [logits], tolerance=2e-3,
        )

    def test_embedding_gradcheck_and_repeated_index(self) -> None:
        table = Tensor([0.2, -0.1, 0.7, 0.8, 0.1, -0.2], (3, 2), requires_grad=True)
        self.assertGradcheck(lambda: table.embedding([1, 1, 2]).mean(), [table])

    def test_linear_gradcheck(self) -> None:
        layer = Linear(2, 3, rng=random.Random(3))
        inputs = Tensor([0.1, -0.2, 0.7, 0.3], (2, 2), requires_grad=True)
        self.assertGradcheck(lambda: layer(inputs).mean(), [inputs, layer.weight, layer.bias])

    def test_zero_grad(self) -> None:
        value = Tensor([2.0], (), requires_grad=True)
        (value * value).backward()
        self.assertIsNotNone(value.grad)
        value.zero_grad()
        self.assertIsNone(value.grad)


if __name__ == "__main__":
    unittest.main()
