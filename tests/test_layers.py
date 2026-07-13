"""Базовые тесты модулей."""

import random
import unittest

from mimillm.layers import Linear, ReLU
from mimillm.tensor import Tensor


class LayerTests(unittest.TestCase):
    def test_linear_shape_and_parameters(self) -> None:
        layer = Linear(3, 4, rng=random.Random(5))
        output = layer(Tensor(range(6), (2, 3)))
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual([name for name, _ in layer.named_parameters()], ["weight", "bias"])

    def test_linear_preserves_leading_dimensions(self) -> None:
        layer = Linear(3, 2, rng=random.Random(5))
        output = layer(Tensor(range(12), (2, 2, 3)))
        self.assertEqual(output.shape, (2, 2, 2))

    def test_relu(self) -> None:
        output = ReLU()(Tensor([-2, 0, 3], (3,)))
        self.assertEqual(output.tolist(), [0.0, 0.0, 3.0])


if __name__ == "__main__":
    unittest.main()

