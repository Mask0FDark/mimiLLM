"""Compare optional CUDA kernels with the reference Python backend."""

import unittest
from array import array

from mimillm import backend_python
from mimillm.backend_cuda import CudaBackend, is_available


@unittest.skipUnless(is_available(), "CUDA backend is unavailable")
class CudaBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cuda = CudaBackend()

    def assertClose(self, actual, expected, places: int = 4) -> None:
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=places)

    def test_device_information(self) -> None:
        self.assertTrue(self.cuda.device_name)
        self.assertGreater(self.cuda.device_memory, 0)
        self.assertGreater(self.cuda.multiprocessors, 0)

    def test_elementwise(self) -> None:
        left, right = [1.0, -2.0, 3.5], [0.5, 4.0, -1.0]
        self.assertClose(self.cuda.add(left, right), backend_python.add(left, right))
        self.assertClose(self.cuda.multiply(left, right), backend_python.multiply(left, right))
        self.assertClose(self.cuda.scalar_multiply(left, 0.25), backend_python.scalar_multiply(left, 0.25))

    def test_matmul_and_batched_matmul(self) -> None:
        left = [1, 2, 3, 4, 5, 6]
        right = [7, 8, 9, 10, 11, 12]
        self.assertClose(
            self.cuda.matmul(left, right, 2, 3, 2),
            backend_python.matmul(left, right, 2, 3, 2),
        )
        batched_left = [1, 2, 3, 4, 2, 0, 0, 2]
        batched_right = [1, 0, 0, 1, 1, 2, 3, 4]
        self.assertClose(
            self.cuda.batched_matmul(batched_left, batched_right, 2, 2, 2, 2),
            backend_python.batched_matmul(batched_left, batched_right, 2, 2, 2, 2),
        )

    def test_permute_broadcast_and_reductions(self) -> None:
        values = [float(index) for index in range(24)]
        self.assertClose(
            self.cuda.permute(values, (2, 3, 4), (1, 0, 2)),
            backend_python.permute(values, (2, 3, 4), (1, 0, 2)),
        )
        left, right = [1, 2, 3, 4, 5, 6], [10, 20, 30]
        self.assertClose(
            self.cuda.broadcast_binary(left, right, (2, 3), (3,), (2, 3), "mul"),
            [10, 40, 90, 40, 100, 180],
        )
        grad_left, grad_right = self.cuda.broadcast_binary_backward(
            left, right, [1] * 6, (2, 3), (3,), (2, 3), "mul",
        )
        self.assertClose(grad_left, [10, 20, 30, 10, 20, 30])
        self.assertClose(grad_right, [5, 7, 9])
        self.assertClose(self.cuda.sum_rows(values, 6, 4), backend_python.sum_rows(values, 6, 4))
        gradient = [0.25, -0.5, 1.0, 2.0, -1.0, 0.75]
        self.assertClose(
            self.cuda.sum_rows_backward(gradient, 6, 4),
            backend_python.sum_rows_backward(gradient, 6, 4),
        )

    def test_softmax_and_backward(self) -> None:
        values = [100.0, 101.0, 102.0, -2.0, 0.0, 2.0]
        probabilities = self.cuda.softmax_rows(values, 2, 3)
        expected = backend_python.softmax_rows(values, 2, 3)
        self.assertClose(probabilities, expected)
        upstream = [0.5, -1.0, 0.25, 2.0, 1.0, -0.5]
        self.assertClose(
            self.cuda.softmax_backward(probabilities, upstream, 2, 3),
            backend_python.softmax_backward(expected, upstream, 2, 3),
        )

    def test_relu_embedding_and_cross_entropy(self) -> None:
        values, upstream = [-2.0, 0.5, 3.0], [1.0, 2.0, 4.0]
        self.assertClose(self.cuda.relu(values), backend_python.relu(values))
        self.assertClose(
            self.cuda.relu_backward(values, upstream),
            backend_python.relu_backward(values, upstream),
        )
        table, indices = [0.1, 0.2, 1.0, 2.0, 3.0, 4.0], [1, 1, 2]
        self.assertClose(
            self.cuda.embedding_gather(table, indices, 3, 2),
            backend_python.embedding_gather(table, indices, 3, 2),
        )
        self.assertClose(
            self.cuda.embedding_scatter_add(indices, [1.0] * 6, 3, 2),
            backend_python.embedding_scatter_add(indices, [1.0] * 6, 3, 2),
        )
        logits, targets = [0.2, -0.1, 0.7, 0.8, 0.1, -0.2], [2, 0]
        self.assertAlmostEqual(
            self.cuda.cross_entropy(logits, targets, 2, 3),
            backend_python.cross_entropy(logits, targets, 2, 3), places=4,
        )
        self.assertClose(
            self.cuda.cross_entropy_backward(logits, targets, 2, 3),
            backend_python.cross_entropy_backward(logits, targets, 2, 3),
        )

    def test_weighted_cross_entropy(self) -> None:
        logits = [0.2, -0.1, 0.7, 0.8, 0.1, -0.2]
        targets, weights = [2, 0], [0.0, 2.0]
        loss, gradient = self.cuda.weighted_cross_entropy(logits, targets, weights, 2, 3)
        assert gradient is not None
        expected_loss = backend_python.cross_entropy(logits[3:], targets[1:], 1, 3)
        expected_gradient = array("f", [0.0, 0.0, 0.0, *backend_python.cross_entropy_backward(logits[3:], targets[1:], 1, 3)])
        self.assertAlmostEqual(loss, expected_loss, places=4)
        self.assertClose(gradient, expected_gradient)

    def test_adamw_and_gradient_helpers(self) -> None:
        parameter = array("f", [1.0, -2.0])
        gradient = array("f", [0.5, -0.25])
        first = array("f", [0.0, 0.0])
        second = array("f", [0.0, 0.0])
        self.cuda.adamw_update(
            parameter, gradient, first, second, learning_rate=0.1,
            beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, step=1,
        )
        self.assertClose(first, [0.05, -0.025], places=5)
        self.assertClose(second, [0.00025, 0.0000625], places=6)
        self.assertClose(parameter, [0.899, -1.898], places=4)
        self.assertAlmostEqual(self.cuda.sum_squares([3.0, 4.0]), 25.0, places=4)
        values = array("f", [2.0, -4.0])
        self.cuda.scale_inplace(values, 0.5)
        self.assertClose(values, [1.0, -2.0])


if __name__ == "__main__":
    unittest.main()
