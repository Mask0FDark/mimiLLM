"""Compare optional CUDA kernels with the reference Python backend."""

import math
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
        stats = self.cuda.memory_stats()
        self.assertGreater(stats["pool_limit_bytes"], 0)
        self.assertLessEqual(stats["pool_bytes"], stats["pool_limit_bytes"])

    def test_workspace_pool_is_bounded_for_varying_shapes(self) -> None:
        self.cuda.empty_cache()
        for count in range(257, 4097, 31):
            result = self.cuda.scalar_multiply([1.0] * count, 0.5)
            self.assertAlmostEqual(result[0], 0.5)
        stats = self.cuda.memory_stats()
        self.assertLessEqual(stats["pool_bytes"], stats["pool_limit_bytes"])
        self.assertLessEqual(stats["pool_blocks"], 4)
        self.cuda.empty_cache()
        self.assertEqual(self.cuda.memory_stats()["pool_bytes"], 0)

    def test_elementwise(self) -> None:
        left, right = [1.0, -2.0, 3.5], [0.5, 4.0, -1.0]
        self.assertClose(self.cuda.add(left, right), backend_python.add(left, right))
        self.assertClose(self.cuda.multiply(left, right), backend_python.multiply(left, right))
        self.assertClose(self.cuda.scalar_multiply(left, 0.25), backend_python.scalar_multiply(left, 0.25))
        matrix, bias = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10.0, 20.0, 30.0]
        self.assertClose(
            self.cuda.add_row_vector(matrix, bias, 2, 3),
            [11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        )
        self.assertClose(
            self.cuda.sum_columns(matrix, 2, 3),
            [5.0, 7.0, 9.0],
        )

    def test_device_resident_outputs_chain_without_early_host_copy(self) -> None:
        first = self.cuda.add([1.0, 2.0], [3.0, 4.0])
        self.assertFalse(getattr(first, "_host_current", True))
        second = self.cuda.multiply(first, [2.0, 3.0])
        self.assertFalse(getattr(second, "_host_current", True))
        self.assertClose(second, [8.0, 18.0])
        self.assertTrue(getattr(second, "_host_current", False))

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
        grad_output = [1.0, 2.0, 3.0, 4.0]
        self.assertClose(
            self.cuda.matmul_backward_left(
                grad_output, right, 1, 2, 3, 2,
            ),
            backend_python.matmul(
                grad_output,
                backend_python.permute(right, (3, 2), (1, 0)),
                2, 2, 3,
            ),
        )
        self.assertClose(
            self.cuda.matmul_backward_right(
                left, grad_output, 1, 2, 3, 2,
            ),
            backend_python.matmul(
                backend_python.permute(left, (2, 3), (1, 0)),
                grad_output, 3, 2, 2,
            ),
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

    def test_fused_causal_softmax_and_backward(self) -> None:
        values = [
            1.0, 9.0, 8.0,
            1.0, 2.0, 7.0,
            1.0, 2.0, 3.0,
        ]
        scale = 0.5
        masked = [
            value * scale if column <= row else -1.0e9
            for row in range(3)
            for column, value in enumerate(values[row * 3:(row + 1) * 3])
        ]
        expected = backend_python.softmax_rows(masked, 3, 3)
        actual = self.cuda.causal_softmax_rows(values, 3, 3, 3, scale)
        self.assertClose(actual, expected)
        upstream = [0.1, 0.2, 0.3, -0.5, 0.7, 1.0, 1.0, -2.0, 0.5]
        expected_gradient = [
            value * scale
            for value in backend_python.softmax_backward(
                expected, upstream, 3, 3,
            )
        ]
        self.assertClose(
            self.cuda.causal_softmax_backward(
                actual, upstream, 3, 3, scale,
            ),
            expected_gradient,
        )

    def test_fused_rms_norm_and_backward(self) -> None:
        values = [1.0, -2.0, 0.5, 3.0, -1.0, 2.0]
        weight = [0.5, 1.5, -0.75]
        upstream = [0.2, -0.4, 0.7, 1.0, -0.5, 0.25]
        rows, columns, epsilon = 2, 3, 1e-5
        expected_output = []
        expected_input_gradient = []
        expected_weight_gradient = [0.0] * columns
        for row in range(rows):
            start = row * columns
            row_values = values[start:start + columns]
            row_upstream = upstream[start:start + columns]
            inverse_rms = 1.0 / math.sqrt(
                sum(value * value for value in row_values) / columns + epsilon
            )
            dot = sum(
                gradient * scale * value
                for gradient, scale, value in zip(
                    row_upstream, weight, row_values,
                )
            )
            correction = dot * inverse_rms ** 3 / columns
            for column in range(columns):
                expected_output.append(
                    row_values[column] * weight[column] * inverse_rms
                )
                expected_input_gradient.append(
                    row_upstream[column] * weight[column] * inverse_rms
                    - row_values[column] * correction
                )
                expected_weight_gradient[column] += (
                    row_upstream[column] * row_values[column] * inverse_rms
                )
        self.assertClose(
            self.cuda.rms_norm(values, weight, rows, columns, epsilon),
            expected_output,
        )
        input_gradient, weight_gradient = self.cuda.rms_norm_backward(
            values, weight, upstream, rows, columns, epsilon,
        )
        self.assertClose(input_gradient, expected_input_gradient)
        self.assertClose(weight_gradient, expected_weight_gradient)

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
        self.assertAlmostEqual(
            self.cuda.global_sum_squares(([3.0], [4.0])),
            25.0,
            places=4,
        )
        values = array("f", [2.0, -4.0])
        self.cuda.scale_inplace(values, 0.5)
        self.assertClose(values, [1.0, -2.0])

    def test_prepared_optimizer_state_is_lazy_and_host_mutations_refresh_device(self) -> None:
        parameters, first, second = self.cuda.prepare_optimizer_state(
            [array("f", [1.0, -2.0])],
            [array("f", [0.0, 0.0])],
            [array("f", [0.0, 0.0])],
        )
        parameter = parameters[0]
        self.cuda.adamw_update(
            parameter, array("f", [0.5, -0.25]), first[0], second[0],
            learning_rate=0.1, beta1=0.9, beta2=0.999,
            epsilon=1e-8, weight_decay=0.01, step=1,
        )
        self.assertFalse(getattr(parameter, "_host_current", True))
        parameter[0] = 3.0
        self.assertFalse(getattr(parameter, "_device_current", True))
        result = self.cuda.scalar_multiply(parameter, 2.0)
        self.assertClose(result, [6.0, -3.796], places=3)


if __name__ == "__main__":
    unittest.main()
