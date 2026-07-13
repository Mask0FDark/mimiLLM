"""Сравнение C++ kernels с эталонной Python-реализацией."""

import unittest
from array import array

from minillm import backend_python
from minillm.backend_cpp import CppBackend, is_available


@unittest.skipUnless(is_available(), "C++ backend ещё не собран")
class CppBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cpp = CppBackend()

    def assertClose(self, actual, expected, places: int = 5) -> None:
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=places)

    def test_elementwise(self) -> None:
        left, right = [1.0, -2.0, 3.5], [0.5, 4.0, -1.0]
        self.assertClose(self.cpp.add(left, right), backend_python.add(left, right))
        self.assertClose(self.cpp.multiply(left, right), backend_python.multiply(left, right))
        self.assertClose(self.cpp.scalar_multiply(left, 0.25), backend_python.scalar_multiply(left, 0.25))

    def test_matmul(self) -> None:
        left = [1, 2, 3, 4, 5, 6]
        right = [7, 8, 9, 10, 11, 12]
        self.assertClose(self.cpp.matmul(left, right, 2, 3, 2), backend_python.matmul(left, right, 2, 3, 2))

    def test_batched_matmul(self) -> None:
        left = [1, 2, 3, 4, 2, 0, 0, 2]
        right = [1, 0, 0, 1, 1, 2, 3, 4]
        self.assertClose(
            self.cpp.batched_matmul(left, right, 2, 2, 2, 2),
            backend_python.batched_matmul(left, right, 2, 2, 2, 2),
        )

    def test_softmax(self) -> None:
        values = [100.0, 101.0, 102.0, -2.0, 0.0, 2.0]
        self.assertClose(self.cpp.softmax_rows(values, 2, 3), backend_python.softmax_rows(values, 2, 3))

    def test_relu_embedding_and_cross_entropy(self) -> None:
        values, upstream = [-2.0, 0.5, 3.0], [1.0, 2.0, 4.0]
        self.assertClose(self.cpp.relu(values), backend_python.relu(values))
        self.assertClose(
            self.cpp.relu_backward(values, upstream),
            backend_python.relu_backward(values, upstream),
        )
        table, indices = [0.1, 0.2, 1.0, 2.0, 3.0, 4.0], [1, 1, 2]
        gathered = backend_python.embedding_gather(table, indices, 3, 2)
        self.assertClose(self.cpp.embedding_gather(table, indices, 3, 2), gathered)
        gradient = [1.0] * 6
        self.assertClose(
            self.cpp.embedding_scatter_add(indices, gradient, 3, 2),
            backend_python.embedding_scatter_add(indices, gradient, 3, 2),
        )
        logits, targets = [0.2, -0.1, 0.7, 0.8, 0.1, -0.2], [2, 0]
        self.assertAlmostEqual(
            self.cpp.cross_entropy(logits, targets, 2, 3),
            backend_python.cross_entropy(logits, targets, 2, 3), places=5,
        )
        self.assertClose(
            self.cpp.cross_entropy_backward(logits, targets, 2, 3),
            backend_python.cross_entropy_backward(logits, targets, 2, 3),
        )

    def test_adamw_kernel_matches_formula(self) -> None:
        parameter = array("f", [1.0, -2.0])
        gradient = array("f", [0.5, -0.25])
        first = array("f", [0.0, 0.0])
        second = array("f", [0.0, 0.0])
        self.cpp.adamw_update(
            parameter, gradient, first, second, learning_rate=0.1,
            beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, step=1,
        )
        self.assertClose(first, [0.05, -0.025], places=6)
        self.assertClose(second, [0.00025, 0.0000625], places=7)
        self.assertClose(parameter, [0.899, -1.898], places=5)

    def test_thread_setting_and_invalid_value(self) -> None:
        self.cpp.set_num_threads(1)
        self.assertEqual(self.cpp.num_threads, 1)
        self.cpp.set_num_threads(3)
        self.assertEqual(self.cpp.num_threads, 3)
        with self.assertRaisesRegex(RuntimeError, "positive"):
            self.cpp.set_num_threads(0)

    def test_single_and_multi_thread_results_match(self) -> None:
        left = [float((index * 7) % 13 - 6) / 7 for index in range(32 * 48)]
        right = [float((index * 5) % 17 - 8) / 9 for index in range(48 * 24)]
        self.cpp.set_num_threads(1)
        single = self.cpp.matmul(left, right, 32, 48, 24)
        self.cpp.set_num_threads(4)
        multi = self.cpp.matmul(left, right, 32, 48, 24)
        self.assertClose(single, multi, places=6)

    def test_thread_pool_survives_many_sequential_jobs(self) -> None:
        self.cpp.set_num_threads(4)
        left = [float(index % 5) for index in range(8 * 8)]
        right = [float(index % 7) for index in range(8 * 8)]
        expected = backend_python.matmul(left, right, 8, 8, 8)
        for _ in range(1000):
            actual = self.cpp.matmul(left, right, 8, 8, 8)
        self.assertClose(actual, expected)

    def test_bad_shape_is_rejected_in_python(self) -> None:
        with self.assertRaisesRegex(ValueError, "не соответствует"):
            self.cpp.matmul([1, 2], [1, 2], 2, 2, 2)


if __name__ == "__main__":
    unittest.main()
