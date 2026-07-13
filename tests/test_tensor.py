"""Проверки формы, памяти и операций Tensor."""

import unittest

from mimillm.tensor import Tensor


class TensorTests(unittest.TestCase):
    def assertValuesAlmostEqual(self, actual: Tensor, expected: list[float], places: int = 5) -> None:
        self.assertEqual(len(actual.data), len(expected))
        for left, right in zip(actual.data, expected):
            self.assertAlmostEqual(left, right, places=places)

    def test_shape_strides_and_indexing(self) -> None:
        tensor = Tensor(range(6), (2, 3))
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.strides, (3, 1))
        self.assertEqual(tensor[1, 2], 5.0)
        self.assertEqual(tensor[1].shape, (3,))
        self.assertValuesAlmostEqual(tensor[1], [3, 4, 5])

    def test_bad_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "требует"):
            Tensor([1, 2, 3], (2, 2))

    def test_reshape_and_inferred_dimension(self) -> None:
        tensor = Tensor(range(6), (2, 3)).reshape(3, -1)
        self.assertEqual(tensor.shape, (3, 2))
        self.assertValuesAlmostEqual(tensor, [0, 1, 2, 3, 4, 5])

    def test_transpose_is_contiguous_copy(self) -> None:
        tensor = Tensor(range(6), (2, 3)).transpose(0, 1)
        self.assertEqual(tensor.shape, (3, 2))
        self.assertEqual(tensor.strides, (2, 1))
        self.assertValuesAlmostEqual(tensor, [0, 3, 1, 4, 2, 5])

    def test_elementwise_and_broadcasting(self) -> None:
        left = Tensor([1, 2, 3, 4], (2, 2))
        right = Tensor([10, 20], (2,))
        self.assertValuesAlmostEqual(left + right, [11, 22, 13, 24])
        self.assertValuesAlmostEqual(left * 2 - 1, [1, 3, 5, 7])
        self.assertValuesAlmostEqual(left / 2, [0.5, 1, 1.5, 2])

    def test_reductions(self) -> None:
        tensor = Tensor([1, 2, 3, 4], (2, 2))
        self.assertAlmostEqual(tensor.sum().item(), 10.0)
        self.assertValuesAlmostEqual(tensor.sum(axis=1), [3, 7])
        self.assertValuesAlmostEqual(tensor.mean(axis=0), [2, 3])

    def test_matmul(self) -> None:
        left = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        right = Tensor([7, 8, 9, 10, 11, 12], (3, 2))
        result = left @ right
        self.assertEqual(result.shape, (2, 2))
        self.assertValuesAlmostEqual(result, [58, 64, 139, 154])

    def test_batched_matmul(self) -> None:
        left = Tensor([1, 2, 3, 4, 2, 0, 0, 2], (2, 2, 2))
        right = Tensor([1, 0, 0, 1, 1, 2, 3, 4], (2, 2, 2))
        self.assertValuesAlmostEqual(left.batched_matmul(right), [1, 2, 3, 4, 2, 4, 6, 8])

    def test_softmax_rows_sum_to_one(self) -> None:
        result = Tensor([1, 2, 3, -1, 0, 1], (2, 3)).softmax()
        self.assertAlmostEqual(sum(result.data[:3]), 1.0, places=6)
        self.assertAlmostEqual(sum(result.data[3:]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
