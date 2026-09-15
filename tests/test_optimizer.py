"""Проверки оптимизаторов на простой выпуклой функции."""

import unittest
from array import array

from mimillm.optim import AdamW, SGD
from mimillm.parameter import Parameter


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

    def test_adamw_step_clipped_matches_explicit_clip_and_step(self) -> None:
        left = Parameter([1.0, -2.0])
        right = Parameter([1.0, -2.0])
        left.grad = array("f", [3.0, 4.0])
        right.grad = array("f", [3.0, 4.0])
        explicit = AdamW([left], learning_rate=0.01, weight_decay=0.0)
        combined = AdamW([right], learning_rate=0.01, weight_decay=0.0)
        expected_norm = explicit.clip_grad_norm(1.0)
        explicit.step()
        actual_norm = combined.step_clipped(1.0)
        self.assertAlmostEqual(actual_norm, expected_norm)
        for actual, expected in zip(right.data, left.data):
            self.assertAlmostEqual(actual, expected, places=6)

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

    def test_gradient_accumulation_matches_averaged_gradient(self) -> None:
        accumulated_value = Parameter([1.0], ())
        reference_value = Parameter([1.0], ())
        accumulated = AdamW(
            [accumulated_value],
            learning_rate=0.01,
            weight_decay=0.0,
            gradient_accumulation_steps=2,
        )
        reference = AdamW(
            [reference_value],
            learning_rate=0.01,
            weight_decay=0.0,
        )

        accumulated_value.grad = array("f", [2.0])
        self.assertEqual(accumulated.step_clipped(10.0), 0.0)
        self.assertEqual(accumulated.step_count, 0)
        self.assertAlmostEqual(accumulated_value.data[0], 1.0)
        accumulated.zero_grad()

        accumulated_value.grad = array("f", [4.0])
        accumulated_norm = accumulated.step_clipped(10.0)
        accumulated.zero_grad()

        reference_value.grad = array("f", [3.0])
        reference_norm = reference.step_clipped(10.0)
        reference.zero_grad()

        self.assertAlmostEqual(accumulated_norm, reference_norm)
        self.assertEqual(accumulated.step_count, 1)
        self.assertEqual(reference.step_count, 1)
        self.assertAlmostEqual(
            accumulated_value.data[0], reference_value.data[0], places=6,
        )
        self.assertAlmostEqual(
            accumulated.first_moments[0][0],
            reference.first_moments[0][0],
            places=6,
        )

    def test_selective_weight_decay_skips_excluded_parameter(self) -> None:
        matrix = Parameter([1.0] * 4, (2, 2))
        bias = Parameter([1.0] * 2, (2,))
        matrix.grad = array("f", [0.0] * 4)
        bias.grad = array("f", [0.0] * 2)
        optimizer = AdamW(
            [matrix, bias],
            learning_rate=0.1,
            weight_decay=0.1,
            decay_mask=[True, False],
        )
        optimizer.step()
        self.assertAlmostEqual(matrix.data[0], 0.99, places=6)
        self.assertAlmostEqual(bias.data[0], 1.0, places=6)

    def test_partial_accumulation_state_resumes_exactly(self) -> None:
        original_value = Parameter([1.0], ())
        original = AdamW(
            [original_value],
            learning_rate=0.01,
            weight_decay=0.0,
            gradient_accumulation_steps=2,
        )
        original_value.grad = array("f", [2.0])
        original.step()
        original.zero_grad()

        restored_value = Parameter([1.0], ())
        restored = AdamW([restored_value], learning_rate=0.5)
        restored.load_state_dict(original.state_dict())
        self.assertEqual(restored.accumulation_count, 1)
        self.assertEqual(restored.gradient_accumulation_steps, 2)

        original_value.grad = array("f", [4.0])
        restored_value.grad = array("f", [4.0])
        original.step()
        restored.step()
        self.assertAlmostEqual(
            original_value.data[0], restored_value.data[0], places=6,
        )
        self.assertEqual(original.step_count, restored.step_count)


if __name__ == "__main__":
    unittest.main()
