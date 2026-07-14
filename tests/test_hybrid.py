"""Numerical tests for experimental CPU+GPU gradient aggregation."""

import unittest

from mimillm import backend_python
from mimillm.backend import backend_scope
from mimillm.hybrid import HybridDataParallel
from mimillm.transformer import DecoderTransformer, TransformerConfig
from mimillm.utils import flatten


class HybridDataParallelTests(unittest.TestCase):
    def test_split_batch_matches_single_model_loss_and_gradients(self) -> None:
        config = TransformerConfig(
            context_length=5,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_mlp=16,
            batch_size=4,
            steps=1,
            validation_interval=1,
            checkpoint_interval=1,
        )
        inputs = [
            [257, 10, 11, 12, 13],
            [257, 20, 21, 22, 23],
            [257, 30, 31, 32, 33],
            [257, 40, 41, 42, 43],
        ]
        targets = [
            [10, 11, 12, 13, 258],
            [20, 21, 22, 23, 258],
            [30, 31, 32, 33, 258],
            [40, 41, 42, 43, 258],
        ]
        weights = [
            [0.0, 0.5, 1.0, 2.0, 3.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
            [0.25, 0.5, 1.0, 1.0, 2.0],
            [0.0, 0.0, 1.0, 2.0, 2.0],
        ]
        expected_model = DecoderTransformer(config)
        with backend_scope(backend_python):
            expected_logits = expected_model(inputs)
            expected_loss = expected_logits.reshape(-1, config.vocab_size).cross_entropy(
                flatten(targets), weights=flatten(weights),  # type: ignore[arg-type]
            )
            expected_loss.backward()

        actual_model = DecoderTransformer(config)
        with HybridDataParallel(
            actual_model,
            cpu_batch_size=2,
            gpu_backend=backend_python,
            cpu_backend=backend_python,
        ) as hybrid:
            result = hybrid.forward_backward(inputs, targets, weights)

        self.assertAlmostEqual(result.loss, expected_loss.item(), places=5)
        self.assertEqual((result.gpu_batch_size, result.cpu_batch_size), (2, 2))
        for expected, actual in zip(
            expected_model.parameters(), actual_model.parameters(), strict=True,
        ):
            self.assertIsNotNone(expected.grad)
            self.assertIsNotNone(actual.grad)
            for expected_value, actual_value in zip(expected.grad, actual.grad):  # type: ignore[arg-type]
                self.assertAlmostEqual(expected_value, actual_value, places=5)


if __name__ == "__main__":
    unittest.main()
