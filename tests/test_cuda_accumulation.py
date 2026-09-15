"""CUDA-регрессии для accumulation и выборочного AdamW decay."""

from __future__ import annotations

import os
import unittest

from mimillm.backend import get_backend, reset_backend
from mimillm.backend_cuda import is_available as cuda_is_available
from mimillm.optim import AdamW
from mimillm.transformer import DecoderTransformer, TransformerConfig
from mimillm.utils import flatten


@unittest.skipUnless(cuda_is_available(), "CUDA backend is unavailable")
class CudaAccumulationTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MIMILLM_BACKEND"] = "cuda"
        reset_backend()

    def tearDown(self) -> None:
        reset_backend()

    def test_two_microsteps_match_one_averaged_update(self) -> None:
        common = dict(
            context_length=8,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_mlp=32,
            batch_size=1,
            steps=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_steps=0,
            validation_interval=2,
            checkpoint_interval=2,
            cuda_tf32=False,
            seed=42,
        )
        accumulated_config = TransformerConfig(
            **common,
            gradient_accumulation_steps=2,
        )
        reference_config = TransformerConfig(**common)
        accumulated_model = DecoderTransformer(accumulated_config)
        reference_model = DecoderTransformer(reference_config)
        reference_model.load_state_dict(accumulated_model.state_dict())
        accumulated_optimizer = AdamW(
            accumulated_model.parameters(),
            learning_rate=1e-3,
            weight_decay=0.0,
        )
        reference_optimizer = AdamW(
            reference_model.parameters(),
            learning_rate=1e-3,
            weight_decay=0.0,
        )

        batches = (
            ([[257, 1, 2, 3, 4, 5, 6, 7]], [[1, 2, 3, 4, 5, 6, 7, 258]]),
            ([[257, 8, 9, 10, 11, 12, 13, 14]], [[8, 9, 10, 11, 12, 13, 14, 258]]),
        )

        accumulated_gradients: list[list[float]] = []
        for inputs, targets in batches:
            loss = accumulated_model(inputs).reshape(-1, 260).cross_entropy(
                flatten(targets)
            )
            loss.backward()
            accumulated_gradients.append([
                float(value)
                for parameter in accumulated_model.parameters()
                for value in (parameter.grad or [])
            ])
            accumulated_optimizer.step_clipped(1e9)
            accumulated_optimizer.zero_grad()

        offset = 0
        for parameter in reference_model.parameters():
            count = parameter.numel
            parameter.grad = type(parameter.data)(
                "f",
                (
                    (accumulated_gradients[0][offset + index]
                     + accumulated_gradients[1][offset + index]) / 2.0
                    for index in range(count)
                ),
            )
            offset += count
        reference_optimizer.step_clipped(1e9)
        reference_optimizer.zero_grad()
        get_backend().synchronize()

        self.assertEqual(accumulated_optimizer.step_count, 1)
        self.assertEqual(reference_optimizer.step_count, 1)
        differences = [
            abs(left - right)
            for accumulated_parameter, reference_parameter in zip(
                accumulated_model.parameters(), reference_model.parameters(),
            )
            for left, right in zip(
                accumulated_parameter.data, reference_parameter.data,
            )
        ]
        self.assertLess(max(differences), 3e-5)

    def test_model_hints_disable_decay_for_one_dimensional_parameters(self) -> None:
        config = TransformerConfig(
            context_length=8,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_mlp=32,
            batch_size=1,
            steps=2,
            gradient_accumulation_steps=2,
            weight_decay_exclude_1d=True,
            validation_interval=2,
            checkpoint_interval=2,
        )
        model = DecoderTransformer(config)
        optimizer = AdamW(model.parameters(), weight_decay=0.1)
        self.assertEqual(optimizer.gradient_accumulation_steps, 2)
        self.assertTrue(any(optimizer.decay_mask))
        self.assertTrue(any(not value for value in optimizer.decay_mask))
        for parameter, enabled in zip(optimizer.parameters, optimizer.decay_mask):
            self.assertEqual(enabled, parameter.ndim >= 2)


if __name__ == "__main__":
    unittest.main()
