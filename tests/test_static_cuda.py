"""Correctness checks for the fixed-shape CUDA Graph training path."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from mimillm.backend import get_backend, reset_backend
from mimillm.backend_cuda import is_available as cuda_is_available
from mimillm.optim import AdamW
from mimillm.static_cuda import compile_static_cuda_training
from mimillm.training import train_model
from mimillm.transformer import DecoderTransformer, TransformerConfig
from mimillm.utils import flatten


@unittest.skipUnless(cuda_is_available(), "CUDA backend is unavailable")
class StaticCudaTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MIMILLM_BACKEND"] = "cuda"
        reset_backend()

    def test_dynamic_inputs_and_loss_weights_match_eager_training(self) -> None:
        config = TransformerConfig(
            context_length=8,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_mlp=32,
            batch_size=2,
            steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_steps=0,
            validation_interval=1,
            checkpoint_interval=1,
            seed=42,
        )
        batches = (
            (
                [[257, 1, 2, 3, 4, 5, 6, 7], [257, 8, 9, 10, 11, 12, 13, 14]],
                [[1, 2, 3, 4, 5, 6, 7, 258], [8, 9, 10, 11, 12, 13, 14, 258]],
                [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]],
            ),
            (
                [[257, 20, 21, 22, 23, 24, 25, 26], [257, 30, 31, 32, 33, 34, 35, 36]],
                [[20, 21, 22, 23, 24, 25, 26, 258], [30, 31, 32, 33, 34, 35, 36, 258]],
                [[0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]],
            ),
        )
        eager_model = DecoderTransformer(config)
        static_model = DecoderTransformer(config)
        eager_optimizer = AdamW(
            eager_model.parameters(), 1e-3, weight_decay=0.0,
        )
        static_optimizer = AdamW(
            static_model.parameters(), 1e-3, weight_decay=0.0,
        )
        trainer = compile_static_cuda_training(
            static_model, static_optimizer, *batches[0],
        )
        try:
            for step in range(4):
                inputs, targets, weights = batches[step % len(batches)]
                logits = eager_model(inputs)
                loss = logits.reshape(-1, config.vocab_size).cross_entropy(
                    flatten(targets), weights=flatten(weights),
                )
                expected_loss = loss.item()
                loss.backward()
                expected_norm = eager_optimizer.clip_grad_norm(1.0)
                eager_optimizer.step()
                eager_optimizer.zero_grad()
                get_backend().synchronize()

                actual = trainer.step(
                    inputs,
                    targets,
                    loss_weights=weights,
                    clip_grad_norm=1.0,
                )
                self.assertAlmostEqual(expected_loss, actual.loss, places=5)
                self.assertIsNotNone(actual.gradient_norm)
                self.assertAlmostEqual(
                    expected_norm, actual.gradient_norm or 0.0, places=4,
                )
        finally:
            trainer.close()

        differences = [
            abs(left - right)
            for eager_parameter, static_parameter in zip(
                eager_model.parameters(), static_model.parameters(),
            )
            for left, right in zip(eager_parameter.data, static_parameter.data)
        ]
        self.assertLess(max(differences), 2e-4)
        self.assertLess(sum(differences) / len(differences), 2e-7)

    def test_train_model_pads_short_batches_for_graph_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text(
                "Вопрос: Привет?\nОтвет: Привет!\n\n"
                "Вопрос: Как тебя зовут?\nОтвет: mimiLLM.\n",
                encoding="utf-8",
            )
            (root / "validation.txt").write_text(
                "Вопрос: Кто ты?\nОтвет: Учебная модель.\n",
                encoding="utf-8",
            )
            config = TransformerConfig(
                question_train_path="train.txt",
                question_validation_path="validation.txt",
                text_ratio=0.0,
                context_length=16,
                d_model=8,
                n_layers=1,
                n_heads=2,
                d_mlp=16,
                batch_size=1,
                steps=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                warmup_steps=0,
                validation_interval=2,
                checkpoint_interval=2,
                seed=7,
            )
            result = train_model(
                config,
                base_dir=root,
                output_dir="weights",
            )
            self.assertEqual(result.step, 2)
            self.assertTrue((root / "weights" / "model.safetensors").is_file())
