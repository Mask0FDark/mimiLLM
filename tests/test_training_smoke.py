"""Короткий полный шаг обучения и продолжение из checkpoint."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MINILLM_BACKEND", "python")

from minillm.checkpoint import load_checkpoint, save_checkpoint
from minillm.optim import AdamW
from minillm.transformer import DecoderTransformer, TransformerConfig


class TrainingSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TransformerConfig(
            context_length=4, d_model=4, n_layers=1, n_heads=1, d_mlp=8,
            batch_size=1, steps=3, learning_rate=0.02, weight_decay=0.0,
            warmup_steps=0, validation_interval=1, checkpoint_interval=1, seed=7,
        )

    def test_overfit_one_batch_and_resume(self) -> None:
        inputs, targets = [[257, 10, 11, 12]], [10, 11, 12, 258]
        model = DecoderTransformer(self.config)
        optimizer = AdamW(model.parameters(), self.config.learning_rate, weight_decay=0.0)
        initial = model(inputs).reshape(-1, 260).cross_entropy(targets).item()
        for _ in range(20):
            loss = model(inputs).reshape(-1, 260).cross_entropy(targets)
            loss.backward()
            optimizer.clip_grad_norm(1.0)
            optimizer.step()
            optimizer.zero_grad()
        final = model(inputs).reshape(-1, 260).cross_entropy(targets).item()
        self.assertLess(final, initial * 0.4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.bin"
            save_checkpoint(path, model, optimizer, config=self.config.to_dict(), step=20, seed=7)
            restored = DecoderTransformer(self.config)
            restored_optimizer = AdamW(restored.parameters(), 0.1)
            data = load_checkpoint(path, restored, restored_optimizer)
            resumed_loss = restored(inputs).reshape(-1, 260).cross_entropy(targets)
            resumed_loss.backward()
            restored_optimizer.step()
        self.assertEqual(data.step, 20)
        self.assertEqual(restored_optimizer.step_count, 21)


if __name__ == "__main__":
    unittest.main()
