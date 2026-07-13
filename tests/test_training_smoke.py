"""Короткий полный шаг обучения и продолжение из checkpoint."""

import os
import json
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MIMILLM_BACKEND", "python")

from mimillm.checkpoint import load_checkpoint, save_checkpoint
from mimillm.optim import AdamW
from mimillm.transformer import DecoderTransformer, TransformerConfig
from mimillm import load_model, train_from_config


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

    def test_project_config_paths_export_standard_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "data/text/train", "data/text/validation",
                "data/question/train", "data/question/validation",
            ):
                (root / relative).mkdir(parents=True)
            (root / "data/text/train/text.txt").write_text(
                "A small training text for the language model.", encoding="utf-8"
            )
            (root / "data/text/validation/text.txt").write_text(
                "A separate validation text.", encoding="utf-8"
            )
            (root / "data/question/train/questions.txt").write_text(
                "Вопрос: A?\nОтвет: B.\n", encoding="utf-8"
            )
            (root / "data/question/validation/questions.txt").write_text(
                "Вопрос: C?\nОтвет: D.\n", encoding="utf-8"
            )
            config = TransformerConfig(
                context_length=4, d_model=4, n_layers=1, n_heads=1, d_mlp=8,
                batch_size=1, steps=1, learning_rate=0.01, weight_decay=0.0,
                warmup_steps=0, validation_interval=1, checkpoint_interval=1,
                text_ratio=0.5,
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(config.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
            result = train_from_config(config_path)
            restored = load_model(result.weights_dir)
            self.assertEqual(result.step, 1)
            self.assertTrue((result.weights_dir / "model.safetensors").is_file())
            self.assertTrue(result.checkpoint_path.is_file())
            self.assertEqual(restored.config, config)


if __name__ == "__main__":
    unittest.main()
