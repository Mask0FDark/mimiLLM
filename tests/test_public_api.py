"""Публичный импорт библиотеки должен покрывать типичный пользовательский путь."""

import unittest
import tempfile
from pathlib import Path

import mimillm


class PublicApiTests(unittest.TestCase):
    def test_create_model_from_named_options(self) -> None:
        model = mimillm.create_model(
            context_length=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_mlp=16,
            batch_size=1,
            steps=1,
            validation_interval=1,
            checkpoint_interval=1,
        )
        logits = model([[mimillm.ByteTokenizer.BOS, 1, 2]])
        self.assertEqual(logits.shape, (1, 3, mimillm.ByteTokenizer.VOCAB_SIZE))
        self.assertIs(mimillm.LanguageModel, mimillm.DecoderTransformer)
        self.assertIs(mimillm.ModelConfig, mimillm.TransformerConfig)

    def test_create_model_rejects_ambiguous_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "не оба"):
            mimillm.create_model(mimillm.ModelConfig(), d_model=8)

    def test_version_and_exports(self) -> None:
        self.assertEqual(mimillm.__version__, "0.2.0")
        for name in ("Tensor", "AdamW", "TokenDataset", "create_model", "load_model"):
            self.assertIn(name, mimillm.__all__)

    def test_load_model_restores_configuration_and_weights(self) -> None:
        config = mimillm.ModelConfig(
            context_length=4, d_model=4, n_layers=1, n_heads=1, d_mlp=8,
            batch_size=1, steps=1, validation_interval=1, checkpoint_interval=1,
        )
        model = mimillm.create_model(config)
        expected = model([[257, 1, 2, 3]]).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            mimillm.save_checkpoint(
                path, model, None, config=config.to_dict(), step=0, seed=config.seed
            )
            restored = mimillm.load_model(path)
            actual = restored([[257, 1, 2, 3]]).detach()
        self.assertEqual(restored.config, config)
        self.assertEqual(actual.data, expected.data)
