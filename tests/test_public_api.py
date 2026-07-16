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
        self.assertEqual(mimillm.__version__, "0.9.0")
        for name in (
            "Tensor", "AdamW", "TokenDataset", "create_model", "load_model",
            "save_model", "train_from_config", "CudaBackend", "cuda_is_available",
            "UnicodeByteTokenizer", "BpeTokenizer", "create_tokenizer",
            "train_bpe_tokenizer", "train_tokenizer_from_config",
            "load_tokenizer", "save_tokenizer", "tokenize", "detokenize",
            "pretokenize", "TokenizerReport", "analyze_tokenizer",
            "DatasetAuditReport", "audit_dataset", "PipelineResult",
            "train_pipeline", "DialogueEvaluationReport", "evaluate_dialogues",
            "save_dialogue_evaluation",
            "HailoRuntimeInfo", "HailoHefInfo", "hailo_is_available",
            "inspect_hailo_runtime", "inspect_hailo_hef",
        ):
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

    def test_standard_model_directory_round_trip(self) -> None:
        config = mimillm.ModelConfig(
            context_length=4, d_model=4, n_layers=1, n_heads=1, d_mlp=8,
            batch_size=1, steps=1, validation_interval=1, checkpoint_interval=1,
        )
        model = mimillm.create_model(config)
        expected = model([[257, 1, 2, 3]]).detach()
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "weights"
            saved = mimillm.save_model(model_dir, model)
            self.assertEqual(saved, model_dir)
            self.assertTrue((model_dir / "config.json").is_file())
            self.assertTrue((model_dir / "model.safetensors").is_file())
            restored = mimillm.load_model(model_dir)
            restored_from_file = mimillm.load_model(model_dir / "model.safetensors")
        self.assertEqual(restored([[257, 1, 2, 3]]).data, expected.data)
        self.assertEqual(restored_from_file.config, config)

    def test_bpe_model_directory_round_trip_saves_tokenizer_json(self) -> None:
        tokenizer = mimillm.train_bpe_tokenizer(
            ["hello hello hello", "привет привет"], vocab_size=280, min_frequency=1,
        )
        config = mimillm.ModelConfig(
            tokenizer="bpe", vocab_size=tokenizer.VOCAB_SIZE,
            context_length=8, d_model=4, n_layers=1, n_heads=1, d_mlp=8,
            batch_size=1, steps=1, validation_interval=1, checkpoint_interval=1,
        )
        model = mimillm.create_model(config, tokenizer_model=tokenizer)
        inputs = [tokenizer.encode("hello", add_bos=True)[:8]]
        expected = model(inputs).detach()
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "weights"
            mimillm.save_model(model_dir, model)
            self.assertTrue((model_dir / "tokenizer.json").is_file())
            restored = mimillm.load_model(model_dir)
            restored_from_file = mimillm.load_model(model_dir / "model.safetensors")
        self.assertIsInstance(restored.tokenizer, mimillm.BpeTokenizer)
        self.assertEqual(restored.tokenizer.to_dict(), tokenizer.to_dict())
        self.assertEqual(restored(inputs).data, expected.data)
        self.assertEqual(restored_from_file.tokenizer.VOCAB_SIZE, tokenizer.VOCAB_SIZE)
