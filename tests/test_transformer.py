"""Проверки attention, causal mask и полной decoder-модели."""

import os
import random
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MIMILLM_BACKEND", "python")

from mimillm.attention import MultiHeadCausalSelfAttention
from mimillm.layers import Embedding, RMSNorm
from mimillm.tensor import Tensor
from mimillm.transformer import DecoderTransformer, TransformerBlock, TransformerConfig
from mimillm.tokenizer import UnicodeByteTokenizer


class TransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TransformerConfig(
            context_length=8, d_model=8, n_layers=1, n_heads=2, d_mlp=16,
            batch_size=1, steps=2, validation_interval=1, checkpoint_interval=1,
        )

    def test_embedding_shape_and_repeated_gradient(self) -> None:
        layer = Embedding(10, 4, rng=random.Random(1))
        output = layer([2, 2, 3])
        self.assertEqual(output.shape, (3, 4))
        output.sum().backward()
        self.assertEqual(list(layer.weight.grad[8:12]), [2.0] * 4)  # type: ignore[index]

    def test_rmsnorm_shape_and_scale(self) -> None:
        layer = RMSNorm(4)
        output = layer(Tensor([1, -1, 1, -1], (1, 4)))
        self.assertEqual(output.shape, (1, 4))
        self.assertAlmostEqual(sum(value * value for value in output.data) / 4, 1.0, places=4)

    def test_attention_and_block_shape(self) -> None:
        inputs = Tensor.randn((1, 4, 8), rng=random.Random(2), requires_grad=True)
        attention = MultiHeadCausalSelfAttention(8, 2, rng=random.Random(3))
        self.assertEqual(attention(inputs).shape, inputs.shape)
        self.assertEqual(TransformerBlock(self.config, rng=random.Random(4))(inputs).shape, inputs.shape)

    def test_model_shape_and_parameter_count(self) -> None:
        model = DecoderTransformer(self.config)
        logits = model([[257, 1, 2, 3]])
        self.assertEqual(logits.shape, (1, 4, 260))
        self.assertGreater(model.parameter_count(), 2000)

    def test_tied_word_embeddings_reuse_the_input_matrix(self) -> None:
        tied_config = replace(self.config, tie_word_embeddings=True)
        untied_config = replace(self.config, tie_word_embeddings=False)
        tied = DecoderTransformer(tied_config)
        untied = DecoderTransformer(untied_config)

        self.assertIsNone(tied.output)
        self.assertEqual(
            tied.parameter_count(),
            untied.parameter_count() - self.config.vocab_size * self.config.d_model,
        )
        logits = tied([[257, 1, 2, 3]])
        self.assertEqual(logits.shape, (1, 4, self.config.vocab_size))
        logits.sum().backward()
        self.assertIsNotNone(tied.token_embedding.weight.grad)

    def test_causal_mask_blocks_future_tokens(self) -> None:
        model = DecoderTransformer(self.config)
        first = model([[257, 10, 11, 12]]).detach()
        second = model([[257, 10, 99, 100]]).detach()
        vocab = self.config.vocab_size
        for position in (0, 1):
            start = position * vocab
            for left, right in zip(first.data[start:start + vocab], second.data[start:start + vocab]):
                self.assertAlmostEqual(left, right, places=5)

    def test_config_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "делиться"):
            TransformerConfig(d_model=10, n_heads=3)
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            TransformerConfig(vocab_size=100)
        with self.assertRaisesRegex(ValueError, "text_ratio"):
            TransformerConfig(text_ratio=1.1)
        with self.assertRaisesRegex(ValueError, "learning_rate_schedule"):
            TransformerConfig(learning_rate_schedule="magic")
        with self.assertRaisesRegex(ValueError, "gradient_clip_norm"):
            TransformerConfig(gradient_clip_norm=0.0)
        with self.assertRaisesRegex(ValueError, "qa_source_weights"):
            TransformerConfig(qa_source_weights={"qa.txt": -1.0})

    def test_unicode_tokenizer_changes_vocabulary_and_model_shape(self) -> None:
        config = TransformerConfig(
            tokenizer="unicode", vocab_size=UnicodeByteTokenizer.VOCAB_SIZE,
            context_length=8, d_model=8, n_layers=1, n_heads=2, d_mlp=16,
            batch_size=1, steps=2, validation_interval=1, checkpoint_interval=1,
        )
        model = DecoderTransformer(config)
        tokens = model.tokenizer.encode("Привет")
        self.assertEqual(
            model([tokens]).shape,
            (1, len(tokens), UnicodeByteTokenizer.VOCAB_SIZE),
        )

    def test_all_example_configs_are_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        configs = sorted((root / "configs").glob("*.json"))
        self.assertGreaterEqual(len(configs), 4)
        for path in configs:
            with self.subTest(config=path.name):
                TransformerConfig.from_json(path)


if __name__ == "__main__":
    unittest.main()
