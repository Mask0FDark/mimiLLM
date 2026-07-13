"""Проверки выбора token и EOS-остановки генератора."""

import random
import unittest

from mimillm.generation import (
    generate, generate_response, generate_text, sample_token,
)
from mimillm.tensor import Tensor
from mimillm.tokenizer import ByteTokenizer


class _Config:
    context_length = 4
    vocab_size = 260


class _EosModel:
    config = _Config()

    def __call__(self, tokens):
        time = len(tokens[0])
        values = [-10.0] * (time * 260)
        values[(time - 1) * 260 + ByteTokenizer.EOS] = 10.0
        return Tensor(values, (1, time, 260))


class GenerationTests(unittest.TestCase):
    def test_greedy(self) -> None:
        self.assertEqual(
            sample_token([0.1, 2.0, 1.0], temperature=0.0, top_k=0, rng=random.Random(1)),
            1,
        )

    def test_top_k_one_is_greedy(self) -> None:
        self.assertEqual(
            sample_token([3.0, 1.0, 2.0], temperature=5.0, top_k=1, rng=random.Random(9)),
            0,
        )

    def test_eos_stops_without_returning_eos(self) -> None:
        generated = generate(
            _EosModel(), [ByteTokenizer.BOS, 1], max_new_tokens=10,
            temperature=0.0, top_k=0,
        )
        self.assertEqual(generated, [])

    def test_generate_text_returns_a_string(self) -> None:
        self.assertEqual(
            generate_text(_EosModel(), "hello", max_new_tokens=2, temperature=0.0, top_k=1),
            "",
        )
        self.assertEqual(
            generate_text(
                _EosModel(), "hello", include_prompt=True,
                max_new_tokens=2, temperature=0.0, top_k=1,
            ),
            "hello",
        )

    def test_response_always_uses_the_model_request_prompt(self) -> None:
        settings = {"max_new_tokens": 2, "temperature": 0.0, "top_k": 1}
        self.assertEqual(generate_response(_EosModel(), "кто ты", **settings), "")
        self.assertEqual(generate_response(_EosModel(), "напиши рассказ", **settings), "")


if __name__ == "__main__":
    unittest.main()
