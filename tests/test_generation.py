"""Проверки выбора token и EOS-остановки генератора."""

import random
import unittest

from minillm.generation import generate, sample_token
from minillm.tensor import Tensor
from minillm.tokenizer import ByteTokenizer


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


if __name__ == "__main__":
    unittest.main()
