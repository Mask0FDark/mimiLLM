"""Тесты byte-level, Unicode and BPE tokenizers."""

import tempfile
import unittest
from pathlib import Path

from mimillm.tokenizer import (
    BpeTokenizer,
    ByteTokenizer,
    UnicodeByteTokenizer,
    create_tokenizer,
    load_tokenizer,
    save_tokenizer,
    train_bpe_tokenizer,
)


class ByteTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ByteTokenizer()

    def test_russian_roundtrip(self) -> None:
        text = "Привет, мир! Ёжик."
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_english_digits_and_punctuation(self) -> None:
        text = "Hello, model 2026!"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_empty_string(self) -> None:
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_special_tokens_and_qa_format(self) -> None:
        tokens = self.tokenizer.encode_qa("Как дела?", "Хорошо.")
        self.assertEqual(tokens[0], ByteTokenizer.BOS)
        self.assertEqual(tokens[-1], ByteTokenizer.EOS)
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, "Вопрос: Как дела?\nОтвет: Хорошо.")

    def test_prompt_has_no_eos(self) -> None:
        tokens = self.tokenizer.encode_prompt("Кто ты?")
        self.assertEqual(tokens[0], ByteTokenizer.BOS)
        self.assertNotIn(ByteTokenizer.EOS, tokens)
        self.assertTrue(self.tokenizer.decode(tokens).endswith("Ответ:"))

    def test_invalid_utf8_is_safe(self) -> None:
        self.assertEqual(self.tokenizer.decode([0xD0]), "\ufffd")

    def test_invalid_token_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "вне диапазона"):
            self.tokenizer.decode([260])


class UnicodeByteTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = UnicodeByteTokenizer()

    def test_russian_uses_one_token_per_common_character(self) -> None:
        text = "Привет, мир!"
        encoded = self.tokenizer.encode(text)
        self.assertEqual(self.tokenizer.decode(encoded), text)
        self.assertLess(len(encoded), len(text.encode("utf-8")))

    def test_unknown_unicode_falls_back_to_reversible_utf8(self) -> None:
        text = "English 漢字 🚀"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_qa_format_and_factory(self) -> None:
        tokenizer = create_tokenizer("unicode")
        tokens = tokenizer.encode_qa("Кто ты?", "Я модель.")
        self.assertIsInstance(tokenizer, UnicodeByteTokenizer)
        self.assertEqual(
            tokenizer.decode(tokens), "Вопрос: Кто ты?\nОтвет: Я модель.",
        )
        self.assertEqual(create_tokenizer("byte").VOCAB_SIZE, 260)


class BpeTokenizerTests(unittest.TestCase):
    def test_trained_bpe_is_reversible_and_shorter_than_bytes(self) -> None:
        text = "Привет, модель. Привет, модель. Hello model."
        tokenizer = train_bpe_tokenizer([text], vocab_size=300, min_frequency=1)
        encoded = tokenizer.encode(text)
        self.assertIsInstance(tokenizer, BpeTokenizer)
        self.assertEqual(tokenizer.decode(encoded), text)
        self.assertLess(len(encoded), len(ByteTokenizer().encode(text)))
        self.assertLessEqual(tokenizer.VOCAB_SIZE, 300)
        self.assertGreater(tokenizer.VOCAB_SIZE, ByteTokenizer.VOCAB_SIZE)

    def test_bpe_save_load_and_factory(self) -> None:
        tokenizer = train_bpe_tokenizer(
            ["memory memory memory", "память память"], vocab_size=290, min_frequency=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            saved = save_tokenizer(tokenizer, path)
            restored = load_tokenizer(saved)
            restored_from_factory = create_tokenizer("bpe", path=saved)
        text = "memory память 🚀"
        self.assertEqual(restored.decode(restored.encode(text)), text)
        self.assertEqual(restored_from_factory.decode(restored_from_factory.encode(text)), text)
        self.assertEqual(restored.VOCAB_SIZE, tokenizer.VOCAB_SIZE)
        self.assertEqual(restored.to_dict(), tokenizer.to_dict())

    def test_bpe_requires_tokenizer_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "tokenizer.json"):
            create_tokenizer("bpe")

    def test_bpe_rejects_invalid_merges(self) -> None:
        with self.assertRaisesRegex(ValueError, "special"):
            BpeTokenizer([(ByteTokenizer.BOS, 1)])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            BpeTokenizer([(999, 1)])


if __name__ == "__main__":
    unittest.main()
