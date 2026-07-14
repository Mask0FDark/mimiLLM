"""Тесты byte-level токенизатора."""

import unittest

from mimillm.tokenizer import ByteTokenizer, UnicodeByteTokenizer, create_tokenizer


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


if __name__ == "__main__":
    unittest.main()
