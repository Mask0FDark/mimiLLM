"""Тесты byte-level, Unicode and BPE tokenizers."""

import tempfile
import unittest
from pathlib import Path

from mimillm.tokenizer import (
    BpeTokenizer,
    ByteTokenizer,
    UnicodeByteTokenizer,
    analyze_tokenizer,
    create_tokenizer,
    detokenize,
    format_qa_text,
    load_tokenizer,
    pretokenize,
    save_tokenizer,
    save_tokenizer_report,
    tokenize,
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

    def test_training_and_inference_share_one_prompt_formatter(self) -> None:
        question = "  Сколько дней в неделе?  "
        answer = "  В неделе семь дней.  "
        prompt = format_qa_text(question)
        completed = format_qa_text(question, answer)
        self.assertEqual(prompt, "Вопрос: Сколько дней в неделе?\nОтвет:")
        self.assertEqual(completed, prompt + " В неделе семь дней.")
        self.assertEqual(
            self.tokenizer.encode_qa(question, answer)[
                :len(self.tokenizer.encode_prompt(question))
            ],
            self.tokenizer.encode_prompt(question),
        )

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
    def test_unicode_pretokenizer_is_lossless_and_attaches_spaces(self) -> None:
        text = "Hello, мир! 42\nNext"
        chunks = pretokenize(text)
        self.assertEqual("".join(chunks), text)
        self.assertEqual(chunks, ["Hello", ",", " мир", "!", " 42", "\n", "Next"])

    def test_trained_bpe_is_reversible_and_shorter_than_bytes(self) -> None:
        text = "Привет, модель. Привет, модель. Hello model."
        tokenizer = train_bpe_tokenizer([text], vocab_size=300, min_frequency=1)
        encoded = tokenizer.encode(text)
        self.assertIsInstance(tokenizer, BpeTokenizer)
        self.assertEqual(tokenizer.decode(encoded), text)
        self.assertLess(len(encoded), len(ByteTokenizer().encode(text)))
        self.assertLessEqual(tokenizer.VOCAB_SIZE, 300)
        self.assertGreater(tokenizer.VOCAB_SIZE, ByteTokenizer.VOCAB_SIZE)
        self.assertEqual(tokenizer.format_version, 3)
        self.assertEqual(tokenizer.pretokenizer, "unicode_words_v1")
        self.assertTrue(tokenizer.unicode_character_merges)
        self.assertTrue(all(len(tokenizer.encode(character)) == 1 for character in "Привет"))

    def test_bpe_learns_leading_space_pieces(self) -> None:
        tokenizer = train_bpe_tokenizer(
            ["alpha beta alpha beta alpha beta"], vocab_size=290, min_frequency=1,
        )
        self.assertLess(
            len(tokenizer.encode(" beta")), len(ByteTokenizer().encode(" beta")),
        )

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

    def test_bpe_reserves_required_low_frequency_piece(self) -> None:
        tokenizer = train_bpe_tokenizer(
            ["обычный русский текст", "m0fdii"],
            vocab_size=300,
            min_frequency=100,
            required_pieces=["m0fdii"],
        )
        self.assertEqual(len(tokenizer.encode("m0fdii")), 1)
        self.assertEqual(tokenizer.required_pieces, ("m0fdii",))
        with tempfile.TemporaryDirectory() as directory:
            path = save_tokenizer(tokenizer, Path(directory) / "tokenizer.json")
            restored = load_tokenizer(path)
        self.assertEqual(restored.required_pieces, ("m0fdii",))
        self.assertEqual(restored.decode(restored.encode("m0fdii")), "m0fdii")

    def test_bpe_rejects_invalid_or_unreservable_required_piece(self) -> None:
        with self.assertRaisesRegex(ValueError, "whitespace"):
            train_bpe_tokenizer(
                ["small corpus"], required_pieces=["two words"],
                ensure_unicode_characters=False,
            )
        with self.assertRaisesRegex(ValueError, "too small"):
            train_bpe_tokenizer(
                ["small corpus"], vocab_size=264, min_frequency=100,
                required_pieces=["m0fdii"], ensure_unicode_characters=False,
            )
        with self.assertRaisesRegex(ValueError, "version 3"):
            BpeTokenizer(
                [(109, 48)], format_version=2, required_pieces=["m0"],
            )
        legacy_values = BpeTokenizer([(109, 48)], format_version=2).to_dict()
        legacy_values["required_pieces"] = ["m0"]
        with self.assertRaisesRegex(ValueError, "version 3"):
            BpeTokenizer.from_dict(legacy_values)

    def test_version_one_tokenizer_remains_compatible(self) -> None:
        legacy = BpeTokenizer(
            [(104, 101), (260, 108)],
            pretokenizer="legacy_whitespace",
            format_version=1,
        )
        values = legacy.to_dict()
        self.assertEqual(values["version"], 1)
        self.assertNotIn("pretokenizer", values)
        restored = BpeTokenizer.from_dict(values)
        text = "hello hello"
        self.assertEqual(restored.encode(text), legacy.encode(text))
        self.assertEqual(restored.decode(restored.encode(text)), text)

    def test_version_two_tokenizer_remains_compatible(self) -> None:
        previous = BpeTokenizer(
            [(208, 191)], pretokenizer="unicode_words_v1", format_version=2,
        )
        values = previous.to_dict()
        self.assertEqual(values["version"], 2)
        self.assertNotIn("unicode_character_merges", values)
        restored = BpeTokenizer.from_dict(values)
        self.assertEqual(restored.to_dict(), values)

    def test_quality_report_measures_unicode_and_writes_json(self) -> None:
        text = "Привет, модель. Привет, модель."
        tokenizer = train_bpe_tokenizer([text], vocab_size=300, min_frequency=1)
        report = analyze_tokenizer(tokenizer, [text])
        self.assertEqual(report.roundtrip_errors, 0)
        self.assertEqual(report.unicode_atomic_coverage, 1.0)
        self.assertLess(report.compression_ratio, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = save_tokenizer_report(
                report, Path(directory) / "tokenizer_report.json",
            )
            self.assertTrue(path.is_file())

    def test_bpe_requires_tokenizer_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "tokenizer.json"):
            create_tokenizer("bpe")

    def test_bpe_rejects_invalid_merges(self) -> None:
        with self.assertRaisesRegex(ValueError, "special"):
            BpeTokenizer([(ByteTokenizer.BOS, 1)])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            BpeTokenizer([(999, 1)])


class ConvenienceTokenizerTests(unittest.TestCase):
    def test_named_tokenizer_round_trip(self) -> None:
        text = "Привет, mimiLLM!"
        tokens = tokenize(text, "unicode", add_bos=True, add_eos=True)
        self.assertEqual(tokens[0], ByteTokenizer.BOS)
        self.assertEqual(tokens[-1], ByteTokenizer.EOS)
        self.assertEqual(detokenize(tokens, "unicode"), text)

    def test_bpe_artifact_path_round_trip(self) -> None:
        tokenizer = train_bpe_tokenizer(
            ["tokenization tokenization"], vocab_size=280, min_frequency=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_tokenizer(tokenizer, Path(directory) / "tokenizer.json")
            tokens = tokenize("tokenization", path)
            restored = detokenize(tokens, path)
        self.assertEqual(restored, "tokenization")


if __name__ == "__main__":
    unittest.main()
