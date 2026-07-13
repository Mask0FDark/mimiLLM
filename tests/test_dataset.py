"""Проверки формата и воспроизводимости локального датасета."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from mimillm.dataset import (
    TokenDataset, discover_question_files, discover_text_files,
    load_qa_text, load_text_documents,
)
from tools.make_dataset import build_dataset


class DatasetTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self) -> None:
        records = [
            {"question": f"Вопрос {index}?", "answer": f"Ответ {index}.",
             "variants": [f"Вариант {index}?"]}
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "seed.jsonl"
            source.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in records),
                encoding="utf-8",
            )
            first_train, first_validation = root / "train1.txt", root / "validation1.txt"
            second_train, second_validation = root / "train2.txt", root / "validation2.txt"
            build_dataset(source, first_train, first_validation, 123, 0.2)
            build_dataset(source, second_train, second_validation, 123, 0.2)
            self.assertEqual(first_train.read_bytes(), second_train.read_bytes())
            self.assertEqual(first_validation.read_bytes(), second_validation.read_bytes())
            train_questions = {question for question, _ in load_qa_text(first_train)}
            validation_questions = {question for question, _ in load_qa_text(first_validation)}
            self.assertTrue(train_questions.isdisjoint(validation_questions))

    def test_token_batch_is_shifted_by_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            path.write_text("Вопрос: А?\nОтвет: Б.\n", encoding="utf-8")
            dataset = TokenDataset(path)
            inputs, targets = dataset.sample_batch(1, 4, random.Random(1))
            self.assertEqual(inputs[0][1:], targets[0][:-1])

    def test_question_directory_is_loaded_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            first = root / "a.txt"
            second = nested / "b.txt"
            first.write_text("Вопрос: A?\nОтвет: B.\n", encoding="utf-8")
            second.write_text("Вопрос: C?\nОтвет: D.\n", encoding="utf-8")
            self.assertEqual(discover_question_files(root), [first, second])
            self.assertEqual(len(load_qa_text(root)), 2)
            self.assertEqual(len(TokenDataset(root).examples), 2)

    def test_short_example_uses_dynamic_context_without_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            path.write_text("Вопрос: А?\nОтвет: Б.\n", encoding="utf-8")
            dataset = TokenDataset(path)
            inputs, targets = dataset.sample_batch(1, 1000, random.Random(2))
            self.assertEqual(len(inputs[0]), len(dataset.sequences[0]) - 1)
            self.assertNotIn(dataset.tokenizer.PAD, inputs[0])

    def test_training_batch_masks_prompt_and_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            path.write_text(
                "Вопрос: A?\nОтвет: B.\n\n"
                "Вопрос: A much longer question?\nОтвет: A longer answer.\n",
                encoding="utf-8",
            )
            dataset = TokenDataset(path)
            inputs, targets, weights = dataset.sample_batch_with_loss_weights(
                2, 1000, random.Random(3)
            )
            self.assertEqual(len(inputs[0]), len(inputs[1]))
            self.assertEqual(len(targets[0]), len(weights[0]))
            self.assertIn(0.0, weights[0])
            self.assertIn(1.0, weights[0])
            for row_targets, row_weights in zip(targets, weights):
                for target, weight in zip(row_targets, row_weights):
                    if target == dataset.tokenizer.PAD:
                        self.assertEqual(weight, 0.0)

    def test_raw_text_batch_uses_language_model_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qa = root / "qa.txt"
            qa.write_text("Вопрос: А?\nОтвет: Б.\n", encoding="utf-8")
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "ru.txt").write_text("Язык хранит смысл в последовательности слов.", encoding="utf-8")
            dataset = TokenDataset(qa, text_paths=corpus, text_ratio=1.0)
            inputs, targets = dataset.sample_batch(2, 12, random.Random(3))
            self.assertEqual(dataset.last_source, "text")
            self.assertEqual(inputs[0][1:], targets[0][:-1])
            self.assertEqual(dataset.source_weights(), [("text", 1.0)])
            self.assertGreater(dataset.text_tokens, 0)

    def test_text_only_dataset_does_not_require_qa_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory) / "corpus.txt"
            text.write_text("Text-only training is a supported library workflow.", encoding="utf-8")
            dataset = TokenDataset(text_paths=text, text_ratio=1.0)
            inputs, targets = dataset.sample_batch(1, 16, random.Random(5))
            self.assertEqual(dataset.examples, [])
            self.assertEqual(dataset.source_weights(), [("text", 1.0)])
            self.assertEqual(inputs[0][1:], targets[0][:-1])

    def test_mixed_sampler_reaches_both_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qa = root / "qa.txt"
            text = root / "language.md"
            qa.write_text("Вопрос: А?\nОтвет: Б.\n", encoding="utf-8")
            text.write_text("An ordinary document teaches word order.", encoding="utf-8")
            dataset = TokenDataset(qa, text_paths=text, text_ratio=0.5)
            rng = random.Random(4)
            sources = set()
            for _ in range(20):
                dataset.sample_batch(1, 8, rng)
                sources.add(dataset.last_source)
            self.assertEqual(sources, {"qa", "text"})
            self.assertEqual(dataset.source_weights(), [("qa", 0.5), ("text", 0.5)])

    def test_text_discovery_is_recursive_and_rejects_empty_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            first = root / "a.txt"
            second = nested / "b.md"
            first.write_text("первый", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            (root / "ignored.json").write_text("{}", encoding="utf-8")
            self.assertEqual(discover_text_files(root), [first, second])
            self.assertEqual(len(load_text_documents(root)), 2)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "нет файлов"):
                load_text_documents(empty)


if __name__ == "__main__":
    unittest.main()
