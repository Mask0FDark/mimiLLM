"""Tests for held-out generation and dialogue-memory quality gates."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mimillm.evaluation import evaluate_dialogues, save_dialogue_evaluation


class DialogueEvaluationTests(unittest.TestCase):
    def _suite(self, root: Path, *, exact: str = "Тебя зовут Ира.") -> Path:
        path = root / "dialogue_eval.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "min_pass_rate": 1.0,
                    "generation": {
                        "max_new_tokens": 32,
                        "temperature": 0,
                        "top_k": 1,
                    },
                    "cases": [
                        {
                            "name": "remember-name",
                            "messages": [
                                {"role": "user", "content": "Меня зовут Ира."},
                                {
                                    "role": "assistant",
                                    "contains_any": ["приятно", "Ира"],
                                    "forbidden": ["не знаю"],
                                },
                                {"role": "user", "content": "Как меня зовут?"},
                                {"role": "assistant", "exact": exact},
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_uses_generated_answer_as_multi_turn_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts: list[str] = []

            def answer(_model, _tokenizer, prompt, **_settings):
                prompts.append(prompt)
                return ("Приятно познакомиться, Ира." if len(prompts) == 1
                        else "Тебя зовут Ира.")

            model = type("Model", (), {"tokenizer": object()})()
            with patch("mimillm.evaluation.answer_question", side_effect=answer):
                report = evaluate_dialogues(model, self._suite(root))
            self.assertTrue(report.ok)
            self.assertEqual(report.pass_rate, 1.0)
            self.assertEqual(len(prompts), 2)
            self.assertIn("Ответ: Приятно познакомиться, Ира.", prompts[1])
            self.assertTrue(prompts[1].endswith("Вопрос: Как меня зовут?"))

    def test_failure_report_records_real_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = type("Model", (), {"tokenizer": object()})()
            with patch(
                "mimillm.evaluation.answer_question",
                side_effect=["Не знаю.", "Тебя зовут Олег."],
            ):
                report = evaluate_dialogues(model, self._suite(root))
            self.assertFalse(report.ok)
            self.assertEqual(report.passed, 0)
            self.assertGreaterEqual(len(report.results[0].failures), 2)
            destination = save_dialogue_evaluation(report, root / "report.json")
            stored = json.loads(destination.read_text(encoding="utf-8"))
            self.assertFalse(stored["ok"])
            self.assertEqual(stored["results"][0]["responses"][1], "Тебя зовут Олег.")

    def test_rejects_invalid_generation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._suite(root)
            values = json.loads(path.read_text(encoding="utf-8"))
            values["generation"]["top_k"] = True
            path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top_k"):
                evaluate_dialogues(type("Model", (), {})(), path)

    def test_response_shape_checks_reject_degenerate_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "shape_eval.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "min_pass_rate": 1.0,
                        "cases": [{
                            "name": "natural-answer",
                            "messages": [
                                {"role": "user", "content": "Привет!"},
                                {
                                    "role": "assistant",
                                    "min_characters": 12,
                                    "min_cyrillic_characters": 12,
                                    "max_repeated_word_fraction": 0.5,
                                },
                            ],
                        }],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            model = type("Model", (), {"tokenizer": object()})()
            with patch(
                "mimillm.evaluation.answer_question",
                return_value="Ответ: Ответ: ------------------------",
            ):
                report = evaluate_dialogues(model, path)
            self.assertFalse(report.ok)
            self.assertIn("Cyrillic", " ".join(report.results[0].failures))
            self.assertIn("most frequent word", " ".join(report.results[0].failures))

    def test_response_shape_checks_accept_normal_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "shape_eval.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cases": [{
                            "name": "natural-answer",
                            "messages": [
                                {"role": "user", "content": "Привет!"},
                                {
                                    "role": "assistant",
                                    "min_characters": 12,
                                    "min_cyrillic_characters": 12,
                                    "max_repeated_word_fraction": 0.5,
                                },
                            ],
                        }],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            model = type("Model", (), {"tokenizer": object()})()
            with patch(
                "mimillm.evaluation.answer_question",
                return_value="Привет! Рад с тобой познакомиться.",
            ):
                report = evaluate_dialogues(model, path)
            self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
