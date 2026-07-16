"""Dataset audit regression tests."""

import tempfile
import unittest
from pathlib import Path

from mimillm.audit import audit_dataset, normalize_training_text, save_dataset_audit


def _write_qa(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n\n".join(
            f"Вопрос: {question}\nОтвет: {answer}" for question, answer in pairs
        ),
        encoding="utf-8",
    )


class DatasetAuditTests(unittest.TestCase):
    def test_normalization_is_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(
            normalize_training_text("  Кто\nТЫ? "),
            normalize_training_text("кто ты?"),
        )

    def test_audit_detects_leakage_conflicts_and_foreign_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_qa(
                root / "train.txt",
                [
                    ("Кто ты?", "Я LocalModel."),
                    ("кто   ты?", "Я Open Assistant."),
                    ("Кто ты?", "Я LocalModel."),
                ],
            )
            _write_qa(root / "validation.txt", [("КТО ТЫ?", "Я LocalModel.")])
            report = audit_dataset(
                question_train_path=root / "train.txt",
                question_validation_path=root / "validation.txt",
                forbidden_phrases=["Open Assistant"],
            )
            self.assertFalse(report.ok)
            self.assertEqual(report.duplicate_train_qa_pairs, 1)
            self.assertEqual(report.conflicting_train_questions, 1)
            self.assertEqual(report.train_validation_question_overlap, 1)
            self.assertEqual(report.forbidden_phrase_hits, 1)
            saved = save_dataset_audit(report, root / "audit.json")
            self.assertTrue(saved.is_file())

    def test_clean_disjoint_data_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_qa(root / "train.txt", [("Кто ты?", "Я модель.")])
            _write_qa(root / "validation.txt", [("Как тебя зовут?", "mimiLLM.")])
            report = audit_dataset(
                question_train_path=root / "train.txt",
                question_validation_path=root / "validation.txt",
            )
            self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
