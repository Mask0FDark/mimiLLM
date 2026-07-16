"""Safe staged-training pipeline tests."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mimillm.evaluation import DialogueEvaluationReport
from mimillm.pipeline import train_pipeline


def _json(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")


def _config(*, text_ratio: float) -> dict[str, object]:
    return {
        "vocab_size": 260,
        "tokenizer": "byte",
        "tie_word_embeddings": True,
        "context_length": 16,
        "d_model": 4,
        "n_layers": 1,
        "n_heads": 1,
        "d_mlp": 8,
        "batch_size": 1,
        "steps": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "validation_interval": 1,
        "checkpoint_interval": 1,
        "seed": 7,
        "text_ratio": text_ratio,
        "qa_prompt_weight": 0.0,
        "text_train_path": "data/text/train",
        "text_validation_path": "data/text/validation",
        "question_train_path": "data/question/train",
        "question_validation_path": "data/question/validation",
    }


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_backend = os.environ.get("MIMILLM_BACKEND")
        os.environ["MIMILLM_BACKEND"] = "python"

    def tearDown(self) -> None:
        if self.previous_backend is None:
            os.environ.pop("MIMILLM_BACKEND", None)
        else:
            os.environ["MIMILLM_BACKEND"] = self.previous_backend

    def test_pipeline_trains_tokenizer_and_connects_pretrain_to_sft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, text in (
                ("train", "Привет, мир. Модель сначала изучает русский язык."),
                ("validation", "Новый проверочный текст не входит в обучение."),
            ):
                path = root / "pretrain" / "data" / "text" / split / "text.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            for split, question, answer in (
                ("train", "Кто ты?", "Я небольшая языковая модель."),
                ("validation", "Как тебя зовут?", "Меня зовут mimiLLM."),
            ):
                path = root / "sft" / "data" / "question" / split / "qa.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"Вопрос: {question}\nОтвет: {answer}", encoding="utf-8",
                )
            _json(root / "model.json", _config(text_ratio=0.5))
            _json(root / "pretrain" / "config.json", {"text_ratio": 1.0})
            _json(root / "sft" / "config.json", {"text_ratio": 0.0})
            _json(
                root / "pipeline.json",
                {
                    "version": 1,
                    "base_config": "model.json",
                    "tokenizer": {
                        "type": "bpe",
                        "path": "artifacts/tokenizer.json",
                        "vocab_size": 300,
                        "min_frequency": 1,
                    },
                    "dataset_checks": {"forbidden_phrases": ["Open Assistant"]},
                    "stages": [
                        {
                            "name": "language",
                            "kind": "pretrain",
                            "config": "pretrain/config.json",
                            "output_dir": "weights/language",
                        },
                        {
                            "name": "dialogue",
                            "kind": "sft",
                            "config": "sft/config.json",
                            "output_dir": "weights/dialogue",
                        },
                    ],
                },
            )
            result = train_pipeline(root / "pipeline.json", backend="python")
            self.assertIsNone(result.interrupted_stage)
            self.assertEqual(
                result.final_weights, (root / "weights" / "dialogue").resolve(),
            )
            self.assertTrue((root / "artifacts" / "tokenizer.json").is_file())
            self.assertTrue((root / "tokenizer_report.json").is_file())
            self.assertTrue((root / "pipeline_audit.json").is_file())
            lineage = json.loads(
                (root / "weights" / "dialogue" / "lineage.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(lineage["status"], "complete")
            self.assertEqual(lineage["parent_stage"], "language")
            self.assertEqual(
                Path(lineage["initialized_from"]),
                (root / "weights" / "language").resolve(),
            )
            _json(
                root / "sft" / "config.json",
                {"text_ratio": 0.0, "steps": 2},
            )
            resumed = train_pipeline(
                root / "pipeline.json", backend="python", resume_stage="dialogue",
            )
            self.assertEqual(resumed.stages[-1].step, 2)
            resumed_lineage = json.loads(
                (root / "weights" / "dialogue" / "lineage.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(resumed_lineage["effective_config"]["steps"], 2)

    def test_sft_from_scratch_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _json(root / "sft.json", _config(text_ratio=0.0))
            _json(
                root / "pipeline.json",
                {
                    "version": 1,
                    "tokenizer": {"type": "byte"},
                    "stages": [
                        {
                            "name": "dialogue",
                            "kind": "sft",
                            "config": "sft.json",
                            "output_dir": "weights/dialogue",
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "first stage must be pretrain"):
                train_pipeline(root / "pipeline.json", backend="python")

    def test_failed_generation_gate_stops_pipeline_and_updates_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, text in (
                ("train", "A language training document."),
                ("validation", "A separate validation document."),
            ):
                source = root / "data" / "text" / split / "text.txt"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(text, encoding="utf-8")
            _json(root / "config.json", _config(text_ratio=1.0))
            _json(root / "evaluation.json", {})
            _json(
                root / "pipeline.json",
                {
                    "version": 1,
                    "tokenizer": {"type": "byte"},
                    "stages": [
                        {
                            "name": "language",
                            "kind": "pretrain",
                            "config": "config.json",
                            "output_dir": "weights/language",
                            "evaluation": "evaluation.json",
                        }
                    ],
                },
            )
            failed = DialogueEvaluationReport(
                cases=1, passed=0, pass_rate=0.0, min_pass_rate=1.0,
                ok=False, results=(),
            )
            with patch("mimillm.pipeline.evaluate_dialogues", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "quality gate"):
                    train_pipeline(root / "pipeline.json", backend="python")
            lineage = json.loads(
                (root / "weights" / "language" / "lineage.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(lineage["status"], "quality_failed")
            self.assertEqual(lineage["generation_evaluation"]["pass_rate"], 0.0)
            self.assertTrue(
                (root / "weights" / "language" / "generation_report.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
