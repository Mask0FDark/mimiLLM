"""Safe staged-training pipeline tests."""

import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mimillm.api import load_model, save_model
from mimillm.evaluation import DialogueEvaluationReport
from mimillm.pipeline import (
    PipelineQualityError,
    PipelineStage,
    _evaluate_generation_candidates,
    _tokenizer_settings,
    _validate_tokenizer_reports,
    train_pipeline,
)
from mimillm.tokenizer import BpeTokenizer, TokenizerReport
from mimillm.training import TrainingResult
from mimillm.transformer import DecoderTransformer, TransformerConfig


def _json(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    def test_required_pieces_are_only_valid_for_bpe(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires type='bpe'"):
            _tokenizer_settings({
                "tokenizer": {
                    "type": "byte",
                    "required_pieces": ["model-name"],
                },
            })

    def test_tokenizer_gate_rejects_fragmented_validation_words(self) -> None:
        report = TokenizerReport(
            documents=1,
            characters=20,
            words=2,
            utf8_bytes=30,
            tokens=14,
            unique_tokens=8,
            vocab_size=260,
            raw_byte_tokens=4,
            non_ascii_byte_tokens=0,
            roundtrip_errors=0,
            replacement_characters=0,
            compression_ratio=14 / 30,
            tokens_per_word=7.0,
            vocab_utilization=8 / 260,
            unicode_atomic_coverage=1.0,
            warnings=(),
        )
        pipeline = {
            "tokenizer": {
                "type": "bpe",
                "max_tokens_per_word": 4.0,
                "max_raw_byte_fraction": 0.5,
            },
        }
        with self.assertRaisesRegex(ValueError, "tokens/word"):
            _validate_tokenizer_reports(
                pipeline, BpeTokenizer([]), {"validation": report},
            )

    def test_tokenizer_gate_rejects_excessive_byte_fallback(self) -> None:
        report = TokenizerReport(
            documents=1,
            characters=20,
            words=5,
            utf8_bytes=30,
            tokens=15,
            unique_tokens=8,
            vocab_size=260,
            raw_byte_tokens=12,
            non_ascii_byte_tokens=0,
            roundtrip_errors=0,
            replacement_characters=0,
            compression_ratio=0.5,
            tokens_per_word=3.0,
            vocab_utilization=8 / 260,
            unicode_atomic_coverage=1.0,
            warnings=(),
        )
        pipeline = {
            "tokenizer": {
                "type": "bpe",
                "max_tokens_per_word": 4.0,
                "max_raw_byte_fraction": 0.5,
            },
        }
        with self.assertRaisesRegex(ValueError, "raw-byte fraction"):
            _validate_tokenizer_reports(
                pipeline, BpeTokenizer([]), {"validation": report},
            )

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
                        "vocab_size": 340,
                        "min_frequency": 1,
                        "required_pieces": ["mimiLLM"],
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
            tokenizer = load_model(result.final_weights).tokenizer
            self.assertEqual(len(tokenizer.encode("mimiLLM")), 1)
            self.assertTrue((root / "tokenizer_report.json").is_file())
            self.assertTrue((root / "pipeline_audit.json").is_file())
            lineage = json.loads(
                (root / "weights" / "dialogue" / "lineage.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(lineage["status"], "complete")
            self.assertEqual(lineage["parent_stage"], "language")
            language_lineage = json.loads(
                (root / "weights/language/lineage.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(
                lineage["tokenizer_sha256"], language_lineage["tokenizer_sha256"],
            )
            self.assertEqual(
                Path(lineage["initialized_from"]),
                (root / "weights" / "language" / "best").resolve(),
            )
            self.assertEqual(
                lineage["initialized_from_model_sha256"],
                _sha256(root / "weights/language/best/model.safetensors"),
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

            with self.assertRaisesRegex(FileExistsError, "archive or move"):
                train_pipeline(
                    root / "pipeline.json", backend="python",
                    restart_stage="dialogue",
                )
            dialogue_output = root / "weights" / "dialogue"
            dialogue_output.rename(root / "weights" / "dialogue_archived")
            restarted = train_pipeline(
                root / "pipeline.json", backend="python",
                restart_stage="dialogue",
            )
            self.assertEqual(restarted.stages[-1].step, 2)
            restarted_lineage = json.loads(
                (dialogue_output / "lineage.json").read_text(encoding="utf-8")
            )
            self.assertEqual(restarted_lineage["status"], "complete")
            self.assertEqual(restarted_lineage["parent_stage"], "language")

            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                train_pipeline(
                    root / "pipeline.json", backend="python",
                    resume_stage="dialogue", restart_stage="dialogue",
                )

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

    def test_pipeline_can_start_sft_from_declared_initial_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_values = _config(text_ratio=0.0)
            initial = root / "initial"
            save_model(
                initial,
                DecoderTransformer(TransformerConfig.from_dict(config_values)),
            )
            for split, question, answer in (
                ("train", "Who are you?", "A small model."),
                ("validation", "What are you?", "A language model."),
            ):
                path = root / "data" / "question" / split / "qa.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"Вопрос: {question}\nОтвет: {answer}\n", encoding="utf-8",
                )
            _json(root / "sft.json", config_values)
            _json(
                root / "pipeline.json",
                {
                    "version": 1,
                    "initial_weights": "initial",
                    "tokenizer": {"type": "byte"},
                    "stages": [
                        {
                            "name": "instruction",
                            "kind": "sft",
                            "config": "sft.json",
                            "output_dir": "weights/instruction",
                        }
                    ],
                },
            )
            result = train_pipeline(root / "pipeline.json", backend="python")
            self.assertEqual(result.final_weights, (root / "weights/instruction").resolve())
            lineage = json.loads(
                (root / "weights/instruction/lineage.json").read_text(encoding="utf-8")
            )
            self.assertEqual(lineage["parent_stage"], "initial_weights")
            self.assertEqual(Path(lineage["initialized_from"]), initial.resolve())
            self.assertEqual(
                lineage["initialized_from_model_sha256"],
                _sha256(initial / "model.safetensors"),
            )
            self.assertEqual(
                lineage["initialized_from_config_sha256"],
                _sha256(initial / "config.json"),
            )

    def test_resume_before_first_checkpoint_restarts_verified_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, text in (
                ("train", "A tiny language document."),
                ("validation", "A separate tiny validation document."),
            ):
                source = root / "data" / "text" / split / "text.txt"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(text, encoding="utf-8")
            _json(root / "config.json", _config(text_ratio=1.0))
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
                        }
                    ],
                },
            )
            with patch(
                "mimillm.pipeline.train_model",
                side_effect=RuntimeError("simulated crash before checkpoint"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    train_pipeline(root / "pipeline.json", backend="python")
            output = root / "weights" / "language"
            self.assertTrue((output / "lineage.json").is_file())
            self.assertFalse((output / "training_checkpoint.bin").exists())

            resumed = train_pipeline(
                root / "pipeline.json", backend="python", resume_stage="language",
            )
            self.assertEqual(resumed.stages[-1].step, 1)
            lineage = json.loads(
                (output / "lineage.json").read_text(encoding="utf-8")
            )
            self.assertEqual(lineage["status"], "complete")

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

    def test_generation_gate_can_promote_better_last_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = TransformerConfig.from_dict(_config(text_ratio=1.0))
            best_model = DecoderTransformer(config)
            last_model = DecoderTransformer(config)
            next(iter(best_model.parameters())).data[0] = 0.1
            next(iter(last_model.parameters())).data[0] = 0.9
            save_model(root, best_model)
            save_model(root / "last", last_model)
            evaluation_path = root / "evaluation.json"
            _json(evaluation_path, {})
            stage = PipelineStage(
                "core", "sft", root / "config.json", root,
                evaluation_path=evaluation_path,
            )
            result = TrainingResult(
                best_model, root, root / "training_checkpoint.bin", 1, False,
            )
            failed = DialogueEvaluationReport(
                cases=2, passed=0, pass_rate=0.0, min_pass_rate=0.5,
                ok=False, results=(),
            )
            passed = DialogueEvaluationReport(
                cases=2, passed=1, pass_rate=0.5, min_pass_rate=0.5,
                ok=True, results=(),
            )
            with patch(
                "mimillm.pipeline.evaluate_dialogues",
                side_effect=(failed, passed),
            ):
                report, candidate = _evaluate_generation_candidates(result, stage)
            self.assertTrue(report.ok)
            self.assertEqual(candidate, "last")
            promoted = load_model(root)
            self.assertAlmostEqual(
                next(iter(promoted.parameters())).data[0], 0.9, places=6,
            )
            archived = load_model(root / "best_validation")
            self.assertAlmostEqual(
                next(iter(archived.parameters())).data[0], 0.1, places=6,
            )
            candidates = json.loads(
                (root / "generation_candidates.json").read_text(encoding="utf-8")
            )
            self.assertTrue(candidates["promoted"])
            self.assertEqual(candidates["selected"], "last")

    def test_generation_gate_can_promote_intermediate_validation_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = TransformerConfig.from_dict(_config(text_ratio=1.0))
            best_model = DecoderTransformer(config)
            middle_model = DecoderTransformer(config)
            next(iter(best_model.parameters())).data[0] = 0.1
            next(iter(middle_model.parameters())).data[0] = 0.7
            save_model(root, best_model)
            middle = root / "validation" / "step_00000020"
            save_model(middle, middle_model)
            evaluation_path = root / "evaluation.json"
            _json(evaluation_path, {})
            stage = PipelineStage(
                "assistant", "sft", root / "config.json", root,
                evaluation_path=evaluation_path,
            )
            result = TrainingResult(
                best_model, root, root / "training_checkpoint.bin", 20, False,
            )
            failed = DialogueEvaluationReport(
                cases=2, passed=0, pass_rate=0.0, min_pass_rate=0.5,
                ok=False, results=(),
            )
            passed = DialogueEvaluationReport(
                cases=2, passed=2, pass_rate=1.0, min_pass_rate=0.5,
                ok=True, results=(),
            )
            with patch(
                "mimillm.pipeline.evaluate_dialogues",
                side_effect=(failed, passed),
            ):
                report, candidate = _evaluate_generation_candidates(result, stage)
            self.assertTrue(report.ok)
            self.assertEqual(candidate, "step_00000020")
            promoted = load_model(root)
            self.assertAlmostEqual(
                next(iter(promoted.parameters())).data[0], 0.7, places=6,
            )

    def test_validation_loss_gate_stops_pipeline_and_updates_lineage(self) -> None:
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
                            "max_validation_loss": 1e-12,
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(PipelineQualityError, "validation loss"):
                train_pipeline(root / "pipeline.json", backend="python")
            lineage = json.loads(
                (root / "weights" / "language" / "lineage.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(lineage["status"], "quality_failed")
            self.assertFalse(lineage["validation_loss_gate"]["passed"])

    def test_validation_improvement_gate_records_progress_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, text in (
                ("train", "A language training document with several words."),
                ("validation", "A separate validation sentence with other words."),
            ):
                source = root / "data" / "text" / split / "text.txt"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(text, encoding="utf-8")
            config = _config(text_ratio=1.0)
            config["steps"] = 2
            config["save_validation_checkpoints"] = True
            _json(root / "config.json", config)
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
                            "min_validation_loss_improvement": 1000.0,
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(
                PipelineQualityError, "validation loss improvement",
            ):
                train_pipeline(root / "pipeline.json", backend="python")
            lineage = json.loads(
                (root / "weights" / "language" / "lineage.json").read_text(
                    encoding="utf-8",
                )
            )
            gate = lineage["validation_loss_gate"]
            self.assertEqual(gate["first_step"], 1)
            self.assertEqual(gate["best_step"], 2)
            self.assertIn("improvement", gate)
            self.assertFalse(gate["passed"])
            self.assertGreater(gate["improvement"], 0.0)

            pipeline_path = root / "pipeline.json"
            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
            pipeline["stages"][0]["min_validation_loss_improvement"] = (
                gate["improvement"] / 2.0
            )
            _json(pipeline_path, pipeline)
            result = train_pipeline(
                pipeline_path, backend="python", resume_stage="language",
            )
            self.assertEqual(result.stages[0].step, 2)
            resumed = json.loads(
                (root / "weights" / "language" / "lineage.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(resumed["status"], "complete")
            self.assertTrue(resumed["validation_loss_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
