"""Validated language-pretraining -> dialogue-SFT training pipelines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .api import load_model, save_model
from .audit import DatasetAuditReport, audit_dataset, normalize_training_text
from .backend import reset_backend
from .dataset import load_qa_text, load_text_documents
from .evaluation import (
    DialogueEvaluationReport,
    evaluate_dialogues,
    save_dialogue_evaluation,
)
from .tokenizer import (
    BpeTokenizer,
    ByteTokenizer,
    TokenizerReport,
    UnicodeByteTokenizer,
    analyze_tokenizer,
    create_tokenizer,
    format_qa_text,
    load_tokenizer,
    save_tokenizer,
    train_bpe_tokenizer,
)
from .training import TrainingResult, train_model
from .transformer import TransformerConfig


PIPELINE_VERSION = 1
_ARCHITECTURE_FIELDS = (
    "vocab_size", "tokenizer", "tie_word_embeddings", "context_length",
    "d_model", "n_layers", "n_heads", "d_mlp",
)
_RESUME_MUTABLE_FIELDS = {
    "steps", "validation_interval", "checkpoint_interval",
    "save_validation_checkpoints", "early_stopping_patience",
    "early_stopping_min_delta", "batches_per_epoch",
}


@dataclass(frozen=True)
class PipelineStage:
    """One validated stage in a linear training curriculum."""

    name: str
    kind: str
    config_path: Path
    output_dir: Path
    evaluation_path: Path | None = None
    max_validation_loss: float | None = None
    min_validation_loss_improvement: float | None = None


class PipelineQualityError(RuntimeError):
    """A stage finished training but did not meet its configured quality gates."""


@dataclass(frozen=True)
class PipelineResult:
    """Outputs of a complete or safely interrupted pipeline run."""

    pipeline_path: Path
    tokenizer_path: Path | None
    stages: tuple[TrainingResult, ...]
    final_weights: Path | None
    interrupted_stage: str | None


def _write_json(path: Path, values: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(values, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(values: dict[str, Any]) -> str:
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parent_artifact_hashes(parent_weights: Path | None) -> dict[str, str | None]:
    if parent_weights is None:
        return {"model": None, "config": None, "tokenizer": None}
    if parent_weights.is_dir():
        paths = {
            "model": parent_weights / "model.safetensors",
            "config": parent_weights / "config.json",
            "tokenizer": parent_weights / "tokenizer.json",
        }
    else:
        paths = {
            "model": parent_weights,
            "config": parent_weights.with_name("config.json"),
            "tokenizer": parent_weights.with_name("tokenizer.json"),
        }
    return {
        name: _sha256(artifact) if artifact.is_file() else None
        for name, artifact in paths.items()
    }


def _path(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_pipeline(path: Path) -> tuple[dict[str, Any], list[PipelineStage]]:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid pipeline JSON: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError("pipeline JSON root must be an object")
    if values.get("version") != PIPELINE_VERSION:
        raise ValueError(f"pipeline version must be {PIPELINE_VERSION}")
    root_unknown = sorted(
        set(values) - {
            "version", "name", "base_config", "tokenizer", "dataset_checks",
            "allow_sft_from_scratch", "initial_weights", "stages",
        }
    )
    if root_unknown:
        raise ValueError(f"pipeline has unknown fields: {root_unknown}")
    raw_stages = values.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("pipeline stages must be a non-empty list")
    stages: list[PipelineStage] = []
    names: set[str] = set()
    outputs: set[Path] = set()
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, dict):
            raise ValueError(f"pipeline stage {index} must be an object")
        unknown = sorted(
            set(raw) - {
                "name", "kind", "config", "output_dir", "evaluation",
                "max_validation_loss", "min_validation_loss_improvement",
            }
        )
        if unknown:
            raise ValueError(f"pipeline stage {index} has unknown fields: {unknown}")
        name = raw.get("name")
        kind = raw.get("kind")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"pipeline stage {index} needs a non-empty name")
        name = name.strip()
        if name in names:
            raise ValueError(f"duplicate pipeline stage name: {name}")
        if kind not in {"pretrain", "sft"}:
            raise ValueError(f"stage {name}: kind must be 'pretrain' or 'sft'")
        config_path = _path(path.parent, raw.get("config"), f"stage {name}.config")
        output_dir = _path(
            path.parent, raw.get("output_dir"), f"stage {name}.output_dir",
        )
        if output_dir in outputs:
            raise ValueError(f"stage {name}: output_dir is already used")
        if not config_path.is_file():
            raise FileNotFoundError(f"stage {name} config not found: {config_path}")
        evaluation_path = None
        if "evaluation" in raw:
            evaluation_path = _path(
                path.parent, raw["evaluation"], f"stage {name}.evaluation",
            )
            if not evaluation_path.is_file():
                raise FileNotFoundError(
                    f"stage {name} evaluation suite not found: {evaluation_path}"
                )
        max_validation_loss = raw.get("max_validation_loss")
        if max_validation_loss is not None:
            if (
                not isinstance(max_validation_loss, (int, float))
                or isinstance(max_validation_loss, bool)
                or not math.isfinite(max_validation_loss)
                or max_validation_loss <= 0.0
            ):
                raise ValueError(
                    f"stage {name}.max_validation_loss must be a positive number"
                )
            max_validation_loss = float(max_validation_loss)
        min_validation_loss_improvement = raw.get(
            "min_validation_loss_improvement"
        )
        if min_validation_loss_improvement is not None:
            if (
                not isinstance(min_validation_loss_improvement, (int, float))
                or isinstance(min_validation_loss_improvement, bool)
                or not math.isfinite(min_validation_loss_improvement)
                or min_validation_loss_improvement <= 0.0
            ):
                raise ValueError(
                    f"stage {name}.min_validation_loss_improvement must be "
                    "a positive number"
                )
            min_validation_loss_improvement = float(
                min_validation_loss_improvement
            )
        names.add(name)
        outputs.add(output_dir)
        stages.append(
            PipelineStage(
                name, kind, config_path, output_dir, evaluation_path,
                max_validation_loss, min_validation_loss_improvement,
            )
        )
    allow_sft = values.get("allow_sft_from_scratch", False)
    if not isinstance(allow_sft, bool):
        raise TypeError("allow_sft_from_scratch must be a boolean")
    initial_weights = None
    if "initial_weights" in values:
        initial_weights = _path(path.parent, values["initial_weights"], "initial_weights")
        if not (initial_weights / "config.json").is_file() or not (
            initial_weights / "model.safetensors"
        ).is_file():
            raise FileNotFoundError(
                "initial_weights must contain config.json and model.safetensors: "
                f"{initial_weights}"
            )
    if stages[0].kind != "pretrain" and initial_weights is None and not allow_sft:
        raise ValueError(
            "the first stage must be pretrain or the pipeline must declare "
            "initial_weights; set allow_sft_from_scratch=true only for an "
            "intentional diagnostic"
        )
    return values, stages


def _load_stage_configs(
    pipeline_path: Path,
    pipeline: dict[str, Any],
    stages: list[PipelineStage],
) -> list[TransformerConfig]:
    base_values: dict[str, Any] = {}
    if "base_config" in pipeline:
        base_path = _path(
            pipeline_path.parent, pipeline["base_config"], "base_config",
        )
        try:
            loaded_base = json.loads(base_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid base config JSON: {exc}") from exc
        if not isinstance(loaded_base, dict):
            raise ValueError("base_config JSON root must be an object")
        base_values = loaded_base
    configs: list[TransformerConfig] = []
    for stage in stages:
        try:
            stage_values = json.loads(stage.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"stage {stage.name} has invalid config JSON: {exc}") from exc
        if not isinstance(stage_values, dict):
            raise ValueError(f"stage {stage.name} config root must be an object")
        configs.append(TransformerConfig.from_dict({**base_values, **stage_values}))
    return configs


def _corpus(
    config: TransformerConfig, base_dir: Path, *, split: str,
) -> Iterable[str]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if config.text_ratio > 0.0:
        configured = (
            config.text_train_path if split == "train" else config.text_validation_path
        )
        path = (base_dir / configured).resolve() if not Path(configured).is_absolute() else Path(configured)
        yield from (text for _source, text in load_text_documents(path))
    if config.text_ratio < 1.0:
        configured = (
            config.question_train_path
            if split == "train"
            else config.question_validation_path
        )
        path = (base_dir / configured).resolve() if not Path(configured).is_absolute() else Path(configured)
        yield from (
            format_qa_text(question, answer)
            for question, answer in load_qa_text(path)
        )


def _combined_corpus(
    configs: list[TransformerConfig],
    stages: list[PipelineStage],
    *,
    split: str,
) -> Iterable[str]:
    """Yields unique documents so replayed sources do not bias the tokenizer."""
    seen: set[str] = set()
    for config, stage in zip(configs, stages):
        for text in _corpus(config, stage.config_path.parent, split=split):
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                yield text


def _active_paths(
    config: TransformerConfig, base_dir: Path,
) -> dict[str, Path | None]:
    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()

    return {
        "question_train_path": resolve(config.question_train_path) if config.text_ratio < 1.0 else None,
        "question_validation_path": resolve(config.question_validation_path) if config.text_ratio < 1.0 else None,
        "text_train_path": resolve(config.text_train_path) if config.text_ratio > 0.0 else None,
        "text_validation_path": resolve(config.text_validation_path) if config.text_ratio > 0.0 else None,
    }


def _validate_stage(stage: PipelineStage, config: TransformerConfig) -> None:
    if stage.kind == "pretrain" and config.text_ratio != 1.0:
        raise ValueError(
            f"stage {stage.name}: pretrain requires text_ratio=1.0; put complete "
            "language/dialogue documents in the text corpus"
        )
    if stage.kind == "sft":
        if config.text_ratio >= 1.0:
            raise ValueError(f"stage {stage.name}: sft requires a QA/dialogue source")
        if config.qa_prompt_weight != 0.0:
            raise ValueError(
                f"stage {stage.name}: sft requires qa_prompt_weight=0 so loss "
                "targets assistant answers rather than memorizing prompts"
            )


def _tokenizer_settings(values: dict[str, Any]) -> dict[str, Any]:
    raw = values.get("tokenizer")
    if not isinstance(raw, dict):
        raise ValueError("pipeline tokenizer must be an object")
    allowed = {
        "type", "path", "vocab_size", "min_frequency",
        "ensure_unicode_characters", "retrain", "max_compression_ratio",
        "min_unicode_atomic_coverage", "required_pieces",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"pipeline tokenizer has unknown fields: {unknown}")
    tokenizer_type = raw.get("type", "bpe")
    if tokenizer_type not in {"byte", "unicode", "bpe"}:
        raise ValueError("pipeline tokenizer.type must be byte, unicode, or bpe")
    settings = dict(raw)
    settings["type"] = tokenizer_type
    required_pieces = settings.get("required_pieces", [])
    if (
        not isinstance(required_pieces, list)
        or not all(isinstance(piece, str) and piece for piece in required_pieces)
        or len(set(required_pieces)) != len(required_pieces)
    ):
        raise ValueError(
            "pipeline tokenizer.required_pieces must be a list of unique "
            "non-empty strings"
        )
    if required_pieces and tokenizer_type != "bpe":
        raise ValueError("pipeline tokenizer.required_pieces requires type='bpe'")
    return settings


def _prepare_tokenizer(
    pipeline_path: Path,
    pipeline: dict[str, Any],
    raw_configs: list[TransformerConfig],
    stages: list[PipelineStage],
) -> tuple[ByteTokenizer, Path | None]:
    settings = _tokenizer_settings(pipeline)
    tokenizer_type = settings["type"]
    if tokenizer_type != "bpe":
        return create_tokenizer(tokenizer_type), None
    artifact = _path(
        pipeline_path.parent,
        settings.get("path", "tokenizer.json"),
        "tokenizer.path",
    )
    retrain = settings.get("retrain", False)
    if not isinstance(retrain, bool):
        raise TypeError("tokenizer.retrain must be a boolean")
    required_pieces = settings.get("required_pieces", [])
    if artifact.is_file() and not retrain:
        tokenizer = load_tokenizer(artifact)
        missing = [
            piece for piece in required_pieces
            if len(tokenizer.encode(piece)) != 1
        ]
        if missing:
            raise ValueError(
                "existing BPE tokenizer does not encode required_pieces atomically: "
                f"{missing}; set tokenizer.retrain=true for a new full training run"
            )
        return tokenizer, artifact
    vocab_size = settings.get("vocab_size", 4096)
    min_frequency = settings.get("min_frequency", 2)
    ensure_unicode = settings.get("ensure_unicode_characters", True)
    corpus = _combined_corpus(raw_configs, stages, split="train")
    tokenizer = train_bpe_tokenizer(
        corpus,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        ensure_unicode_characters=ensure_unicode,
        required_pieces=required_pieces,
    )
    save_tokenizer(tokenizer, artifact)
    return tokenizer, artifact


def _audit_stages(
    pipeline_path: Path,
    pipeline: dict[str, Any],
    configs: list[TransformerConfig],
    stages: list[PipelineStage],
) -> dict[str, DatasetAuditReport]:
    raw_checks = pipeline.get("dataset_checks", {})
    if not isinstance(raw_checks, dict):
        raise ValueError("dataset_checks must be an object")
    unknown = sorted(set(raw_checks) - {"enabled", "forbidden_phrases"})
    if unknown:
        raise ValueError(f"dataset_checks has unknown fields: {unknown}")
    enabled = raw_checks.get("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError("dataset_checks.enabled must be a boolean")
    forbidden = raw_checks.get("forbidden_phrases", [])
    if not isinstance(forbidden, list):
        raise TypeError("dataset_checks.forbidden_phrases must be a list")
    if not enabled:
        return {}
    reports: dict[str, DatasetAuditReport] = {}
    for config, stage in zip(configs, stages):
        report = audit_dataset(
            **_active_paths(config, stage.config_path.parent),
            forbidden_phrases=forbidden,
        )
        reports[stage.name] = report
    train_documents = {
        normalize_training_text(text)
        for text in _combined_corpus(configs, stages, split="train")
    }
    validation_documents = {
        normalize_training_text(text)
        for text in _combined_corpus(configs, stages, split="validation")
    }
    global_overlap = len(train_documents & validation_documents)
    _write_json(
        pipeline_path.parent / "pipeline_audit.json",
        {
            "stages": {name: report.to_dict() for name, report in reports.items()},
            "global_train_validation_overlap": global_overlap,
        },
    )
    failures = [name for name, report in reports.items() if not report.ok]
    if failures:
        details = "; ".join(
            f"{name}: {', '.join(reports[name].errors)}" for name in failures
        )
        raise ValueError(f"dataset audit failed before training: {details}")
    if global_overlap:
        raise ValueError(
            "dataset audit failed before training: "
            f"{global_overlap} documents occur in train and validation across stages"
        )
    return reports


def _save_tokenizer_reports(
    pipeline_path: Path,
    tokenizer: ByteTokenizer,
    configs: list[TransformerConfig],
    stages: list[PipelineStage],
) -> dict[str, TokenizerReport]:
    reports = {
        "train": analyze_tokenizer(
            tokenizer,
            _combined_corpus(configs, stages, split="train"),
        ),
        "validation": analyze_tokenizer(
            tokenizer,
            _combined_corpus(configs, stages, split="validation"),
        ),
    }
    _write_json(
        pipeline_path.parent / "tokenizer_report.json",
        {name: report.to_dict() for name, report in reports.items()},
    )
    return reports


def _validate_tokenizer_reports(
    pipeline: dict[str, Any],
    tokenizer: ByteTokenizer,
    reports: dict[str, TokenizerReport],
) -> None:
    if not isinstance(tokenizer, BpeTokenizer):
        return
    settings = _tokenizer_settings(pipeline)
    maximum = settings.get("max_compression_ratio", 0.80)
    minimum_unicode = settings.get("min_unicode_atomic_coverage", 0.95)
    if (
        not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or not 0.0 < maximum <= 1.0
    ):
        raise ValueError("tokenizer.max_compression_ratio must be in (0, 1]")
    if (
        not isinstance(minimum_unicode, (int, float))
        or isinstance(minimum_unicode, bool)
        or not 0.0 <= minimum_unicode <= 1.0
    ):
        raise ValueError("tokenizer.min_unicode_atomic_coverage must be in [0, 1]")
    failures: list[str] = []
    for split, report in reports.items():
        if report.roundtrip_errors:
            failures.append(f"{split} has {report.roundtrip_errors} round-trip errors")
        if report.compression_ratio > float(maximum):
            failures.append(
                f"{split} tokens/byte {report.compression_ratio:.3f} exceeds {maximum}"
            )
        if report.unicode_atomic_coverage < float(minimum_unicode):
            failures.append(
                f"{split} Unicode coverage {report.unicode_atomic_coverage:.1%} "
                f"is below {float(minimum_unicode):.1%}"
            )
    if failures:
        raise ValueError(
            "tokenizer quality gate failed before training: " + "; ".join(failures)
        )


def _architecture(config: TransformerConfig) -> tuple[object, ...]:
    return tuple(getattr(config, field) for field in _ARCHITECTURE_FIELDS)


def _lineage(
    *,
    pipeline_path: Path,
    pipeline: dict[str, Any],
    stage: PipelineStage,
    config: TransformerConfig,
    tokenizer_path: Path | None,
    parent_stage: str | None,
    parent_weights: Path | None,
    status: str,
    result: TrainingResult | None = None,
    evaluation: DialogueEvaluationReport | None = None,
    validation_loss_gate: dict[str, object] | None = None,
    evaluation_candidate: str | None = None,
) -> dict[str, Any]:
    parent_hashes = _parent_artifact_hashes(parent_weights)
    values: dict[str, Any] = {
        "format": "mimiLLM-lineage-v1",
        "pipeline": str(pipeline_path),
        "pipeline_sha256": _json_sha256(pipeline),
        "stage": stage.name,
        "kind": stage.kind,
        "status": status,
        "config": str(stage.config_path),
        "effective_config": config.to_dict(),
        "effective_config_sha256": _json_sha256(config.to_dict()),
        "tokenizer": str(tokenizer_path) if tokenizer_path else config.tokenizer,
        "tokenizer_sha256": _sha256(tokenizer_path) if tokenizer_path else None,
        "parent_stage": parent_stage,
        "initialized_from": str(parent_weights) if parent_weights else None,
        "initialized_from_model_sha256": parent_hashes["model"],
        "initialized_from_config_sha256": parent_hashes["config"],
        "initialized_from_tokenizer_sha256": parent_hashes["tokenizer"],
        "optimizer_reset": parent_weights is not None,
        "evaluation_suite": (
            str(stage.evaluation_path) if stage.evaluation_path else None
        ),
    }
    if result is not None:
        model_path = result.weights_dir / "model.safetensors"
        values.update({
            "step": result.step,
            "interrupted": result.interrupted,
            "weights": str(result.weights_dir),
            "model_sha256": _sha256(model_path) if model_path.is_file() else None,
            "checkpoint": str(result.checkpoint_path),
        })
    if evaluation is not None:
        values["generation_evaluation"] = {
            "report": str(stage.output_dir / "generation_report.json"),
            "candidate": evaluation_candidate,
            "passed": evaluation.passed,
            "cases": evaluation.cases,
            "pass_rate": evaluation.pass_rate,
            "min_pass_rate": evaluation.min_pass_rate,
        }
    if validation_loss_gate is not None:
        values["validation_loss_gate"] = validation_loss_gate
    return values


def _best_validation(weights_dir: Path) -> tuple[float, int]:
    path = weights_dir / "best_validation.json"
    if not path.is_file():
        raise FileNotFoundError(f"best validation metadata not found: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    loss = values.get("loss")
    step = values.get("step")
    if (
        not isinstance(loss, (int, float))
        or isinstance(loss, bool)
        or not math.isfinite(loss)
        or not isinstance(step, int)
        or isinstance(step, bool)
    ):
        raise ValueError(f"invalid best validation metadata: {path}")
    configured_weights = values.get("weights")
    best_weights = (
        Path(configured_weights)
        if isinstance(configured_weights, str) and configured_weights
        else weights_dir
    )
    if not best_weights.is_absolute():
        best_weights = (weights_dir / best_weights).resolve()
    expected_hash = values.get("model_sha256")
    model_path = best_weights / "model.safetensors"
    if expected_hash is not None and (
        not isinstance(expected_hash, str)
        or not model_path.is_file()
        or _sha256(model_path) != expected_hash
    ):
        raise ValueError(
            f"best validation metadata does not match its model weights: {path}"
        )
    return float(loss), step


def _validation_loss_progress(
    weights_dir: Path,
) -> tuple[float, int, float, int]:
    """Returns the first and best recorded validation losses for a stage."""
    snapshots = weights_dir / "validation"
    records: list[tuple[int, float]] = []
    for path in sorted(snapshots.glob("step_*/validation.json")):
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid validation metadata: {path}") from exc
        loss = values.get("loss")
        step = values.get("step")
        if (
            not isinstance(loss, (int, float))
            or isinstance(loss, bool)
            or not math.isfinite(loss)
            or not isinstance(step, int)
            or isinstance(step, bool)
            or step <= 0
        ):
            raise ValueError(f"invalid validation metadata: {path}")
        records.append((step, float(loss)))
    if len(records) < 2:
        raise ValueError(
            "min_validation_loss_improvement requires at least two validation "
            "snapshots; enable save_validation_checkpoints and validate more "
            f"than once: {snapshots}"
        )
    records.sort()
    first_step, first_loss = records[0]
    best_step, best_loss = min(records, key=lambda item: item[1])
    return first_loss, first_step, best_loss, best_step


def _best_validation_weights(weights_dir: Path) -> Path:
    """Returns the immutable validation-best artifact when metadata provides it."""
    path = weights_dir / "best_validation.json"
    if not path.is_file():
        return weights_dir
    values = json.loads(path.read_text(encoding="utf-8"))
    configured = values.get("weights")
    if not isinstance(configured, str) or not configured:
        return weights_dir
    selected = Path(configured)
    selected = selected if selected.is_absolute() else (weights_dir / selected).resolve()
    model_path = selected / "model.safetensors"
    expected_hash = values.get("model_sha256")
    if expected_hash is not None and (
        not isinstance(expected_hash, str)
        or not model_path.is_file()
        or _sha256(model_path) != expected_hash
    ):
        raise ValueError(
            f"best validation metadata does not match its model weights: {path}"
        )
    return selected


def _evaluate_generation_candidates(
    result: TrainingResult,
    stage: PipelineStage,
) -> tuple[DialogueEvaluationReport, str]:
    """Compare lowest-validation-loss and final weights on real generation."""
    if stage.evaluation_path is None:
        raise ValueError("generation candidates require an evaluation suite")
    validation_best = _best_validation_weights(result.weights_dir)
    candidates = [("best_validation", validation_best)]
    seen = {validation_best.resolve()}
    validation_dir = result.weights_dir / "validation"
    if validation_dir.is_dir():
        for checkpoint in sorted(validation_dir.glob("step_*")):
            resolved = checkpoint.resolve()
            if (
                resolved not in seen
                and (checkpoint / "model.safetensors").is_file()
            ):
                candidates.append((checkpoint.name, checkpoint))
                seen.add(resolved)
    last_dir = result.weights_dir / "last"
    if (
        last_dir.resolve() not in seen
        and (last_dir / "model.safetensors").is_file()
    ):
        candidates.append(("last", last_dir))

    evaluated: list[tuple[str, Path, DialogueEvaluationReport]] = []
    for name, weights in candidates:
        report = evaluate_dialogues(load_model(weights), stage.evaluation_path)
        evaluated.append((name, weights, report))
        print(
            f"Generation candidate [{stage.name}/{name}]: {report.passed}/"
            f"{report.cases} cases passed ({report.pass_rate:.1%})",
            flush=True,
        )
    selected_name, selected_weights, selected_report = max(
        evaluated,
        key=lambda item: (
            item[2].passed,
            item[2].pass_rate,
            item[0] == "best_validation",
        ),
    )
    promoted = selected_name != "best_validation" and selected_report.ok
    if promoted:
        archived_best = validation_best
        if archived_best == result.weights_dir:
            archived_best = result.weights_dir / "best_validation"
            save_model(archived_best, load_model(result.weights_dir))
        save_model(result.weights_dir, load_model(selected_weights))
        evaluated = [
            (
                name,
                archived_best if name == "best_validation" else weights,
                report,
            )
            for name, weights, report in evaluated
        ]
        print(
            f"Promoted generation candidate [{stage.name}/{selected_name}] to "
            f"{result.weights_dir}; validation-best weights archived at "
            f"{archived_best}",
            flush=True,
        )
    save_dialogue_evaluation(
        selected_report, stage.output_dir / "generation_report.json",
    )
    _write_json(
        stage.output_dir / "generation_candidates.json",
        {
            "format": "mimiLLM-generation-candidates-v1",
            "selected": selected_name,
            "promoted": promoted,
            "candidates": {
                name: {
                    "weights": str(weights),
                    **report.to_dict(),
                }
                for name, weights, report in evaluated
            },
        },
    )
    return selected_report, selected_name


def train_pipeline(
    pipeline_path: str | Path = "pipeline.json",
    *,
    backend: str | None = None,
    resume_stage: str | None = None,
) -> PipelineResult:
    """Runs a checked linear curriculum and automatically connects its stages.

    A normal pipeline starts with language pretraining. A follow-up curriculum
    may instead declare ``initial_weights`` and begin with SFT from an existing
    compatible model. Every later stage receives the preceding best weights;
    users never need to wire per-stage ``init_from`` paths manually. Dataset
    audits and tokenizer reports are written before the first optimizer step.
    """
    path = Path(pipeline_path).resolve()
    pipeline, stages = _load_pipeline(path)
    if resume_stage is not None and resume_stage not in {stage.name for stage in stages}:
        raise ValueError(f"unknown resume stage: {resume_stage}")
    if resume_stage is None:
        occupied = [
            stage.output_dir
            for stage in stages
            if stage.output_dir.exists() and any(stage.output_dir.iterdir())
        ]
        if occupied:
            raise FileExistsError(
                "stage outputs are not empty; use new output directories or "
                f"resume_stage instead of overwriting weights: {occupied}"
            )
    elif _tokenizer_settings(pipeline).get("retrain", False):
        raise ValueError("tokenizer.retrain cannot be used while resuming a stage")
    raw_configs = _load_stage_configs(path, pipeline, stages)
    for stage, config in zip(stages, raw_configs):
        _validate_stage(stage, config)
    tokenizer, tokenizer_path = _prepare_tokenizer(path, pipeline, raw_configs, stages)
    effective_configs = [
        replace(
            config,
            tokenizer=(
                "bpe" if isinstance(tokenizer, BpeTokenizer)
                else "unicode" if isinstance(tokenizer, UnicodeByteTokenizer)
                else "byte"
            ),
            tokenizer_path=str(tokenizer_path or config.tokenizer_path),
            vocab_size=tokenizer.VOCAB_SIZE,
        )
        for config in raw_configs
    ]
    expected_architecture = _architecture(effective_configs[0])
    for stage, config in zip(stages[1:], effective_configs[1:]):
        if _architecture(config) != expected_architecture:
            raise ValueError(
                f"stage {stage.name}: model architecture differs from the first stage"
            )
    reports = _audit_stages(path, pipeline, effective_configs, stages)
    tokenizer_reports = _save_tokenizer_reports(
        path, tokenizer, effective_configs, stages,
    )
    _validate_tokenizer_reports(pipeline, tokenizer, tokenizer_reports)
    for name, report in reports.items():
        for warning in report.warnings:
            print(f"Dataset warning [{name}]: {warning}", flush=True)
    for split, report in tokenizer_reports.items():
        print(
            f"Tokenizer [{split}]: tokens/byte={report.compression_ratio:.3f} | "
            f"tokens/word={report.tokens_per_word:.2f} | "
            f"unicode_atomic={report.unicode_atomic_coverage:.1%}",
            flush=True,
        )
        for warning in report.warnings:
            print(f"Tokenizer warning [{split}]: {warning}", flush=True)

    if backend is not None:
        selected = backend.strip().lower()
        if selected not in {"auto", "cuda", "cpp", "python"}:
            raise ValueError("backend must be auto, cuda, cpp, or python")
        os.environ["MIMILLM_BACKEND"] = selected
        reset_backend()

    results: list[TrainingResult] = []
    final_weights: Path | None = None
    parent_weights = (
        _path(path.parent, pipeline["initial_weights"], "initial_weights")
        if "initial_weights" in pipeline else None
    )
    parent_stage: str | None = "initial_weights" if parent_weights else None
    skipping = resume_stage is not None
    for stage, config in zip(stages, effective_configs):
        lineage_path = stage.output_dir / "lineage.json"
        if skipping and stage.name != resume_stage:
            if not lineage_path.is_file():
                raise FileNotFoundError(
                    f"cannot skip incomplete parent stage {stage.name}: {lineage_path}"
                )
            previous = json.loads(lineage_path.read_text(encoding="utf-8"))
            if previous.get("status") != "complete":
                raise ValueError(f"parent stage {stage.name} is not complete")
            previous_evaluation = previous.get("generation_evaluation")
            previous_candidate = (
                previous_evaluation.get("candidate")
                if isinstance(previous_evaluation, dict) else None
            )
            parent_weights = (
                stage.output_dir
                if previous_candidate == "last"
                else _best_validation_weights(stage.output_dir)
            )
            parent_stage = stage.name
            final_weights = stage.output_dir
            continue
        skipping = False
        resume: Path | None = None
        initialized_from = parent_weights
        if resume_stage == stage.name:
            if not lineage_path.is_file():
                raise FileNotFoundError(
                    f"stage lineage not found; refusing an unverified resume: {lineage_path}"
                )
            previous = json.loads(lineage_path.read_text(encoding="utf-8"))
            if previous.get("format") != "mimiLLM-lineage-v1":
                raise ValueError(f"stage {stage.name} has invalid lineage metadata")
            previous_config = previous.get("effective_config")
            if not isinstance(previous_config, dict):
                raise ValueError(f"stage {stage.name} lineage has no effective config")
            current_config = config.to_dict()
            changed = {
                key
                for key in set(previous_config) | set(current_config)
                if previous_config.get(key) != current_config.get(key)
            }
            disallowed = sorted(changed - _RESUME_MUTABLE_FIELDS)
            if disallowed:
                raise ValueError(
                    f"stage {stage.name} changed immutable resume fields: {disallowed}"
                )
            expected_tokenizer_hash = _sha256(tokenizer_path) if tokenizer_path else None
            if previous.get("tokenizer_sha256") != expected_tokenizer_hash:
                raise ValueError(
                    f"stage {stage.name} tokenizer changed since the checkpoint was created"
                )
            current_parent_hashes = _parent_artifact_hashes(parent_weights)
            recorded_parent_hash = previous.get("initialized_from_model_sha256")
            if (
                parent_weights is not None
                and recorded_parent_hash is not None
                and recorded_parent_hash != current_parent_hashes["model"]
            ):
                raise ValueError(
                    f"stage {stage.name} parent weights changed since the stage began"
                )
            interrupted = stage.output_dir / "training_checkpoint_interrupted.bin"
            regular = stage.output_dir / "training_checkpoint.bin"
            resume = interrupted if interrupted.is_file() else regular
            if not resume.is_file():
                if previous.get("status") != "running" or "step" in previous:
                    raise FileNotFoundError(
                        f"stage checkpoint not found and the stage cannot be safely "
                        f"restarted: {stage.output_dir}"
                    )
                if parent_weights is not None and (
                    recorded_parent_hash is None
                    or recorded_parent_hash != current_parent_hashes["model"]
                ):
                    raise ValueError(
                        f"stage {stage.name} cannot restart without the exact verified "
                        "parent weights"
                    )
                resume = None
                initialized_from = parent_weights
                print(
                    f"Stage {stage.name} stopped before its first checkpoint; "
                    "restarting the stage from its verified inputs",
                    flush=True,
                )
            else:
                initialized_from = None
        elif stage.output_dir.exists() and any(stage.output_dir.iterdir()):
            raise FileExistsError(
                f"stage output is not empty: {stage.output_dir}; use a new output "
                "or resume_stage instead of silently overwriting weights"
            )
        parent_hashes_before = _parent_artifact_hashes(parent_weights)
        stage.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            lineage_path,
            _lineage(
                pipeline_path=path, pipeline=pipeline, stage=stage,
                config=config, tokenizer_path=tokenizer_path,
                parent_stage=parent_stage, parent_weights=parent_weights,
                status="running",
            ),
        )
        _write_json(
            path.parent / "pipeline_state.json",
            {
                "format": "mimiLLM-pipeline-state-v1",
                "pipeline": str(path),
                "stage": stage.name,
                "status": "running",
                "weights": str(stage.output_dir),
                "checkpoint": str(resume) if resume is not None else None,
            },
        )
        result = train_model(
            config,
            base_dir=stage.config_path.parent,
            output_dir=stage.output_dir,
            resume=resume,
            init_from=initialized_from,
        )
        if _parent_artifact_hashes(parent_weights) != parent_hashes_before:
            raise RuntimeError(
                f"stage {stage.name} parent artifacts changed while training"
            )
        results.append(result)
        evaluation: DialogueEvaluationReport | None = None
        evaluation_candidate: str | None = None
        validation_loss_gate: dict[str, object] | None = None
        validation_loss_ok = True
        if not result.interrupted and (
            stage.max_validation_loss is not None
            or stage.min_validation_loss_improvement is not None
        ):
            best_loss, best_step = _best_validation(result.weights_dir)
            maximum_ok = (
                stage.max_validation_loss is None
                or best_loss <= stage.max_validation_loss
            )
            validation_loss_gate = {
                "best_loss": best_loss,
                "best_step": best_step,
                "max_loss": stage.max_validation_loss,
                "min_improvement": stage.min_validation_loss_improvement,
            }
            improvement_ok = True
            if stage.min_validation_loss_improvement is not None:
                first_loss, first_step, recorded_best, recorded_best_step = (
                    _validation_loss_progress(result.weights_dir)
                )
                if abs(recorded_best - best_loss) > 1e-9:
                    raise ValueError(
                        "validation snapshots disagree with best_validation.json: "
                        f"{result.weights_dir}"
                    )
                improvement = first_loss - recorded_best
                improvement_ok = (
                    improvement >= stage.min_validation_loss_improvement
                )
                validation_loss_gate.update({
                    "first_loss": first_loss,
                    "first_step": first_step,
                    "best_step": recorded_best_step,
                    "improvement": improvement,
                })
            validation_loss_ok = maximum_ok and improvement_ok
            validation_loss_gate["passed"] = validation_loss_ok
            requirements: list[str] = []
            if stage.max_validation_loss is not None:
                requirements.append(f"best<={stage.max_validation_loss:g}")
            if stage.min_validation_loss_improvement is not None:
                requirements.append(
                    "improvement>="
                    f"{stage.min_validation_loss_improvement:g}"
                )
            details = ""
            if "improvement" in validation_loss_gate:
                details = (
                    f" | first={validation_loss_gate['first_loss']:.5f} at "
                    f"step {validation_loss_gate['first_step']} | improvement="
                    f"{validation_loss_gate['improvement']:.5f}"
                )
            print(
                f"Validation loss gate [{stage.name}]: best={best_loss:.5f} at "
                f"step {best_step}{details} | required "
                f"{', '.join(requirements)} | "
                f"{'passed' if validation_loss_ok else 'failed'}",
                flush=True,
            )
        if result.interrupted:
            status = "interrupted"
        else:
            if stage.evaluation_path is not None:
                evaluation, evaluation_candidate = _evaluate_generation_candidates(
                    result, stage,
                )
                print(
                    f"Generation gate [{stage.name}]: {evaluation.passed}/"
                    f"{evaluation.cases} cases passed ({evaluation.pass_rate:.1%}; "
                    f"required {evaluation.min_pass_rate:.1%})",
                    flush=True,
                )
            generation_ok = evaluation is None or evaluation.ok
            status = (
                "complete"
                if validation_loss_ok and generation_ok
                else "quality_failed"
            )
        _write_json(
            lineage_path,
            _lineage(
                pipeline_path=path, pipeline=pipeline, stage=stage,
                config=config, tokenizer_path=tokenizer_path,
                parent_stage=parent_stage, parent_weights=parent_weights,
                status=status, result=result, evaluation=evaluation,
                validation_loss_gate=validation_loss_gate,
                evaluation_candidate=evaluation_candidate,
            ),
        )
        _write_json(
            path.parent / "pipeline_state.json",
            {
                "format": "mimiLLM-pipeline-state-v1",
                "pipeline": str(path),
                "stage": stage.name,
                "status": status,
                "weights": str(result.weights_dir),
                "checkpoint": str(result.checkpoint_path),
            },
        )
        if result.interrupted:
            return PipelineResult(
                path, tokenizer_path, tuple(results), None, stage.name,
            )
        if status == "quality_failed":
            failures: list[str] = []
            if not validation_loss_ok and validation_loss_gate is not None:
                if (
                    validation_loss_gate.get("max_loss") is not None
                    and validation_loss_gate["best_loss"]
                    > validation_loss_gate["max_loss"]
                ):
                    failures.append(
                        "best validation loss "
                        f"{validation_loss_gate['best_loss']:.5f} exceeds "
                        f"{validation_loss_gate['max_loss']:g}"
                    )
                if (
                    validation_loss_gate.get("min_improvement") is not None
                    and validation_loss_gate.get("improvement", -math.inf)
                    < validation_loss_gate["min_improvement"]
                ):
                    failures.append(
                        "validation loss improvement "
                        f"{validation_loss_gate['improvement']:.5f} is below "
                        f"{validation_loss_gate['min_improvement']:g}"
                    )
            if evaluation is not None and not evaluation.ok:
                failures.append(
                    f"generation passed {evaluation.passed}/{evaluation.cases} cases; "
                    f"report: {stage.output_dir / 'generation_report.json'}"
                )
            raise PipelineQualityError(
                f"stage {stage.name} failed quality gate(s): {'; '.join(failures)}"
            )
        final_weights = result.weights_dir
        parent_weights = (
            result.weights_dir
            if evaluation_candidate == "last"
            else _best_validation_weights(result.weights_dir)
        )
        parent_stage = stage.name
    return PipelineResult(
        path, tokenizer_path, tuple(results), final_weights, None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a validated mimiLLM pretraining and SFT pipeline",
    )
    parser.add_argument("pipeline", nargs="?", type=Path, default=Path("pipeline.json"))
    parser.add_argument("--backend", choices=("auto", "cuda", "cpp", "python"))
    parser.add_argument("--resume-stage", help="resume one interrupted stage by name")
    args = parser.parse_args()
    try:
        result = train_pipeline(
            args.pipeline, backend=args.backend, resume_stage=args.resume_stage,
        )
    except PipelineQualityError as exc:
        raise SystemExit(f"Pipeline stopped: {exc}") from None
    if result.interrupted_stage:
        print(f"Pipeline interrupted at stage: {result.interrupted_stage}")
    else:
        print(f"Final weights: {result.final_weights}")


if __name__ == "__main__":
    main()
