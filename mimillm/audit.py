"""Dataset checks used before expensive language-model training starts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .dataset import load_qa_text, load_text_documents


_WHITESPACE = re.compile(r"\s+")


def normalize_training_text(text: str) -> str:
    """Normalizes text for duplicate and leakage detection, not for training."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


@dataclass(frozen=True)
class DatasetAuditReport:
    """Counts and actionable problems found across train and validation data."""

    train_qa_examples: int
    validation_qa_examples: int
    train_text_documents: int
    validation_text_documents: int
    duplicate_train_qa_pairs: int
    conflicting_train_questions: int
    train_validation_question_overlap: int
    train_validation_qa_overlap: int
    train_validation_text_overlap: int
    forbidden_phrase_hits: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["ok"] = self.ok
        values["errors"] = list(self.errors)
        values["warnings"] = list(self.warnings)
        return values


def _qa(path: str | Path | None) -> list[tuple[str, str]]:
    return load_qa_text(path) if path is not None else []


def _texts(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    return [text for _source, text in load_text_documents(path)]


def audit_dataset(
    *,
    question_train_path: str | Path | None = None,
    question_validation_path: str | Path | None = None,
    text_train_path: str | Path | None = None,
    text_validation_path: str | Path | None = None,
    forbidden_phrases: tuple[str, ...] | list[str] = (),
) -> DatasetAuditReport:
    """Audits active sources and reports leakage, duplicates, and contamination.

    Forbidden phrases are checked only in assistant answers. A project can use
    them to reject foreign model identities such as an upstream assistant name.
    """
    if not all(isinstance(value, str) and value.strip() for value in forbidden_phrases):
        raise ValueError("forbidden_phrases must contain non-empty strings")
    train_qa = _qa(question_train_path)
    validation_qa = _qa(question_validation_path)
    train_texts = _texts(text_train_path)
    validation_texts = _texts(text_validation_path)

    normalized_train_pairs = [
        (normalize_training_text(question), normalize_training_text(answer))
        for question, answer in train_qa
    ]
    normalized_validation_pairs = [
        (normalize_training_text(question), normalize_training_text(answer))
        for question, answer in validation_qa
    ]
    train_pair_set = set(normalized_train_pairs)
    validation_pair_set = set(normalized_validation_pairs)
    train_questions = {question for question, _answer in normalized_train_pairs}
    validation_questions = {
        question for question, _answer in normalized_validation_pairs
    }
    answers_by_question: dict[str, set[str]] = defaultdict(set)
    for question, answer in normalized_train_pairs:
        answers_by_question[question].add(answer)

    train_text_hashes = {
        hashlib.sha256(normalize_training_text(text).encode("utf-8")).hexdigest()
        for text in train_texts
    }
    validation_text_hashes = {
        hashlib.sha256(normalize_training_text(text).encode("utf-8")).hexdigest()
        for text in validation_texts
    }
    forbidden = tuple(normalize_training_text(value) for value in forbidden_phrases)
    forbidden_hits = sum(
        any(phrase in normalize_training_text(answer) for phrase in forbidden)
        for _question, answer in train_qa
    )
    duplicate_pairs = len(normalized_train_pairs) - len(train_pair_set)
    conflicting_questions = sum(
        len(answers) > 1 for answers in answers_by_question.values()
    )
    question_overlap = len(train_questions & validation_questions)
    pair_overlap = len(train_pair_set & validation_pair_set)
    text_overlap = len(train_text_hashes & validation_text_hashes)

    errors: list[str] = []
    warnings: list[str] = []
    if question_overlap:
        errors.append(
            f"{question_overlap} normalized validation questions also occur in training"
        )
    if pair_overlap:
        errors.append(f"{pair_overlap} complete QA pairs leak into validation")
    if text_overlap:
        errors.append(f"{text_overlap} text documents leak into validation")
    if forbidden_hits:
        errors.append(
            f"{forbidden_hits} assistant answers contain a forbidden phrase"
        )
    if duplicate_pairs:
        warnings.append(f"{duplicate_pairs} duplicate QA pairs occur in training")
    if conflicting_questions:
        warnings.append(
            f"{conflicting_questions} normalized questions have multiple training answers"
        )
    if train_qa and len(train_pair_set) < 100:
        warnings.append("fewer than 100 unique QA pairs; generalization is unlikely")
    if train_texts and sum(len(text) for text in train_texts) < 100_000:
        warnings.append("text corpus contains fewer than 100,000 characters")

    return DatasetAuditReport(
        train_qa_examples=len(train_qa),
        validation_qa_examples=len(validation_qa),
        train_text_documents=len(train_texts),
        validation_text_documents=len(validation_texts),
        duplicate_train_qa_pairs=duplicate_pairs,
        conflicting_train_questions=conflicting_questions,
        train_validation_question_overlap=question_overlap,
        train_validation_qa_overlap=pair_overlap,
        train_validation_text_overlap=text_overlap,
        forbidden_phrase_hits=forbidden_hits,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def save_dataset_audit(report: DatasetAuditReport, path: str | Path) -> Path:
    """Atomically stores an audit report next to a pipeline or model."""
    if not isinstance(report, DatasetAuditReport):
        raise TypeError("report must be a DatasetAuditReport")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination
