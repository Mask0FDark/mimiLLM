#!/usr/bin/env python3
"""Create reproducible train and validation files from a local JSONL file."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_examples(path: Path) -> list[tuple[str, str]]:
    """Read and validate source pairs, including alternate questions."""
    examples: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            question, answer = item.get("question"), item.get("answer")
            variants = item.get("variants", [])
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Line {line_number}: question must be a non-empty string")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Line {line_number}: answer must be a non-empty string")
            if not isinstance(variants, list) or not all(isinstance(v, str) for v in variants):
                raise ValueError(f"Line {line_number}: variants must be an array of strings")
            for variant in [question, *variants]:
                examples.append((variant.strip(), answer.strip()))
    if len(examples) < 2:
        raise ValueError("At least two examples are required to split the dataset")
    return examples


def render(examples: list[tuple[str, str]]) -> str:
    """Render examples in the text format understood by mimiLLM."""
    return "\n\n".join(f"Вопрос: {q}\nОтвет: {a}" for q, a in examples) + "\n"


def build_dataset(source: Path, train: Path, validation: Path, seed: int, ratio: float) -> tuple[int, int]:
    """Shuffle data with a fixed seed and write both splits."""
    if not 0.05 <= ratio <= 0.5:
        raise ValueError("validation-ratio must be between 0.05 and 0.5")
    examples = load_examples(source)
    random.Random(seed).shuffle(examples)
    validation_count = max(1, round(len(examples) * ratio))
    validation_items = examples[:validation_count]
    train_items = examples[validation_count:]
    train.parent.mkdir(parents=True, exist_ok=True)
    validation.parent.mkdir(parents=True, exist_ok=True)
    train.write_text(render(train_items), encoding="utf-8", newline="\n")
    validation.write_text(render(validation_items), encoding="utf-8", newline="\n")
    return len(train_items), len(validation_items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mimiLLM question-answer dataset")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data" / "qa_seed.jsonl",
        help="source JSONL file with question, answer, and optional variants fields",
    )
    parser.add_argument(
        "--train", type=Path,
        default=ROOT / "data" / "question" / "train" / "questions.txt",
        help="destination for the training split",
    )
    parser.add_argument(
        "--validation", type=Path,
        default=ROOT / "data" / "question" / "validation" / "questions.txt",
        help="destination for the validation split",
    )
    parser.add_argument("--seed", type=int, default=42, help="reproducible shuffle seed (default: 42)")
    parser.add_argument(
        "--validation-ratio", type=float, default=0.15,
        help="fraction reserved for validation, from 0.05 to 0.5 (default: 0.15)",
    )
    args = parser.parse_args()
    train_count, validation_count = build_dataset(
        args.source, args.train, args.validation, args.seed, args.validation_ratio
    )
    print(f"Done: train={train_count}, validation={validation_count}, seed={args.seed}")


if __name__ == "__main__":
    main()
