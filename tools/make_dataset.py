#!/usr/bin/env python3
"""Создаёт воспроизводимые train/validation файлы из локального JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_examples(path: Path) -> list[tuple[str, str]]:
    """Читает и проверяет исходные пары, включая варианты вопросов."""
    examples: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некорректный JSON в строке {line_number}: {exc}") from exc
            question, answer = item.get("question"), item.get("answer")
            variants = item.get("variants", [])
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Строка {line_number}: question должна быть непустой строкой")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Строка {line_number}: answer должна быть непустой строкой")
            if not isinstance(variants, list) or not all(isinstance(v, str) for v in variants):
                raise ValueError(f"Строка {line_number}: variants должен быть массивом строк")
            for variant in [question, *variants]:
                examples.append((variant.strip(), answer.strip()))
    if len(examples) < 2:
        raise ValueError("Для разделения датасета нужно минимум два примера")
    return examples


def render(examples: list[tuple[str, str]]) -> str:
    """Представляет примеры простым читаемым текстовым форматом."""
    return "\n\n".join(f"Вопрос: {q}\nОтвет: {a}" for q, a in examples) + "\n"


def build_dataset(source: Path, train: Path, validation: Path, seed: int, ratio: float) -> tuple[int, int]:
    """Перемешивает данные с фиксированным seed и атомарно записывает части."""
    if not 0.05 <= ratio <= 0.5:
        raise ValueError("validation-ratio должен быть от 0.05 до 0.5")
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
    parser = argparse.ArgumentParser(description="Создание датасета mimiLLM")
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "qa_seed.jsonl")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.txt")
    parser.add_argument("--validation", type=Path, default=ROOT / "data" / "validation.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    args = parser.parse_args()
    train_count, validation_count = build_dataset(
        args.source, args.train, args.validation, args.seed, args.validation_ratio
    )
    print(f"Готово: train={train_count}, validation={validation_count}, seed={args.seed}")


if __name__ == "__main__":
    main()

