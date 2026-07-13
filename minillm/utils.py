"""Общие небольшие утилиты CLI."""

from __future__ import annotations

import random


def flatten(rows: list[list[int]]) -> list[int]:
    """Объединяет прямоугольный список целых строк."""
    return [value for row in rows for value in row]


def learning_rate_at(step: int, total_steps: int, base: float, warmup: int) -> float:
    """Линейный warmup, затем линейное уменьшение до 10% base."""
    if warmup > 0 and step <= warmup:
        return base * step / warmup
    remaining = max(total_steps - warmup, 1)
    progress = min(max((step - warmup) / remaining, 0.0), 1.0)
    return base * (1.0 - 0.9 * progress)


def seeded_random(seed: int) -> random.Random:
    """Создаёт независимый воспроизводимый генератор."""
    return random.Random(seed)

