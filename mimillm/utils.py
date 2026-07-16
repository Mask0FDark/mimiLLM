"""Общие небольшие утилиты CLI."""

from __future__ import annotations

import math
import random


def flatten(rows: list[list[int]]) -> list[int]:
    """Объединяет прямоугольный список целых строк."""
    return [value for row in rows for value in row]


def learning_rate_at(
    step: int,
    total_steps: int,
    base: float,
    warmup: int,
    *,
    schedule: str = "cosine",
    min_ratio: float = 0.1,
) -> float:
    """Linear warmup followed by constant, linear, or cosine decay."""
    if warmup > 0 and step <= warmup:
        return base * step / warmup
    normalized = schedule.strip().lower()
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be between 0 and 1")
    if normalized == "constant":
        return base
    if normalized not in {"linear", "cosine"}:
        raise ValueError("schedule must be 'constant', 'linear', or 'cosine'")
    remaining = max(total_steps - warmup, 1)
    progress = min(max((step - warmup) / remaining, 0.0), 1.0)
    multiplier = (
        1.0 - progress
        if normalized == "linear"
        else 0.5 * (1.0 + math.cos(math.pi * progress))
    )
    return base * (min_ratio + (1.0 - min_ratio) * multiplier)


def seeded_random(seed: int) -> random.Random:
    """Создаёт независимый воспроизводимый генератор."""
    return random.Random(seed)
