"""Функции потерь."""

from __future__ import annotations

from collections.abc import Sequence

from .tensor import Tensor


def cross_entropy(logits: Tensor, targets: Sequence[int]) -> Tensor:
    """Возвращает среднюю отрицательную log-вероятность правильного класса."""
    return logits.cross_entropy(targets)

