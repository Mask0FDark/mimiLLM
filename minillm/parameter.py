"""Обучаемый параметр модели."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence

from .tensor import Tensor


class Parameter(Tensor):
    """Tensor, который по умолчанию накапливает градиент."""

    def __init__(self, data: Iterable[float] | array, shape: Sequence[int] | int | None = None) -> None:
        super().__init__(data, shape, requires_grad=True)

