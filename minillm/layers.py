"""Базовые нейросетевые слои m0fdii."""

from __future__ import annotations

import math
import random

from .module import Module
from .parameter import Parameter
from .tensor import Tensor


class Linear(Module):
    """Полносвязное преобразование последней оси: y = xW + b."""

    def __init__(
        self, in_features: int, out_features: int, *, bias: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("размеры Linear должны быть положительными")
        self.in_features = in_features
        self.out_features = out_features
        source = rng or random
        bound = math.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(
            (source.uniform(-bound, bound) for _ in range(in_features * out_features)),
            (in_features, out_features),
        )
        self.bias = Parameter([0.0] * out_features, (out_features,)) if bias else None

    def forward(self, inputs: Tensor) -> Tensor:
        """Сохраняет все ведущие размеры и меняет только последнюю ось."""
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear ожидал последнюю ось {self.in_features}, получена форма {inputs.shape}"
            )
        rows = inputs.numel // self.in_features
        output = inputs.reshape(rows, self.in_features).matmul(self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*inputs.shape[:-1], self.out_features)


class ReLU(Module):
    """Слой поэлементной активации ReLU."""

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.relu()

