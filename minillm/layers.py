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


class Embedding(Module):
    """Обучаемая таблица векторных представлений целых индексов."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, *,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("размеры Embedding должны быть положительными")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        source = rng or random
        self.weight = Parameter(
            (source.gauss(0.0, 0.02) for _ in range(num_embeddings * embedding_dim)),
            (num_embeddings, embedding_dim),
        )

    def forward(self, indices: list[int]) -> Tensor:
        return self.weight.embedding(indices)


class RMSNorm(Module):
    """Нормализует RMS последней оси и сохраняет её средний масштаб."""

    def __init__(self, dimension: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        if dimension <= 0 or epsilon <= 0.0:
            raise ValueError("dimension и epsilon RMSNorm должны быть положительными")
        self.dimension = dimension
        self.epsilon = epsilon
        self.weight = Parameter([1.0] * dimension, (dimension,))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.dimension:
            raise ValueError(
                f"RMSNorm ожидал последнюю ось {self.dimension}, получена {inputs.shape}"
            )
        inverse_rms = ((inputs * inputs).mean(axis=-1, keepdims=True) + self.epsilon).sqrt()
        return (inputs / inverse_rms) * self.weight


class FeedForward(Module):
    """Двухслойный MLP с ReLU внутри Transformer-блока."""

    def __init__(
        self, dimension: int, hidden_dimension: int, *, rng: random.Random | None = None,
    ) -> None:
        super().__init__()
        if hidden_dimension <= 0:
            raise ValueError("hidden_dimension должен быть положительным")
        self.input_projection = Linear(dimension, hidden_dimension, rng=rng)
        self.output_projection = Linear(hidden_dimension, dimension, rng=rng)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output_projection(self.input_projection(inputs).relu())
