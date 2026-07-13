"""Многоголовое причинное self-attention."""

from __future__ import annotations

import math
import random

from .layers import Linear
from .module import Module
from .tensor import Tensor


class MultiHeadCausalSelfAttention(Module):
    """Позволяет каждой позиции смешивать только текущие и прошлые токены."""

    def __init__(
        self, dimension: int, num_heads: int, *, rng: random.Random | None = None,
    ) -> None:
        super().__init__()
        if dimension <= 0 or num_heads <= 0 or dimension % num_heads:
            raise ValueError("dimension должен быть положительным и делиться на num_heads")
        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dimension = dimension // num_heads
        self.query = Linear(dimension, dimension, rng=rng)
        self.key = Linear(dimension, dimension, rng=rng)
        self.value = Linear(dimension, dimension, rng=rng)
        self.output = Linear(dimension, dimension, rng=rng)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.dimension:
            raise ValueError(
                f"attention ожидает форму (batch, time, {self.dimension}), получена {inputs.shape}"
            )
        batch, time, _ = inputs.shape
        query = self.query(inputs).reshape(batch, time, self.num_heads, self.head_dimension)
        key = self.key(inputs).reshape(batch, time, self.num_heads, self.head_dimension)
        value = self.value(inputs).reshape(batch, time, self.num_heads, self.head_dimension)
        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 1, 3))
        value = value.permute((0, 2, 1, 3))
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        mask_values = [
            0.0 if column <= row else -1.0e9
            for row in range(time) for column in range(time)
        ]
        mask = Tensor(mask_values, (1, 1, time, time))
        weights = (scores + mask).softmax(axis=-1)
        context = weights.matmul(value).permute((0, 2, 1, 3))
        return self.output(context.reshape(batch, time, self.dimension))

