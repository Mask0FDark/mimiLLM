"""Конфигурация и decoder-only Transformer модели mimiLLM."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .attention import MultiHeadCausalSelfAttention
from .layers import Embedding, FeedForward, Linear, RMSNorm
from .module import Module
from .tensor import Tensor
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class TransformerConfig:
    """Проверяемые размеры модели и параметры обучения."""

    vocab_size: int = ByteTokenizer.VOCAB_SIZE
    context_length: int = 96
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 192
    batch_size: int = 2
    steps: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    validation_interval: int = 20
    checkpoint_interval: int = 50
    seed: int = 42
    text_ratio: float = 0.5
    text_train_path: str = "data/text/train"
    text_validation_path: str = "data/text/validation"
    question_train_path: str = "data/question/train"
    question_validation_path: str = "data/question/validation"

    def __post_init__(self) -> None:
        integer_positive = (
            "vocab_size", "context_length", "d_model", "n_layers", "n_heads",
            "d_mlp", "batch_size", "steps",
        )
        for name in integer_positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} должен быть положительным")
        if self.vocab_size != ByteTokenizer.VOCAB_SIZE:
            raise ValueError(f"vocab_size должен быть равен {ByteTokenizer.VOCAB_SIZE}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model должен делиться на n_heads")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate > 0, weight_decay >= 0")
        if not 0.0 <= self.text_ratio <= 1.0:
            raise ValueError("text_ratio должен быть от 0 до 1")
        if self.warmup_steps < 0 or self.validation_interval <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("интервалы должны быть положительными, warmup_steps >= 0")
        data_paths = (
            "text_train_path", "text_validation_path",
            "question_train_path", "question_validation_path",
        )
        for name in data_paths:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} должен быть непустой строкой")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TransformerConfig":
        """Отклоняет неизвестные ключи вместо молчаливой опечатки."""
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"неизвестные ключи конфигурации: {unknown}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "TransformerConfig":
        with Path(path).open("r", encoding="utf-8") as source:
            values = json.load(source)
        if not isinstance(values, dict):
            raise ValueError("корень конфигурации должен быть JSON-объектом")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TransformerBlock(Module):
    """Pre-norm блок: residual attention, затем residual MLP."""

    def __init__(self, config: TransformerConfig, *, rng: random.Random | None = None) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = MultiHeadCausalSelfAttention(config.d_model, config.n_heads, rng=rng)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = FeedForward(config.d_model, config.d_mlp, rng=rng)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.mlp(self.mlp_norm(hidden))


class DecoderTransformer(Module):
    """Decoder-only Transformer с обучаемыми позиционными embedding."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        rng = random.Random(config.seed)
        self.token_embedding = Embedding(config.vocab_size, config.d_model, rng=rng)
        self.position_embedding = Embedding(config.context_length, config.d_model, rng=rng)
        self._block_names: list[str] = []
        for index in range(config.n_layers):
            name = f"block_{index}"
            setattr(self, name, TransformerBlock(config, rng=rng))
            self._block_names.append(name)
        self.final_norm = RMSNorm(config.d_model)
        self.output = Linear(config.d_model, config.vocab_size, rng=rng)

    @property
    def blocks(self) -> list[TransformerBlock]:
        """Возвращает зарегистрированные блоки в порядке вычисления."""
        return [getattr(self, name) for name in self._block_names]

    def forward(self, token_ids: list[list[int]] | list[int]) -> Tensor:
        """Возвращает logits формы (batch, time, vocab_size)."""
        if not token_ids:
            raise ValueError("token_ids не может быть пустым")
        if isinstance(token_ids[0], int):
            batches = [token_ids]  # type: ignore[list-item]
        else:
            batches = token_ids  # type: ignore[assignment]
        batch = len(batches)
        time = len(batches[0])
        if time == 0 or any(len(row) != time for row in batches):
            raise ValueError("все последовательности batch должны иметь одинаковую ненулевую длину")
        if time > self.config.context_length:
            raise ValueError(f"длина {time} превышает контекст {self.config.context_length}")
        flat = [token for row in batches for token in row]
        token_vectors = self.token_embedding(flat).reshape(batch, time, self.config.d_model)
        positions = list(range(time)) * batch
        position_vectors = self.position_embedding(positions).reshape(batch, time, self.config.d_model)
        hidden = token_vectors + position_vectors
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))

    def parameter_count(self) -> int:
        """Число обучаемых float32-параметров."""
        return sum(parameter.numel for parameter in self.parameters())
