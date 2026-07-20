"""Конфигурация и decoder-only Transformer модели mimiLLM."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .attention import MultiHeadCausalSelfAttention
from .layers import Embedding, FeedForward, Linear, RMSNorm
from .module import Module
from .parameter import Parameter
from .tensor import Tensor
from .tokenizer import BpeTokenizer, ByteTokenizer, UnicodeByteTokenizer, create_tokenizer


@dataclass(frozen=True)
class TransformerConfig:
    """Проверяемые размеры модели и параметры обучения."""

    vocab_size: int = ByteTokenizer.VOCAB_SIZE
    tokenizer: str = "byte"
    tokenizer_path: str = "tokenizer.json"
    tie_word_embeddings: bool = True
    context_length: int = 96
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 192
    batch_size: int = 2
    batches_per_epoch: int | None = None
    steps: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0
    learning_rate_schedule: str = "cosine"
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 10
    validation_interval: int = 20
    checkpoint_interval: int = 50
    save_validation_checkpoints: bool = False
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    seed: int = 42
    text_ratio: float = 0.5
    qa_prompt_weight: float = 0.0
    qa_answer_prefix_weight: float = 1.0
    qa_answer_prefix_tokens: int = 0
    qa_source_weights: dict[str, float] | None = None
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
        if self.batches_per_epoch is not None and (
            not isinstance(self.batches_per_epoch, int)
            or isinstance(self.batches_per_epoch, bool)
            or self.batches_per_epoch <= 0
        ):
            raise ValueError("batches_per_epoch должен быть положительным целым числом или null")
        if not isinstance(self.tokenizer, str):
            raise TypeError("tokenizer must be a string")
        if not isinstance(self.tie_word_embeddings, bool):
            raise TypeError("tie_word_embeddings must be a boolean")
        tokenizer_sizes = {
            "byte": ByteTokenizer.VOCAB_SIZE,
            "unicode": UnicodeByteTokenizer.VOCAB_SIZE,
            "unicode_byte": UnicodeByteTokenizer.VOCAB_SIZE,
        }
        normalized_tokenizer = self.tokenizer.strip().lower()
        if normalized_tokenizer not in {*tokenizer_sizes, "bpe"}:
            raise ValueError("tokenizer must be 'byte', 'unicode', or 'bpe'")
        expected_vocab_size = tokenizer_sizes.get(normalized_tokenizer)
        if expected_vocab_size is not None and self.vocab_size != expected_vocab_size:
            raise ValueError(
                f"vocab_size must be {expected_vocab_size} for tokenizer={self.tokenizer!r}"
            )
        if normalized_tokenizer == "bpe" and self.vocab_size < ByteTokenizer.VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be at least {ByteTokenizer.VOCAB_SIZE} for tokenizer='bpe'"
            )
        if self.d_model % self.n_heads:
            raise ValueError("d_model должен делиться на n_heads")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate > 0, weight_decay >= 0")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("adam_beta1 and adam_beta2 must be in [0, 1)")
        if self.adam_epsilon <= 0.0:
            raise ValueError("adam_epsilon must be positive")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if not isinstance(self.learning_rate_schedule, str):
            raise TypeError("learning_rate_schedule must be a string")
        if self.learning_rate_schedule.strip().lower() not in {"constant", "linear", "cosine"}:
            raise ValueError(
                "learning_rate_schedule must be 'constant', 'linear', or 'cosine'"
            )
        if (
            not math.isfinite(self.min_learning_rate_ratio)
            or not 0.0 <= self.min_learning_rate_ratio <= 1.0
        ):
            raise ValueError("min_learning_rate_ratio must be between 0 and 1")
        if not 0.0 <= self.text_ratio <= 1.0:
            raise ValueError("text_ratio должен быть от 0 до 1")
        if (
            not isinstance(self.qa_prompt_weight, (int, float))
            or isinstance(self.qa_prompt_weight, bool)
            or not math.isfinite(self.qa_prompt_weight)
            or not 0.0 <= self.qa_prompt_weight <= 1.0
        ):
            raise ValueError("qa_prompt_weight must be between 0 and 1")
        if (
            not isinstance(self.qa_answer_prefix_weight, (int, float))
            or isinstance(self.qa_answer_prefix_weight, bool)
            or not math.isfinite(self.qa_answer_prefix_weight)
            or self.qa_answer_prefix_weight < 1.0
        ):
            raise ValueError("qa_answer_prefix_weight must be at least 1")
        if (
            not isinstance(self.qa_answer_prefix_tokens, int)
            or isinstance(self.qa_answer_prefix_tokens, bool)
            or self.qa_answer_prefix_tokens < 0
        ):
            raise ValueError("qa_answer_prefix_tokens must be a non-negative integer")
        if self.qa_source_weights is not None:
            if not isinstance(self.qa_source_weights, dict) or not self.qa_source_weights:
                raise ValueError("qa_source_weights must be a non-empty object or null")
            positive = False
            for key, weight in self.qa_source_weights.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("qa_source_weights keys must be non-empty strings")
                if (
                    not isinstance(weight, (int, float))
                    or isinstance(weight, bool)
                    or not math.isfinite(weight)
                    or weight < 0.0
                ):
                    raise ValueError(
                        f"qa_source_weights[{key!r}] must be finite and non-negative"
                    )
                positive = positive or weight > 0.0
            if not positive:
                raise ValueError("qa_source_weights needs at least one positive weight")
        if self.warmup_steps < 0 or self.validation_interval <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("интервалы должны быть положительными, warmup_steps >= 0")
        if not isinstance(self.save_validation_checkpoints, bool):
            raise TypeError("save_validation_checkpoints must be a boolean")
        if self.early_stopping_patience is not None and (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be a positive integer or null")
        if (
            not isinstance(self.early_stopping_min_delta, (int, float))
            or isinstance(self.early_stopping_min_delta, bool)
            or not math.isfinite(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0.0
        ):
            raise ValueError("early_stopping_min_delta must be finite and non-negative")
        if not isinstance(self.tokenizer_path, str) or not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path must be a non-empty string")
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
        values = dict(values)
        legacy_configuration = "tie_word_embeddings" not in values
        # Configurations written before weight tying existed contain a separate
        # output projection. Preserve their architecture when loading them.
        values.setdefault("tie_word_embeddings", False)
        # Older runs used linear decay and AdamW beta2=0.999. Keeping these
        # values makes an old JSON config reproducible. A modern JSON that
        # explicitly selects weight tying receives the current dataclass
        # defaults instead of silently falling back to the legacy optimizer.
        if legacy_configuration:
            values.setdefault("learning_rate_schedule", "linear")
            values.setdefault("adam_beta2", 0.999)
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

    def __init__(
        self,
        config: TransformerConfig,
        *,
        tokenizer_model: ByteTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        normalized_tokenizer = config.tokenizer.strip().lower()
        if tokenizer_model is None:
            if normalized_tokenizer == "bpe":
                raise ValueError(
                    "BPE models require tokenizer_model or tokenizer.json through load_model"
                )
            tokenizer_model = create_tokenizer(config.tokenizer)
        expected_type: type[ByteTokenizer]
        if normalized_tokenizer == "byte":
            expected_type = ByteTokenizer
            valid_type = type(tokenizer_model) is ByteTokenizer
        elif normalized_tokenizer in {"unicode", "unicode_byte"}:
            expected_type = UnicodeByteTokenizer
            valid_type = isinstance(tokenizer_model, UnicodeByteTokenizer)
        else:
            expected_type = BpeTokenizer
            valid_type = isinstance(tokenizer_model, BpeTokenizer)
        if not valid_type:
            raise ValueError(
                f"tokenizer_model must be {expected_type.__name__} for "
                f"tokenizer={config.tokenizer!r}"
            )
        if tokenizer_model.VOCAB_SIZE != config.vocab_size:
            raise ValueError(
                f"tokenizer vocabulary has {tokenizer_model.VOCAB_SIZE} tokens, "
                f"but config vocab_size is {config.vocab_size}"
            )
        self.tokenizer = tokenizer_model
        rng = random.Random(config.seed)
        self.token_embedding = Embedding(config.vocab_size, config.d_model, rng=rng)
        self.position_embedding = Embedding(config.context_length, config.d_model, rng=rng)
        self._block_names: list[str] = []
        for index in range(config.n_layers):
            name = f"block_{index}"
            setattr(self, name, TransformerBlock(config, rng=rng))
            self._block_names.append(name)
        self.final_norm = RMSNorm(config.d_model)
        if config.tie_word_embeddings:
            self.output = None
            self.output_bias = Parameter([0.0] * config.vocab_size, (config.vocab_size,))
        else:
            self.output = Linear(config.d_model, config.vocab_size, rng=rng)
            self.output_bias = None

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
        normalized = self.final_norm(hidden)
        if self.output is not None:
            return self.output(normalized)
        rows = batch * time
        logits = normalized.reshape(rows, self.config.d_model).matmul(
            self.token_embedding.weight.transpose(0, 1)
        )
        assert self.output_bias is not None
        return (logits + self.output_bias).reshape(batch, time, self.config.vocab_size)

    def parameter_count(self) -> int:
        """Число обучаемых float32-параметров."""
        return sum(parameter.numel for parameter in self.parameters())
