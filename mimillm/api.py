"""Короткие высокоуровневые функции публичного API mimiLLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint
from .transformer import DecoderTransformer, TransformerConfig


ModelConfig = TransformerConfig
LanguageModel = DecoderTransformer


def create_model(
    config: TransformerConfig | None = None, **options: Any,
) -> DecoderTransformer:
    """Создаёт языковую модель из config или именованных параметров.

    Передайте либо готовый ``TransformerConfig``, либо параметры вроде
    ``d_model=128`` и ``n_layers=4``. Смешивать два способа нельзя: это помогает
    сразу заметить неоднозначную конфигурацию.
    """
    if config is not None and options:
        raise ValueError("передайте config или именованные параметры, но не оба варианта")
    return DecoderTransformer(config or TransformerConfig(**options))


def load_model(path: str | Path, *, eval_mode: bool = True) -> DecoderTransformer:
    """Создаёт модель нужного размера и загружает веса из checkpoint."""
    stored = load_checkpoint(path)
    model = DecoderTransformer(TransformerConfig.from_dict(stored.config))
    model.load_state_dict(stored.parameters)
    return model.eval() if eval_mode else model
