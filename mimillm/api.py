"""Короткие высокоуровневые функции публичного API mimiLLM."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint
from .safetensors import load_safetensors, save_safetensors
from .tokenizer import BpeTokenizer, ByteTokenizer, create_tokenizer, save_tokenizer
from .transformer import DecoderTransformer, TransformerConfig


ModelConfig = TransformerConfig
LanguageModel = DecoderTransformer


def create_model(
    config: TransformerConfig | None = None,
    *,
    tokenizer_model: ByteTokenizer | None = None,
    **options: Any,
) -> DecoderTransformer:
    """Создаёт языковую модель из config или именованных параметров.

    Передайте либо готовый ``TransformerConfig``, либо параметры вроде
    ``d_model=128`` и ``n_layers=4``. Смешивать два способа нельзя: это помогает
    сразу заметить неоднозначную конфигурацию.
    """
    if config is not None and options:
        raise ValueError("передайте config или именованные параметры, но не оба варианта")
    return DecoderTransformer(
        config or TransformerConfig(**options), tokenizer_model=tokenizer_model,
    )


def _model_files(path: str | Path) -> tuple[Path, Path]:
    requested = Path(path)
    if requested.suffix.lower() == ".safetensors":
        return requested.with_name("config.json"), requested
    return requested / "config.json", requested / "model.safetensors"


def _tokenizer_for_model(config: TransformerConfig, directory: Path) -> ByteTokenizer:
    if config.tokenizer.strip().lower() == "bpe":
        return create_tokenizer("bpe", path=directory / "tokenizer.json")
    return create_tokenizer(config.tokenizer)


def save_model(path: str | Path, model: DecoderTransformer) -> Path:
    """Saves a reusable model directory with config.json and model.safetensors."""
    config_path, weights_path = _model_files(path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        weights_path,
        model.state_dict(),
        metadata={"format": "mimiLLM", "format_version": "1"},
    )
    if isinstance(model.tokenizer, BpeTokenizer):
        save_tokenizer(model.tokenizer, weights_path.parent / "tokenizer.json")
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(model.config.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(config_path)
    return weights_path.parent


def load_model(path: str | Path, *, eval_mode: bool = True) -> DecoderTransformer:
    """Loads a model directory, a .safetensors file, or a legacy training checkpoint."""
    requested = Path(path)
    if requested.is_dir() or requested.suffix.lower() == ".safetensors":
        config_path, weights_path = _model_files(requested)
        config = TransformerConfig.from_json(config_path)
        parameters, _ = load_safetensors(weights_path)
        tokenizer_model = _tokenizer_for_model(config, config_path.parent)
    else:
        stored = load_checkpoint(requested)
        config = TransformerConfig.from_dict(stored.config)
        parameters = stored.parameters
        tokenizer_model = _tokenizer_for_model(config, requested.parent)
    model = DecoderTransformer(config, tokenizer_model=tokenizer_model)
    model.load_state_dict(parameters)
    return model.eval() if eval_mode else model
