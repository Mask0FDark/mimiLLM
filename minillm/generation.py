"""Авторегрессионная генерация token за token без словаря ответов."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .tensor import no_grad
from .transformer import DecoderTransformer
from .tokenizer import ByteTokenizer


def sample_token(
    logits: Sequence[float], *, temperature: float, top_k: int,
    rng: random.Random,
) -> int:
    """Выбирает greedy token либо делает temperature/top-k sampling."""
    if not logits:
        raise ValueError("logits не могут быть пустыми")
    if temperature < 0.0:
        raise ValueError("temperature не может быть отрицательной")
    if top_k < 0:
        raise ValueError("top_k не может быть отрицательным")
    if temperature == 0.0 or top_k == 1:
        return max(range(len(logits)), key=lambda index: logits[index])
    candidates = list(range(len(logits)))
    if top_k > 0 and top_k < len(candidates):
        candidates.sort(key=lambda index: logits[index], reverse=True)
        candidates = candidates[:top_k]
    maximum = max(logits[index] for index in candidates)
    weights = [math.exp((logits[index] - maximum) / temperature) for index in candidates]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for index, weight in zip(candidates, weights):
        cumulative += weight
        if cumulative >= threshold:
            return index
    return candidates[-1]


def generate(
    model: DecoderTransformer, prompt_tokens: Sequence[int], *,
    max_new_tokens: int = 80, temperature: float = 0.7, top_k: int = 20,
    eos_token: int = ByteTokenizer.EOS, rng: random.Random | None = None,
) -> list[int]:
    """Продолжает prompt, каждый раз подавая доступное контекстное окно."""
    if not prompt_tokens:
        raise ValueError("prompt_tokens не могут быть пустыми")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens не может быть отрицательным")
    source = rng or random.Random()
    tokens = list(prompt_tokens)
    generated: list[int] = []
    with no_grad():
        for _ in range(max_new_tokens):
            context = tokens[-model.config.context_length:]
            logits = model([context])
            start = (len(context) - 1) * model.config.vocab_size
            row = logits.data[start:start + model.config.vocab_size]
            token = sample_token(row, temperature=temperature, top_k=top_k, rng=source)
            tokens.append(token)
            if token == eos_token:
                break
            generated.append(token)
    return generated


def answer_question(
    model: DecoderTransformer, tokenizer: ByteTokenizer, question: str, **settings: object,
) -> str:
    """Формирует только prompt и декодирует реально сгенерированные токены."""
    prompt = tokenizer.encode_prompt(question)
    generated = generate(model, prompt, **settings)  # type: ignore[arg-type]
    return tokenizer.decode(generated).strip()
