"""Авторегрессионная генерация token за token без словаря ответов."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .tensor import no_grad
from .transformer import DecoderTransformer
from .tokenizer import ByteTokenizer


def _valid_utf8_prefix(raw: bytes) -> bool:
    try:
        raw.decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError as exc:
        return exc.reason == "unexpected end of data" and exc.end == len(raw)


def _complete_utf8(raw: bytes) -> bool:
    try:
        raw.decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError:
        return False


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
    enforce_valid_utf8: bool = True,
) -> list[int]:
    """Continues a prompt and prevents invalid UTF-8 byte continuations."""
    if not prompt_tokens:
        raise ValueError("prompt_tokens не могут быть пустыми")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens не может быть отрицательным")
    source = rng or random.Random()
    if not isinstance(enforce_valid_utf8, bool):
        raise TypeError("enforce_valid_utf8 must be a boolean")
    tokenizer = getattr(model, "tokenizer", None) or ByteTokenizer()
    tokens = list(prompt_tokens)
    generated: list[int] = []
    generated_bytes = bytearray()
    with no_grad():
        for _ in range(max_new_tokens):
            context = tokens[-model.config.context_length:]
            logits = model([context])
            start = (len(context) - 1) * model.config.vocab_size
            row = list(logits.data[start:start + model.config.vocab_size])
            for special in (tokenizer.PAD, tokenizer.BOS, tokenizer.SEP):
                if special < len(row):
                    row[special] = -math.inf
            if enforce_valid_utf8:
                prefix = bytes(generated_bytes)
                for candidate in range(len(row)):
                    if candidate == eos_token:
                        if not _complete_utf8(prefix):
                            row[candidate] = -math.inf
                        continue
                    piece = tokenizer.token_bytes(candidate)
                    if piece is None or not _valid_utf8_prefix(prefix + piece):
                        row[candidate] = -math.inf
            token = sample_token(row, temperature=temperature, top_k=top_k, rng=source)
            tokens.append(token)
            if token == eos_token:
                break
            generated.append(token)
            piece = tokenizer.token_bytes(token)
            if piece is not None:
                generated_bytes.extend(piece)
    if enforce_valid_utf8:
        while generated and not _complete_utf8(bytes(generated_bytes)):
            generated.pop()
            generated_bytes = bytearray()
            for token in generated:
                piece = tokenizer.token_bytes(token)
                if piece is not None:
                    generated_bytes.extend(piece)
    return generated


def answer_question(
    model: DecoderTransformer, tokenizer: ByteTokenizer, question: str, **settings: object,
) -> str:
    """Формирует только prompt и декодирует реально сгенерированные токены."""
    prompt = tokenizer.encode_prompt(question)
    generated = generate(model, prompt, **settings)  # type: ignore[arg-type]
    return tokenizer.decode(generated).strip()


def generate_text(
    model: DecoderTransformer,
    prompt: str,
    *,
    tokenizer: ByteTokenizer | None = None,
    include_prompt: bool = False,
    **settings: object,
) -> str:
    """Продолжает обычный UTF-8 текст и возвращает строку, а не token ids."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt должен быть непустой строкой")
    codec = tokenizer or getattr(model, "tokenizer", None) or ByteTokenizer()
    prompt_tokens = codec.encode(prompt, add_bos=True)
    generated = generate(model, prompt_tokens, **settings)  # type: ignore[arg-type]
    continuation = codec.decode(generated)
    return prompt + continuation if include_prompt else continuation


def generate_response(
    model: DecoderTransformer,
    prompt: str,
    *,
    tokenizer: ByteTokenizer | None = None,
    **settings: object,
) -> str:
    """Отвечает на пользовательский запрос через единый prompt модели."""
    codec = tokenizer or getattr(model, "tokenizer", None) or ByteTokenizer()
    return answer_question(model, codec, prompt, **settings)
