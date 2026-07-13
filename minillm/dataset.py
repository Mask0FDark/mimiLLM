"""Загрузка текстовых QA-пар и создание next-token batch."""

from __future__ import annotations

import random
from pathlib import Path

from .tokenizer import ByteTokenizer


def load_qa_text(path: str | Path) -> list[tuple[str, str]]:
    """Читает блоки `Вопрос:`/`Ответ:` из локального UTF-8 файла."""
    text = Path(path).read_text(encoding="utf-8")
    result: list[tuple[str, str]] = []
    for block_number, block in enumerate(text.split("\n\n"), 1):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) < 2 or not lines[0].startswith("Вопрос: ") or not lines[1].startswith("Ответ: "):
            raise ValueError(f"блок {block_number}: ожидались строки Вопрос и Ответ")
        question = lines[0][len("Вопрос: "):].strip()
        answer = "\n".join([lines[1][len("Ответ: "):], *lines[2:]]).strip()
        if not question or not answer:
            raise ValueError(f"блок {block_number}: пустой вопрос или ответ")
        result.append((question, answer))
    if not result:
        raise ValueError(f"датасет пуст: {path}")
    return result


class TokenDataset:
    """Хранит единый поток token id и выбирает причинные окна."""

    def __init__(self, path: str | Path, tokenizer: ByteTokenizer | None = None) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer or ByteTokenizer()
        self.examples = load_qa_text(self.path)
        self.tokens: list[int] = []
        for question, answer in self.examples:
            self.tokens.extend(self.tokenizer.encode_qa(question, answer))
        if len(self.tokens) < 2:
            raise ValueError("в датасете недостаточно токенов")

    def sample_batch(
        self, batch_size: int, context_length: int, rng: random.Random,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Возвращает входы и сдвинутые на один токен цели."""
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size и context_length должны быть положительными")
        available = len(self.tokens) - context_length - 1
        if available < 0:
            raise ValueError("датасет короче context_length + 1")
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        for _ in range(batch_size):
            start = rng.randint(0, available)
            window = self.tokens[start:start + context_length + 1]
            inputs.append(window[:-1])
            targets.append(window[1:])
        return inputs, targets

    def deterministic_batch(
        self, batch_size: int, context_length: int, offset: int = 0,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Создаёт воспроизводимый validation batch без случайного состояния."""
        available = len(self.tokens) - context_length - 1
        if available < 0:
            raise ValueError("датасет короче context_length + 1")
        inputs, targets = [], []
        for index in range(batch_size):
            start = (offset + index * context_length) % (available + 1)
            window = self.tokens[start:start + context_length + 1]
            inputs.append(window[:-1])
            targets.append(window[1:])
        return inputs, targets

