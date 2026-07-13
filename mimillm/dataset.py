"""QA-пары, обычные UTF-8 тексты и next-token batch для языковой модели."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

from .tokenizer import ByteTokenizer


TEXT_SUFFIXES = {".txt", ".md", ".text"}


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


def discover_text_files(paths: Iterable[str | Path] | str | Path) -> list[Path]:
    """Находит поддерживаемые текстовые файлы, сохраняя стабильный порядок."""
    requested = [paths] if isinstance(paths, (str, Path)) else list(paths)
    discovered: list[Path] = []
    for item in requested:
        path = Path(item)
        if path.is_file():
            if path.suffix.lower() not in TEXT_SUFFIXES:
                raise ValueError(f"неподдерживаемое расширение текстового корпуса: {path}")
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(
                child for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES
            )
        else:
            raise FileNotFoundError(f"путь текстового корпуса не найден: {path}")
    # resolved нужен только для устранения повторов; наружу возвращаются удобные исходные пути.
    unique: dict[Path, Path] = {}
    for path in sorted(discovered, key=lambda value: value.as_posix().casefold()):
        unique.setdefault(path.resolve(), path)
    return list(unique.values())


def load_text_documents(paths: Iterable[str | Path] | str | Path) -> list[tuple[Path, str]]:
    """Читает непустые `.txt`, `.md` и `.text` документы строго как UTF-8."""
    files = discover_text_files(paths)
    if not files:
        raise ValueError("в текстовом корпусе нет файлов .txt, .md или .text")
    documents: list[tuple[Path, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append((path, text))
    if not documents:
        raise ValueError("текстовый корпус не содержит непустых документов")
    return documents


class TokenDataset:
    """Смешивает QA и обычные документы, затем выбирает причинные окна."""

    def __init__(
        self,
        path: str | Path | None = None,
        tokenizer: ByteTokenizer | None = None,
        *,
        text_paths: Iterable[str | Path] | str | Path | None = None,
        text_ratio: float = 0.0,
    ) -> None:
        if not 0.0 <= text_ratio <= 1.0:
            raise ValueError("text_ratio должен быть от 0 до 1")
        self.path = Path(path) if path is not None else None
        self.tokenizer = tokenizer or ByteTokenizer()
        self.text_ratio = float(text_ratio)
        self.examples = load_qa_text(self.path) if self.path is not None else []
        self.sequences = [
            self.tokenizer.encode_qa(question, answer)
            for question, answer in self.examples
        ]
        self.text_documents = load_text_documents(text_paths) if text_paths is not None else []
        self.text_sequences = [
            self.tokenizer.encode(text, add_bos=True, add_eos=True)
            for _, text in self.text_documents
        ]
        self.qa_tokens = sum(len(sequence) for sequence in self.sequences)
        self.text_tokens = sum(len(sequence) for sequence in self.text_sequences)
        self.tokens = [token for sequence in [*self.sequences, *self.text_sequences] for token in sequence]
        if len(self.tokens) < 2:
            raise ValueError("в датасете недостаточно токенов")
        if any(len(sequence) < 2 for sequence in [*self.sequences, *self.text_sequences]):
            raise ValueError("пример датасета слишком короткий")
        self.last_source = "qa" if self.sequences else "text"

    def source_weights(self) -> list[tuple[str, float]]:
        """Возвращает фактические вероятности источников с учётом их наличия."""
        if self.text_sequences and self.sequences:
            weights = [("qa", 1.0 - self.text_ratio), ("text", self.text_ratio)]
            return [(source, weight) for source, weight in weights if weight > 0.0]
        if self.text_sequences:
            return [("text", 1.0)]
        return [("qa", 1.0)]

    def _choose_source(self, rng: random.Random) -> str:
        weights = self.source_weights()
        if len(weights) == 1:
            return weights[0][0]
        return "text" if rng.random() < self.text_ratio else "qa"

    def _choose_sequences(
        self, source: str, batch_size: int, context_length: int, rng: random.Random,
    ) -> list[list[int]]:
        sequences = self.text_sequences if source == "text" else self.sequences
        if source == "text":
            # Большие документы содержат больше разных окон и должны встречаться чаще.
            weights = [max(1, len(sequence) - context_length) for sequence in sequences]
            return rng.choices(sequences, weights=weights, k=batch_size)
        return [rng.choice(sequences) for _ in range(batch_size)]

    def sample_batch(
        self, batch_size: int, context_length: int, rng: random.Random,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Возвращает входы и сдвинутые на один токен цели из одного источника."""
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size и context_length должны быть положительными")
        source = self._choose_source(rng)
        selected = self._choose_sequences(source, batch_size, context_length, rng)
        self.last_source = source
        window_size = min(context_length + 1, *(len(sequence) for sequence in selected))
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        for sequence in selected:
            maximum_start = len(sequence) - window_size
            # QA чаще начинается с вопроса; у длинного текста равномерно изучаются все участки.
            prefix_probability = 0.7 if source == "qa" else 0.1
            start = (
                0 if maximum_start == 0 or rng.random() < prefix_probability
                else rng.randint(1, maximum_start)
            )
            window = sequence[start:start + window_size]
            inputs.append(window[:-1])
            targets.append(window[1:])
        return inputs, targets

    def deterministic_batch(
        self, batch_size: int, context_length: int, offset: int = 0, *,
        source: str | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Создаёт воспроизводимый validation batch для указанного источника."""
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size и context_length должны быть положительными")
        selected_source = source or self.source_weights()[0][0]
        if selected_source not in {name for name, _ in self.source_weights()}:
            raise ValueError(f"источник {selected_source!r} отсутствует или имеет нулевой вес")
        sequences = self.text_sequences if selected_source == "text" else self.sequences
        selected = [sequences[(offset + index) % len(sequences)] for index in range(batch_size)]
        window_size = min(context_length + 1, *(len(sequence) for sequence in selected))
        inputs, targets = [], []
        for index, sequence in enumerate(selected):
            maximum_start = len(sequence) - window_size
            start = ((offset + index) * context_length) % (maximum_start + 1)
            window = sequence[start:start + window_size]
            inputs.append(window[:-1])
            targets.append(window[1:])
        return inputs, targets
