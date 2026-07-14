"""QA-пары, обычные UTF-8 тексты и next-token batch для языковой модели."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from pathlib import Path

from .tokenizer import ByteTokenizer


TEXT_SUFFIXES = {".txt", ".md", ".text"}


def discover_question_files(path: str | Path) -> list[Path]:
    """Находит UTF-8 `.txt` файлы с вопросами в файле или каталоге."""
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() != ".txt":
            raise ValueError(f"файл вопросов должен иметь расширение .txt: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"путь набора вопросов не найден: {source}")
    files = sorted(
        (item for item in source.rglob("*.txt") if item.is_file()),
        key=lambda item: item.as_posix().casefold(),
    )
    if not files:
        raise ValueError(f"в каталоге вопросов нет файлов .txt: {source}")
    return files


def load_qa_text(path: str | Path) -> list[tuple[str, str]]:
    """Читает блоки `Вопрос:`/`Ответ:` из UTF-8 файла или каталога."""
    result: list[tuple[str, str]] = []
    for file_path in discover_question_files(path):
        text = file_path.read_text(encoding="utf-8")
        for block_number, block in enumerate(text.split("\n\n"), 1):
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            if len(lines) < 2 or not lines[0].startswith("Вопрос: ") or not lines[1].startswith("Ответ: "):
                raise ValueError(
                    f"{file_path}, блок {block_number}: ожидались строки Вопрос и Ответ"
                )
            question = lines[0][len("Вопрос: "):].strip()
            answer = "\n".join([lines[1][len("Ответ: "):], *lines[2:]]).strip()
            if not question or not answer:
                raise ValueError(f"{file_path}, блок {block_number}: пустой вопрос или ответ")
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
        qa_prompt_weight: float = 0.0,
        qa_answer_prefix_weight: float = 1.0,
        qa_answer_prefix_tokens: int = 0,
    ) -> None:
        if not 0.0 <= text_ratio <= 1.0:
            raise ValueError("text_ratio должен быть от 0 до 1")
        if (
            not isinstance(qa_prompt_weight, (int, float))
            or isinstance(qa_prompt_weight, bool)
            or not math.isfinite(qa_prompt_weight)
            or not 0.0 <= qa_prompt_weight <= 1.0
        ):
            raise ValueError("qa_prompt_weight must be between 0 and 1")
        if (
            not isinstance(qa_answer_prefix_weight, (int, float))
            or isinstance(qa_answer_prefix_weight, bool)
            or not math.isfinite(qa_answer_prefix_weight)
            or qa_answer_prefix_weight < 1.0
        ):
            raise ValueError("qa_answer_prefix_weight must be at least 1")
        if (
            not isinstance(qa_answer_prefix_tokens, int)
            or isinstance(qa_answer_prefix_tokens, bool)
            or qa_answer_prefix_tokens < 0
        ):
            raise ValueError("qa_answer_prefix_tokens must be a non-negative integer")
        self.path = Path(path) if path is not None else None
        self.tokenizer = tokenizer or ByteTokenizer()
        self.text_ratio = float(text_ratio)
        self.qa_prompt_weight = float(qa_prompt_weight)
        self.qa_answer_prefix_weight = float(qa_answer_prefix_weight)
        self.qa_answer_prefix_tokens = qa_answer_prefix_tokens
        self.examples = load_qa_text(self.path) if self.path is not None else []
        self.sequences = [
            self.tokenizer.encode_qa(question, answer)
            for question, answer in self.examples
        ]
        self.qa_answer_starts = {
            id(sequence): len(self.tokenizer.encode_prompt(question))
            for sequence, (question, _) in zip(self.sequences, self.examples)
        }
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

    def sample_batch_with_loss_weights(
        self, batch_size: int, context_length: int, rng: random.Random,
    ) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
        """Создаёт padded batch и обучает QA только на токенах ответа.

        Токены вопроса остаются во входном контексте, но не входят в loss.
        У обычного текста вес имеют все настоящие цели. Padding всегда имеет
        нулевой вес, поэтому примеры разной длины не обрезаются до кратчайшего.
        """
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size и context_length должны быть положительными")
        source = self._choose_source(rng)
        selected = self._choose_sequences(source, batch_size, context_length, rng)
        self.last_source = source
        return self._batch_with_loss_weights(
            selected, source, context_length, rng=rng,
        )

    def _batch_with_loss_weights(
        self,
        selected: list[list[int]],
        source: str,
        context_length: int,
        *,
        rng: random.Random | None = None,
        offset: int = 0,
    ) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
        window_size = min(context_length + 1, max(len(sequence) for sequence in selected))
        row_size = window_size - 1
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        loss_weights: list[list[float]] = []
        for index, sequence in enumerate(selected):
            actual_size = min(window_size, len(sequence))
            maximum_start = len(sequence) - actual_size
            if source == "qa":
                answer_start = self.qa_answer_starts[id(sequence)]
                earliest_answer_window = max(0, answer_start - actual_size + 1)
                latest_context_window = min(maximum_start, max(0, answer_start - 1))
                if rng is None:
                    start = earliest_answer_window
                elif maximum_start == 0:
                    start = 0
                elif answer_start < actual_size and rng.random() < 0.7:
                    start = 0
                else:
                    start = rng.randint(earliest_answer_window, latest_context_window)
            else:
                if rng is None:
                    start = ((offset + index) * context_length) % (maximum_start + 1)
                else:
                    start = 0 if maximum_start == 0 else rng.randint(0, maximum_start)
                answer_start = 0
            window = sequence[start:start + actual_size]
            row_inputs = window[:-1]
            row_targets = window[1:]
            if source == "qa":
                row_weights = [
                    self._qa_loss_weight(absolute_position, answer_start)
                    for absolute_position in range(start + 1, start + actual_size)
                ]
            else:
                row_weights = [1.0] * len(row_targets)
            padding = row_size - len(row_inputs)
            if padding:
                row_inputs.extend([self.tokenizer.PAD] * padding)
                row_targets.extend([self.tokenizer.PAD] * padding)
                row_weights.extend([0.0] * padding)
            inputs.append(row_inputs)
            targets.append(row_targets)
            loss_weights.append(row_weights)
        return inputs, targets, loss_weights

    def _qa_loss_weight(self, target_position: int, answer_start: int) -> float:
        """Returns the configured weight for one prompt or answer target."""
        if target_position < answer_start:
            return self.qa_prompt_weight
        answer_offset = target_position - answer_start
        if answer_offset < self.qa_answer_prefix_tokens:
            return self.qa_answer_prefix_weight
        return 1.0

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

    def deterministic_batch_with_loss_weights(
        self, batch_size: int, context_length: int, offset: int = 0, *,
        source: str | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
        """Воспроизводимый weighted batch для validation high-level обучения."""
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size и context_length должны быть положительными")
        selected_source = source or self.source_weights()[0][0]
        if selected_source not in {name for name, _ in self.source_weights()}:
            raise ValueError(f"источник {selected_source!r} отсутствует или имеет нулевой вес")
        sequences = self.text_sequences if selected_source == "text" else self.sequences
        selected = [sequences[(offset + index) % len(sequences)] for index in range(batch_size)]
        return self._batch_with_loss_weights(
            selected, selected_source, context_length, offset=offset,
        )

    def validation_batches(
        self, batch_size: int, context_length: int, *, source: str,
    ) -> Iterable[tuple[list[list[int]], list[list[int]], list[list[float]]]]:
        """Последовательно покрывает все supervised-токены validation-источника."""
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size и context_length должны быть положительными")
        if source not in {name for name, _ in self.source_weights()}:
            raise ValueError(f"источник {source!r} отсутствует или имеет нулевой вес")
        sequences = self.text_sequences if source == "text" else self.sequences
        pending: list[tuple[list[int], list[int], list[float]]] = []
        for sequence in sequences:
            answer_start = 0 if source == "text" else self.qa_answer_starts[id(sequence)]
            first_target = (
                1 if source == "text" or self.qa_prompt_weight > 0.0
                else answer_start
            )
            target_position = first_target
            while target_position < len(sequence):
                # Оставляем до половины окна как историю и покрываем вторую
                # половину новыми validation-целями. Если весь prompt помещается,
                # первое QA-окно по-прежнему начинается с BOS.
                start = max(0, target_position - context_length // 2)
                end = min(len(sequence), start + context_length + 1)
                window = sequence[start:end]
                row_inputs = window[:-1]
                row_targets = window[1:]
                row_weights = [
                    (
                        1.0 if source == "text"
                        else self._qa_loss_weight(position, answer_start)
                    )
                    if position >= target_position else 0.0
                    for position in range(start + 1, end)
                ]
                pending.append((row_inputs, row_targets, row_weights))
                if len(pending) == batch_size:
                    yield self._pad_loss_rows(pending)
                    pending = []
                target_position = end
        if pending:
            yield self._pad_loss_rows(pending)

    def _pad_loss_rows(
        self, rows: list[tuple[list[int], list[int], list[float]]],
    ) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
        width = max(len(inputs) for inputs, _, _ in rows)
        batch_inputs: list[list[int]] = []
        batch_targets: list[list[int]] = []
        batch_weights: list[list[float]] = []
        for inputs, targets, weights in rows:
            padding = width - len(inputs)
            batch_inputs.append([*inputs, *([self.tokenizer.PAD] * padding)])
            batch_targets.append([*targets, *([self.tokenizer.PAD] * padding)])
            batch_weights.append([*weights, *([0.0] * padding)])
        return batch_inputs, batch_targets, batch_weights
