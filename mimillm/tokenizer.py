"""Reversible byte, Unicode-aware, and trainable BPE tokenizers."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_LEGACY_BPE_CHUNKS = re.compile(r"\s+|[^\s]+")
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class TokenizerReport:
    """Measured tokenizer quality on a concrete corpus.

    ``compression_ratio`` compares produced tokens with the UTF-8 byte count;
    lower is better. ``unicode_atomic_coverage`` is the frequency-weighted share
    of non-ASCII characters represented by one token instead of a partial byte
    sequence. The report deliberately measures a corpus rather than claiming
    that a vocabulary is good only because it can round-trip arbitrary text.
    """

    documents: int
    characters: int
    words: int
    utf8_bytes: int
    tokens: int
    unique_tokens: int
    vocab_size: int
    raw_byte_tokens: int
    non_ascii_byte_tokens: int
    roundtrip_errors: int
    replacement_characters: int
    compression_ratio: float
    tokens_per_word: float
    vocab_utilization: float
    unicode_atomic_coverage: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable representation."""
        values = asdict(self)
        values["warnings"] = list(self.warnings)
        return values


def _unicode_group(character: str) -> str:
    """Classify a non-whitespace character for the Unicode BPE pre-tokenizer."""
    category = unicodedata.category(character)
    if character.isalpha() or category.startswith("M") or character == "_":
        return "word"
    if character.isdecimal():
        return "number"
    return "symbol"


def _unicode_word_chunks(text: str) -> Iterable[str]:
    """Yield lossless word-like chunks and attach horizontal space to the next chunk."""
    pending_space = ""
    index = 0
    while index < len(text):
        if text[index].isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            whitespace = text[index:end]
            if "\n" in whitespace or "\r" in whitespace:
                if pending_space:
                    yield pending_space
                    pending_space = ""
                yield whitespace
            else:
                pending_space += whitespace
            index = end
            continue

        group = _unicode_group(text[index])
        end = index + 1
        while (
            end < len(text)
            and not text[end].isspace()
            and _unicode_group(text[end]) == group
        ):
            end += 1
        yield pending_space + text[index:end]
        pending_space = ""
        index = end
    if pending_space:
        yield pending_space


def pretokenize(text: str, *, mode: str = "unicode_words_v1") -> list[str]:
    """Split text into reversible chunks used independently by BPE merges."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if mode == "legacy_whitespace":
        return [match.group(0) for match in _LEGACY_BPE_CHUNKS.finditer(text)]
    if mode == "unicode_words_v1":
        return list(_unicode_word_chunks(text))
    raise ValueError("BPE pretokenizer must be 'unicode_words_v1' or 'legacy_whitespace'")


def format_qa_text(question: str, answer: str | None = None) -> str:
    """Build the canonical QA text shared by training and inference.

    ``answer=None`` returns the exact generation prompt. A completed training
    example adds one separating space and the stripped assistant answer. Keeping
    this formatting in one function prevents a silent train/inference mismatch.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    prompt = f"Вопрос: {question.strip()}\nОтвет:"
    if answer is None:
        return prompt
    if not isinstance(answer, str):
        raise TypeError("answer must be a string or None")
    return f"{prompt} {answer.strip()}"


class ByteTokenizer:
    """Кодирует текст байтами UTF-8 и четырьмя специальными токенами."""

    PAD = 256
    BOS = 257
    EOS = 258
    SEP = 259
    VOCAB_SIZE = 260

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Преобразует строку в список байтовых токенов."""
        if not isinstance(text, str):
            raise TypeError("text должен быть строкой")
        tokens = list(text.encode("utf-8"))
        if add_bos:
            tokens.insert(0, self.BOS)
        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: Iterable[int], *, skip_special: bool = True) -> str:
        """Декодирует токены; повреждённый UTF-8 заменяет символом U+FFFD."""
        raw = bytearray()
        for token in tokens:
            if not isinstance(token, int):
                raise TypeError("каждый токен должен быть целым числом")
            if 0 <= token <= 255:
                raw.append(token)
            elif token in (self.PAD, self.BOS, self.EOS, self.SEP):
                if not skip_special:
                    names = {
                        self.PAD: "<PAD>", self.BOS: "<BOS>",
                        self.EOS: "<EOS>", self.SEP: "<SEP>",
                    }
                    raw.extend(names[token].encode("ascii"))
            else:
                raise ValueError(f"токен вне диапазона словаря: {token}")
        return raw.decode("utf-8", errors="replace")

    def token_bytes(self, token: int) -> bytes | None:
        """Returns serialized text bytes, or ``None`` for a special token."""
        if not isinstance(token, int) or isinstance(token, bool):
            raise TypeError("token must be an integer")
        if 0 <= token <= 255:
            return bytes([token])
        if token in (self.PAD, self.BOS, self.EOS, self.SEP):
            return None
        raise ValueError(f"token is outside the vocabulary: {token}")

    def encode_qa(self, question: str, answer: str) -> list[int]:
        """Кодирует законченную обучающую пару вопрос–ответ."""
        return [self.BOS, *self.encode(format_qa_text(question, answer)), self.EOS]

    def encode_prompt(self, question: str) -> list[int]:
        """Кодирует начало ответа для авторегрессионной генерации."""
        return [self.BOS, *self.encode(format_qa_text(question))]


class UnicodeByteTokenizer(ByteTokenizer):
    """Uses one token for common Cyrillic characters and bytes as a fallback.

    ASCII and arbitrary UTF-8 remain fully reversible through the original byte
    vocabulary. Common Cyrillic letters and punctuation receive dedicated token
    ids, which prevents every Russian letter from being split into two tokens.
    The vocabulary is fixed so a saved model does not need a separate tokenizer
    file and remains reproducible on another computer.
    """

    EXTRA_CHARACTERS = (
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "ІіЇїЄєҐґЎў"
        "—–…«»„“”’×÷²³°≈≤≥→←"
    )
    if len(set(EXTRA_CHARACTERS)) != len(EXTRA_CHARACTERS):
        raise RuntimeError("UnicodeByteTokenizer.EXTRA_CHARACTERS contains duplicates")
    _CHAR_TO_TOKEN = {
        character: ByteTokenizer.VOCAB_SIZE + index
        for index, character in enumerate(EXTRA_CHARACTERS)
    }
    _TOKEN_TO_CHAR = {token: character for character, token in _CHAR_TO_TOKEN.items()}
    VOCAB_SIZE = ByteTokenizer.VOCAB_SIZE + len(EXTRA_CHARACTERS)

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False,
    ) -> list[int]:
        """Encodes common Cyrillic as characters and everything else as UTF-8."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.BOS)
        for character in text:
            token = self._CHAR_TO_TOKEN.get(character)
            if token is None:
                tokens.extend(character.encode("utf-8"))
            else:
                tokens.append(token)
        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: Iterable[int], *, skip_special: bool = True) -> str:
        """Decodes mixed character tokens and fallback UTF-8 byte sequences."""
        parts: list[str] = []
        raw = bytearray()

        def flush_bytes() -> None:
            if raw:
                parts.append(raw.decode("utf-8", errors="replace"))
                raw.clear()

        special_names = {
            self.PAD: "<PAD>", self.BOS: "<BOS>",
            self.EOS: "<EOS>", self.SEP: "<SEP>",
        }
        for token in tokens:
            if not isinstance(token, int):
                raise TypeError("every token must be an integer")
            if 0 <= token <= 255:
                raw.append(token)
            elif token in special_names:
                flush_bytes()
                if not skip_special:
                    parts.append(special_names[token])
            elif token in self._TOKEN_TO_CHAR:
                flush_bytes()
                parts.append(self._TOKEN_TO_CHAR[token])
            else:
                raise ValueError(f"token is outside the vocabulary: {token}")
        flush_bytes()
        return "".join(parts)

    def token_bytes(self, token: int) -> bytes | None:
        if token in self._TOKEN_TO_CHAR:
            return self._TOKEN_TO_CHAR[token].encode("utf-8")
        return super().token_bytes(token)


class BpeTokenizer(ByteTokenizer):
    """Dependency-free byte-level BPE with a lossless UTF-8 fallback.

    Token ids 0..255 remain raw bytes and 256..259 keep the same special
    tokens as :class:`ByteTokenizer`. Every learned merge receives the next
    stable id starting at 260. Consequently arbitrary text is always
    representable even when it was absent from the tokenizer corpus.
    """

    FORMAT_VERSION = 3
    UNICODE_PRETOKENIZER_VERSION = 2
    LEGACY_FORMAT_VERSION = 1
    DEFAULT_PRETOKENIZER = "unicode_words_v1"
    LEGACY_PRETOKENIZER = "legacy_whitespace"

    def __init__(
        self,
        merges: Iterable[tuple[int, int] | list[int]],
        *,
        pretokenizer: str = DEFAULT_PRETOKENIZER,
        format_version: int = FORMAT_VERSION,
        unicode_character_merges: bool = False,
        required_pieces: Iterable[str] = (),
    ) -> None:
        if format_version not in (
            self.LEGACY_FORMAT_VERSION,
            self.UNICODE_PRETOKENIZER_VERSION,
            self.FORMAT_VERSION,
        ):
            raise ValueError(f"unsupported BPE tokenizer version: {format_version!r}")
        if pretokenizer not in (self.DEFAULT_PRETOKENIZER, self.LEGACY_PRETOKENIZER):
            raise ValueError(
                "BPE pretokenizer must be 'unicode_words_v1' or 'legacy_whitespace'"
            )
        if (
            format_version == self.LEGACY_FORMAT_VERSION
            and pretokenizer != self.LEGACY_PRETOKENIZER
        ):
            raise ValueError("BPE tokenizer version 1 requires legacy_whitespace")
        if not isinstance(unicode_character_merges, bool):
            raise TypeError("unicode_character_merges must be a boolean")
        if format_version < self.FORMAT_VERSION and unicode_character_merges:
            raise ValueError(
                "unicode_character_merges metadata requires BPE tokenizer version 3"
            )
        normalized: list[tuple[int, int]] = []
        pieces: dict[int, bytes] = {value: bytes([value]) for value in range(256)}
        seen: set[tuple[int, int]] = set()
        for index, raw_pair in enumerate(merges):
            if (
                not isinstance(raw_pair, (tuple, list))
                or len(raw_pair) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in raw_pair)
            ):
                raise ValueError(f"BPE merge {index} must contain two integer token ids")
            pair = (raw_pair[0], raw_pair[1])
            token_id = ByteTokenizer.VOCAB_SIZE + index
            for value in pair:
                if value in (self.PAD, self.BOS, self.EOS, self.SEP):
                    raise ValueError("BPE merges cannot contain special tokens")
                if value not in pieces or value >= token_id:
                    raise ValueError(
                        f"BPE merge {index} references unavailable token {value}"
                    )
            if pair in seen:
                raise ValueError(f"duplicate BPE merge pair: {pair}")
            seen.add(pair)
            normalized.append(pair)
            pieces[token_id] = pieces[pair[0]] + pieces[pair[1]]

        self.format_version = format_version
        self.pretokenizer = pretokenizer
        self.unicode_character_merges = unicode_character_merges
        self.merges = tuple(normalized)
        self.VOCAB_SIZE = ByteTokenizer.VOCAB_SIZE + len(self.merges)
        self._merge_tokens = {
            pair: ByteTokenizer.VOCAB_SIZE + index
            for index, pair in enumerate(self.merges)
        }
        self._merge_ranks = {pair: index for index, pair in enumerate(self.merges)}
        self._pieces = pieces
        if isinstance(required_pieces, (str, bytes)):
            raise TypeError("BPE required_pieces must be an iterable of strings")
        normalized_required: list[str] = []
        for piece in required_pieces:
            if not isinstance(piece, str) or not piece:
                raise ValueError("BPE required pieces must be non-empty strings")
            if piece in normalized_required:
                raise ValueError(f"duplicate BPE required piece: {piece!r}")
            if any(character.isspace() for character in piece):
                raise ValueError(
                    "BPE required pieces cannot contain whitespace: "
                    f"{piece!r}"
                )
            encoded_piece = self._encode_piece(piece.encode("utf-8"))
            if len(encoded_piece) != 1:
                raise ValueError(
                    f"BPE required piece is not represented atomically: {piece!r}"
                )
            normalized_required.append(piece)
        if normalized_required and format_version < self.FORMAT_VERSION:
            raise ValueError(
                "BPE required_pieces metadata requires tokenizer version 3"
            )
        self.required_pieces = tuple(normalized_required)
        self._required_tokens = {
            piece: self._encode_piece(piece.encode("utf-8"))[0]
            for piece in self.required_pieces
        }
        self._required_by_length = tuple(
            sorted(self.required_pieces, key=lambda piece: (-len(piece), piece))
        )

    @staticmethod
    def _replace_pair(
        sequence: tuple[int, ...] | list[int], pair: tuple[int, int], token_id: int,
    ) -> tuple[int, ...]:
        result: list[int] = []
        index = 0
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                result.append(token_id)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return tuple(result)

    def _encode_piece(self, raw: bytes) -> list[int]:
        sequence = tuple(raw)
        while len(sequence) > 1:
            candidates = {
                pair
                for pair in zip(sequence, sequence[1:])
                if pair in self._merge_ranks
            }
            if not candidates:
                break
            pair = min(candidates, key=self._merge_ranks.__getitem__)
            sequence = self._replace_pair(sequence, pair, self._merge_tokens[pair])
        return list(sequence)

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False,
    ) -> list[int]:
        """Encodes text with learned merges and raw bytes as the fallback."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.BOS)
        if not self.required_pieces:
            for chunk in pretokenize(text, mode=self.pretokenizer):
                tokens.extend(self._encode_piece(chunk.encode("utf-8")))
        else:
            regular_start = 0
            index = 0
            while index < len(text):
                matched = next(
                    (
                        piece for piece in self._required_by_length
                        if text.startswith(piece, index)
                    ),
                    None,
                )
                if matched is None:
                    index += 1
                    continue
                regular = text[regular_start:index]
                for chunk in pretokenize(regular, mode=self.pretokenizer):
                    tokens.extend(self._encode_piece(chunk.encode("utf-8")))
                tokens.append(self._required_tokens[matched])
                index += len(matched)
                regular_start = index
            regular = text[regular_start:]
            for chunk in pretokenize(regular, mode=self.pretokenizer):
                tokens.extend(self._encode_piece(chunk.encode("utf-8")))
        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: Iterable[int], *, skip_special: bool = True) -> str:
        """Decodes both learned pieces and byte fallback tokens."""
        raw = bytearray()
        special_names = {
            self.PAD: b"<PAD>", self.BOS: b"<BOS>",
            self.EOS: b"<EOS>", self.SEP: b"<SEP>",
        }
        for token in tokens:
            if not isinstance(token, int) or isinstance(token, bool):
                raise TypeError("every token must be an integer")
            if 0 <= token <= 255:
                raw.append(token)
            elif token in special_names:
                if not skip_special:
                    raw.extend(special_names[token])
            elif token in self._pieces:
                raw.extend(self._pieces[token])
            else:
                raise ValueError(f"token is outside the BPE vocabulary: {token}")
        return raw.decode("utf-8", errors="replace")

    def token_bytes(self, token: int) -> bytes | None:
        if not isinstance(token, int) or isinstance(token, bool):
            raise TypeError("token must be an integer")
        if token in self._pieces:
            return self._pieces[token]
        return super().token_bytes(token)

    def to_dict(self) -> dict[str, Any]:
        """Returns the portable tokenizer.json representation."""
        values: dict[str, Any] = {
            "type": "bpe",
            "version": self.format_version,
            "vocab_size": self.VOCAB_SIZE,
            "byte_fallback": True,
            "special_tokens": {
                "pad": self.PAD,
                "bos": self.BOS,
                "eos": self.EOS,
                "sep": self.SEP,
            },
            "merges": [list(pair) for pair in self.merges],
        }
        if self.format_version >= self.FORMAT_VERSION:
            values["pretokenizer"] = self.pretokenizer
            values["unicode_character_merges"] = self.unicode_character_merges
            if self.required_pieces:
                values["required_pieces"] = list(self.required_pieces)
        elif self.format_version >= self.UNICODE_PRETOKENIZER_VERSION:
            values["pretokenizer"] = self.pretokenizer
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BpeTokenizer":
        """Validates and restores a tokenizer.json object."""
        if not isinstance(values, dict):
            raise ValueError("tokenizer JSON root must be an object")
        if values.get("type") != "bpe":
            raise ValueError("tokenizer type must be 'bpe'")
        version = values.get("version")
        if version == cls.LEGACY_FORMAT_VERSION:
            pretokenizer_name = cls.LEGACY_PRETOKENIZER
        elif version in (cls.UNICODE_PRETOKENIZER_VERSION, cls.FORMAT_VERSION):
            pretokenizer_name = values.get("pretokenizer")
            if pretokenizer_name != cls.DEFAULT_PRETOKENIZER:
                raise ValueError(
                    f"unsupported BPE pretokenizer: {pretokenizer_name!r}"
                )
        else:
            raise ValueError(f"unsupported BPE tokenizer version: {version!r}")
        if values.get("byte_fallback") is not True:
            raise ValueError("BPE tokenizer must enable byte_fallback")
        expected_special = {"pad": 256, "bos": 257, "eos": 258, "sep": 259}
        if values.get("special_tokens") != expected_special:
            raise ValueError("BPE tokenizer has incompatible special token ids")
        merges = values.get("merges")
        if not isinstance(merges, list):
            raise ValueError("BPE tokenizer merges must be a list")
        required_pieces = values.get("required_pieces", [])
        if (
            not isinstance(required_pieces, list)
            or not all(isinstance(piece, str) and piece for piece in required_pieces)
        ):
            raise ValueError("BPE required_pieces must be a list of non-empty strings")
        if required_pieces and version != cls.FORMAT_VERSION:
            raise ValueError(
                "BPE required_pieces metadata requires tokenizer version 3"
            )
        tokenizer = cls(
            merges,
            pretokenizer=pretokenizer_name,
            format_version=version,
            unicode_character_merges=(
                values.get("unicode_character_merges")
                if version == cls.FORMAT_VERSION
                else False
            ),
            required_pieces=(
                required_pieces
                if version == cls.FORMAT_VERSION
                else ()
            ),
        )
        if values.get("vocab_size") != tokenizer.VOCAB_SIZE:
            raise ValueError("BPE tokenizer vocab_size does not match its merges")
        return tokenizer

    @classmethod
    def load(cls, path: str | Path) -> "BpeTokenizer":
        """Loads a tokenizer from tokenizer.json."""
        source = Path(path)
        try:
            values = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid tokenizer JSON: {exc}") from exc
        return cls.from_dict(values)

    def save(self, path: str | Path) -> Path:
        """Atomically saves tokenizer.json."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        return destination


def train_bpe_tokenizer(
    texts: Iterable[str] | str,
    *,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    pretokenizer: str = BpeTokenizer.DEFAULT_PRETOKENIZER,
    ensure_unicode_characters: bool = True,
    required_pieces: Iterable[str] = (),
) -> BpeTokenizer:
    """Learns deterministic byte-level BPE merges from training text only.

    By default, frequent multi-byte Unicode characters are made atomic before
    ordinary frequency merges are learned. This is especially important for
    Cyrillic: training targets no longer split a common letter into independent
    UTF-8 bytes, while byte fallback still represents unseen characters.
    """
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size < ByteTokenizer.VOCAB_SIZE
    ):
        raise ValueError(f"vocab_size must be at least {ByteTokenizer.VOCAB_SIZE}")
    if (
        not isinstance(min_frequency, int)
        or isinstance(min_frequency, bool)
        or min_frequency <= 0
    ):
        raise ValueError("min_frequency must be a positive integer")
    if not isinstance(ensure_unicode_characters, bool):
        raise TypeError("ensure_unicode_characters must be a boolean")
    if isinstance(required_pieces, (str, bytes)):
        raise TypeError("required_pieces must be an iterable of strings")
    normalized_required: list[str] = []
    for piece in required_pieces:
        if not isinstance(piece, str) or not piece:
            raise ValueError("required_pieces must contain non-empty strings")
        if piece in normalized_required:
            raise ValueError(f"duplicate required piece: {piece!r}")
        if any(character.isspace() for character in piece):
            raise ValueError(
                "required pieces cannot contain whitespace: "
                f"{piece!r}"
            )
        normalized_required.append(piece)
    documents = [texts] if isinstance(texts, str) else texts
    sequences: Counter[tuple[int, ...]] = Counter()
    character_counts: Counter[str] = Counter()
    for text in documents:
        if not isinstance(text, str):
            raise TypeError("every tokenizer training document must be a string")
        character_counts.update(text)
        for chunk in pretokenize(text, mode=pretokenizer):
            raw = tuple(chunk.encode("utf-8"))
            if raw:
                sequences[raw] += 1
    if not sequences:
        raise ValueError("BPE tokenizer training corpus is empty")

    merges: list[tuple[int, int]] = []

    def add_merge(pair: tuple[int, int]) -> None:
        nonlocal sequences
        token_id = ByteTokenizer.VOCAB_SIZE + len(merges)
        merges.append(pair)
        replaced: Counter[tuple[int, ...]] = Counter()
        for sequence, frequency in sequences.items():
            replaced[BpeTokenizer._replace_pair(sequence, pair, token_id)] += frequency
        sequences = replaced

    def encode_with_known_merges(raw: tuple[int, ...]) -> tuple[int, ...]:
        encoded = raw
        ranks = {pair: rank for rank, pair in enumerate(merges)}
        while len(encoded) > 1:
            candidates = {
                pair for pair in zip(encoded, encoded[1:]) if pair in ranks
            }
            if not candidates:
                break
            pair = min(candidates, key=ranks.__getitem__)
            encoded = BpeTokenizer._replace_pair(
                encoded, pair, ByteTokenizer.VOCAB_SIZE + ranks[pair],
            )
        return encoded

    if ensure_unicode_characters:
        # Reserve complete tokens for the most frequent non-ASCII characters.
        # A conservative upper bound avoids spending the last vocabulary slots
        # on only half of a multi-byte scalar.
        ordered_characters = sorted(
            (
                (character, frequency)
                for character, frequency in character_counts.items()
                if len(character.encode("utf-8")) > 1
            ),
            key=lambda item: (-item[1], ord(item[0])),
        )
        for character, _frequency in ordered_characters:
            raw = tuple(character.encode("utf-8"))
            if len(raw) - 1 > vocab_size - (ByteTokenizer.VOCAB_SIZE + len(merges)):
                continue
            encoded = encode_with_known_merges(raw)
            while len(encoded) > 1:
                if ByteTokenizer.VOCAB_SIZE + len(merges) >= vocab_size:
                    break
                pair = (encoded[0], encoded[1])
                known = {value: rank for rank, value in enumerate(merges)}
                if pair in known:
                    encoded = BpeTokenizer._replace_pair(
                        encoded, pair, ByteTokenizer.VOCAB_SIZE + known[pair],
                    )
                    continue
                add_merge(pair)
                encoded = BpeTokenizer._replace_pair(
                    encoded, pair, ByteTokenizer.VOCAB_SIZE + len(merges) - 1,
                )

    # Domain names, role markers and other important low-frequency strings can
    # otherwise remain split into raw bytes. Reserve their merge paths before
    # ordinary frequency merges so fine-tuning cannot over-amplify one byte of
    # a repeatedly used identifier.
    for piece in normalized_required:
        encoded = encode_with_known_merges(tuple(piece.encode("utf-8")))
        while len(encoded) > 1:
            if ByteTokenizer.VOCAB_SIZE + len(merges) >= vocab_size:
                raise ValueError(
                    f"vocab_size is too small to reserve required piece {piece!r}"
                )
            pair = (encoded[0], encoded[1])
            known = {value: rank for rank, value in enumerate(merges)}
            if pair in known:
                token_id = ByteTokenizer.VOCAB_SIZE + known[pair]
            else:
                add_merge(pair)
                token_id = ByteTokenizer.VOCAB_SIZE + len(merges) - 1
            encoded = BpeTokenizer._replace_pair(encoded, pair, token_id)

    while ByteTokenizer.VOCAB_SIZE + len(merges) < vocab_size:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for sequence, frequency in sequences.items():
            for pair in zip(sequence, sequence[1:]):
                pair_counts[pair] += frequency
        eligible = [pair for pair, frequency in pair_counts.items() if frequency >= min_frequency]
        if not eligible:
            break
        pair = min(eligible, key=lambda item: (-pair_counts[item], item))
        add_merge(pair)
    return BpeTokenizer(
        merges,
        pretokenizer=pretokenizer,
        unicode_character_merges=ensure_unicode_characters,
        required_pieces=normalized_required,
    )


def analyze_tokenizer(
    tokenizer: ByteTokenizer,
    texts: Iterable[str] | str,
) -> TokenizerReport:
    """Measures compression, fallback use, coverage, and exact round trips."""
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer instance")
    documents = [texts] if isinstance(texts, str) else texts
    document_count = 0
    characters = 0
    words = 0
    byte_count = 0
    token_count = 0
    raw_byte_tokens = 0
    non_ascii_byte_tokens = 0
    roundtrip_errors = 0
    replacement_characters = 0
    unique_tokens: set[int] = set()
    non_ascii_characters: Counter[str] = Counter()

    for text in documents:
        if not isinstance(text, str):
            raise TypeError("every tokenizer report document must be a string")
        document_count += 1
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        characters += len(text)
        words += len(_WORD_PATTERN.findall(text))
        byte_count += len(text.encode("utf-8"))
        token_count += len(encoded)
        unique_tokens.update(encoded)
        raw_byte_tokens += sum(0 <= token <= 255 for token in encoded)
        non_ascii_byte_tokens += sum(128 <= token <= 255 for token in encoded)
        roundtrip_errors += int(decoded != text)
        replacement_characters += decoded.count("\ufffd") - text.count("\ufffd")
        non_ascii_characters.update(
            character for character in text if ord(character) > 127
        )

    if document_count == 0 or byte_count == 0:
        raise ValueError("tokenizer report corpus must contain non-empty text")
    unicode_total = sum(non_ascii_characters.values())
    unicode_atomic = sum(
        frequency
        for character, frequency in non_ascii_characters.items()
        if len(tokenizer.encode(character)) == 1
    )
    compression_ratio = token_count / byte_count
    unicode_coverage = unicode_atomic / unicode_total if unicode_total else 1.0
    warnings: list[str] = []
    if roundtrip_errors:
        warnings.append("tokenizer does not round-trip every report document")
    if compression_ratio > 0.75:
        warnings.append(
            "token count exceeds 75% of UTF-8 bytes; consider a larger corpus or vocabulary"
        )
    if unicode_total and unicode_coverage < 0.95:
        warnings.append(
            "less than 95% of non-ASCII characters are represented atomically"
        )
    if token_count and raw_byte_tokens / token_count > 0.50:
        warnings.append("more than half of encoded tokens are raw byte fallback tokens")
    return TokenizerReport(
        documents=document_count,
        characters=characters,
        words=words,
        utf8_bytes=byte_count,
        tokens=token_count,
        unique_tokens=len(unique_tokens),
        vocab_size=tokenizer.VOCAB_SIZE,
        raw_byte_tokens=raw_byte_tokens,
        non_ascii_byte_tokens=non_ascii_byte_tokens,
        roundtrip_errors=roundtrip_errors,
        replacement_characters=max(0, replacement_characters),
        compression_ratio=compression_ratio,
        tokens_per_word=token_count / max(1, words),
        vocab_utilization=len(unique_tokens) / tokenizer.VOCAB_SIZE,
        unicode_atomic_coverage=unicode_coverage,
        warnings=tuple(warnings),
    )


def save_tokenizer_report(report: TokenizerReport, path: str | Path) -> Path:
    """Atomically writes a tokenizer quality report as readable JSON."""
    if not isinstance(report, TokenizerReport):
        raise TypeError("report must be a TokenizerReport")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination


def load_tokenizer(path: str | Path) -> BpeTokenizer:
    """Loads a trainable tokenizer artifact."""
    return BpeTokenizer.load(path)


def save_tokenizer(tokenizer: ByteTokenizer, path: str | Path) -> Path:
    """Saves tokenizers that require a model-side artifact."""
    if not isinstance(tokenizer, BpeTokenizer):
        raise TypeError("only BpeTokenizer has a tokenizer.json artifact")
    return tokenizer.save(path)


def create_tokenizer(
    name: str = "byte", *, path: str | Path | None = None,
) -> ByteTokenizer:
    """Creates a tokenizer by the stable name stored in model config.json."""
    if not isinstance(name, str):
        raise TypeError("tokenizer name must be a string")
    normalized = name.strip().lower()
    if normalized == "byte":
        return ByteTokenizer()
    if normalized in {"unicode", "unicode_byte"}:
        return UnicodeByteTokenizer()
    if normalized == "bpe":
        if path is None:
            raise ValueError("tokenizer='bpe' requires tokenizer.json")
        return BpeTokenizer.load(path)
    raise ValueError("tokenizer must be 'byte', 'unicode', or 'bpe'")


def _resolve_tokenizer(
    tokenizer: ByteTokenizer | str | Path,
    *,
    path: str | Path | None,
) -> ByteTokenizer:
    if isinstance(tokenizer, ByteTokenizer):
        if path is not None:
            raise ValueError("path cannot be used with a tokenizer instance")
        return tokenizer
    if isinstance(tokenizer, Path):
        if path is not None:
            raise ValueError("provide the tokenizer path only once")
        return load_tokenizer(tokenizer)
    if not isinstance(tokenizer, str):
        raise TypeError("tokenizer must be a name, path, or tokenizer instance")
    normalized = tokenizer.strip().lower()
    if normalized in {"byte", "unicode", "unicode_byte", "bpe"}:
        return create_tokenizer(normalized, path=path)
    if path is not None:
        raise ValueError("path can only be combined with tokenizer='bpe'")
    return load_tokenizer(tokenizer)


def tokenize(
    text: str,
    tokenizer: ByteTokenizer | str | Path = "byte",
    *,
    path: str | Path | None = None,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[int]:
    """Encode text in one call using a tokenizer name, artifact path, or instance."""
    selected = _resolve_tokenizer(tokenizer, path=path)
    return selected.encode(text, add_bos=add_bos, add_eos=add_eos)


def detokenize(
    tokens: Iterable[int],
    tokenizer: ByteTokenizer | str | Path = "byte",
    *,
    path: str | Path | None = None,
    skip_special: bool = True,
) -> str:
    """Decode token ids in one call using a tokenizer name, artifact path, or instance."""
    selected = _resolve_tokenizer(tokenizer, path=path)
    return selected.decode(tokens, skip_special=skip_special)
