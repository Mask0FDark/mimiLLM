"""Fixed byte and Unicode-aware tokenizers for UTF-8 text."""

from __future__ import annotations

from collections.abc import Iterable


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

    def encode_qa(self, question: str, answer: str) -> list[int]:
        """Кодирует законченную обучающую пару вопрос–ответ."""
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError("question и answer должны быть строками")
        text = f"Вопрос: {question.strip()}\nОтвет: {answer.strip()}"
        return [self.BOS, *self.encode(text), self.EOS]

    def encode_prompt(self, question: str) -> list[int]:
        """Кодирует начало ответа для авторегрессионной генерации."""
        if not isinstance(question, str):
            raise TypeError("question должен быть строкой")
        return [self.BOS, *self.encode(f"Вопрос: {question.strip()}\nОтвет:")]


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


def create_tokenizer(name: str = "byte") -> ByteTokenizer:
    """Creates a tokenizer by the stable name stored in model config.json."""
    if not isinstance(name, str):
        raise TypeError("tokenizer name must be a string")
    normalized = name.strip().lower()
    if normalized == "byte":
        return ByteTokenizer()
    if normalized in {"unicode", "unicode_byte"}:
        return UnicodeByteTokenizer()
    raise ValueError("tokenizer must be 'byte' or 'unicode'")
