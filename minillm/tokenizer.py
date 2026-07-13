"""Byte-level токенизатор для UTF-8 текста."""

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

