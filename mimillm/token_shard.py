"""Компактные memory-mapped token shards для больших текстовых корпусов."""

from __future__ import annotations

import mmap
import os
import struct
import sys
from array import array
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path


MAGIC = b"MIMTOK1\0"
VERSION = 1
HEADER = struct.Struct("<8sIIIQ")
TOKEN_SHARD_SUFFIX = ".mmtok"
_SUPPORTED_ITEM_SIZES = {2: "H", 4: "I"}


class MappedTokenShard(Sequence[int]):
    """Read-only последовательность token ID поверх mmap без загрузки в RAM."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_vocab_size: int | None = None,
    ) -> None:
        self.path = Path(path)
        self._stream = self.path.open("rb")
        try:
            raw_header = self._stream.read(HEADER.size)
            if len(raw_header) != HEADER.size:
                raise ValueError(f"token shard слишком короткий: {self.path}")
            magic, version, item_size, vocab_size, token_count = HEADER.unpack(
                raw_header
            )
            if magic != MAGIC:
                raise ValueError(f"неверный magic token shard: {self.path}")
            if version != VERSION:
                raise ValueError(
                    f"неподдерживаемая версия token shard {version}: {self.path}"
                )
            if item_size not in _SUPPORTED_ITEM_SIZES:
                raise ValueError(
                    f"token shard использует неподдерживаемый item_size={item_size}"
                )
            if vocab_size <= 0:
                raise ValueError("token shard содержит неверный vocab_size")
            if (
                expected_vocab_size is not None
                and vocab_size != expected_vocab_size
            ):
                raise ValueError(
                    f"token shard vocabulary={vocab_size}, "
                    f"tokenizer vocabulary={expected_vocab_size}: {self.path}"
                )
            expected_size = HEADER.size + token_count * item_size
            actual_size = os.fstat(self._stream.fileno()).st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"размер token shard не совпадает с header: "
                    f"ожидалось {expected_size}, получено {actual_size}: {self.path}"
                )
            if token_count < 2:
                raise ValueError(f"token shard должен содержать минимум 2 токена")
            self.item_size = item_size
            self.vocab_size = vocab_size
            self.token_count = token_count
            self._format = "<H" if item_size == 2 else "<I"
            self._map = mmap.mmap(
                self._stream.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )
        except Exception:
            self._stream.close()
            raise

    def __len__(self) -> int:
        return self.token_count

    def _normalize_index(self, index: int) -> int:
        normalized = index + self.token_count if index < 0 else index
        if not 0 <= normalized < self.token_count:
            raise IndexError("token shard index out of range")
        return normalized

    def __getitem__(self, key: int | slice) -> int | list[int]:
        if isinstance(key, int):
            index = self._normalize_index(key)
            offset = HEADER.size + index * self.item_size
            return int(struct.unpack_from(self._format, self._map, offset)[0])
        if not isinstance(key, slice):
            raise TypeError("token shard поддерживает только int и slice")
        start, stop, step = key.indices(self.token_count)
        if step != 1:
            return [self[index] for index in range(start, stop, step)]
        count = max(0, stop - start)
        if count == 0:
            return []
        begin = HEADER.size + start * self.item_size
        end = begin + count * self.item_size
        typecode = _SUPPORTED_ITEM_SIZES[self.item_size]
        values = array(typecode)
        values.frombytes(self._map[begin:end])
        if sys.byteorder != "little":
            values.byteswap()
        return [int(value) for value in values]

    def close(self) -> None:
        mapping = getattr(self, "_map", None)
        if mapping is not None:
            mapping.close()
            self._map = None  # type: ignore[assignment]
        stream = getattr(self, "_stream", None)
        if stream is not None and not stream.closed:
            stream.close()

    def __enter__(self) -> "MappedTokenShard":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def discover_token_shards(
    paths: Iterable[str | Path] | str | Path,
) -> list[Path]:
    """Находит `.mmtok` файлы в стабильном порядке и удаляет дубли путей."""
    requested = [paths] if isinstance(paths, (str, Path)) else list(paths)
    discovered: list[Path] = []
    for item in requested:
        path = Path(item)
        if path.is_file():
            if path.suffix.lower() != TOKEN_SHARD_SUFFIX:
                raise ValueError(f"ожидался token shard .mmtok: {path}")
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(
                child
                for child in path.rglob(f"*{TOKEN_SHARD_SUFFIX}")
                if child.is_file()
            )
        else:
            raise FileNotFoundError(f"путь token shard не найден: {path}")
    unique: dict[Path, Path] = {}
    for path in sorted(discovered, key=lambda value: value.as_posix().casefold()):
        unique.setdefault(path.resolve(), path)
    if not unique:
        raise ValueError("token shard каталог не содержит .mmtok файлов")
    return list(unique.values())


def load_token_shards(
    paths: Iterable[str | Path] | str | Path,
    *,
    expected_vocab_size: int | None = None,
) -> list[MappedTokenShard]:
    """Открывает все найденные shards через read-only mmap."""
    result: list[MappedTokenShard] = []
    try:
        for path in discover_token_shards(paths):
            result.append(
                MappedTokenShard(
                    path,
                    expected_vocab_size=expected_vocab_size,
                )
            )
    except Exception:
        for shard in result:
            shard.close()
        raise
    return result


def _select_item_size(vocab_size: int) -> int:
    if vocab_size <= 0:
        raise ValueError("vocab_size должен быть положительным")
    if vocab_size <= 0x10000:
        return 2
    if vocab_size <= 0x100000000:
        return 4
    raise ValueError("vocab_size не помещается в uint32")


def write_token_shard(
    path: str | Path,
    tokens: Iterable[int],
    *,
    vocab_size: int,
    item_size: int | None = None,
    chunk_tokens: int = 1_000_000,
) -> Path:
    """Потоково записывает token IDs без материализации всего корпуса в RAM."""
    selected_item_size = item_size or _select_item_size(vocab_size)
    if selected_item_size not in _SUPPORTED_ITEM_SIZES:
        raise ValueError("item_size должен быть 2 или 4")
    if selected_item_size == 2 and vocab_size > 0x10000:
        raise ValueError("uint16 недостаточно для заданного vocab_size")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens должен быть положительным")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    typecode = _SUPPORTED_ITEM_SIZES[selected_item_size]
    maximum = 0xFFFF if selected_item_size == 2 else 0xFFFFFFFF
    count = 0
    buffer = array(typecode)

    try:
        with temporary.open("wb+") as stream:
            stream.write(
                HEADER.pack(
                    MAGIC,
                    VERSION,
                    selected_item_size,
                    int(vocab_size),
                    0,
                )
            )
            for raw_token in tokens:
                if (
                    not isinstance(raw_token, int)
                    or isinstance(raw_token, bool)
                ):
                    raise TypeError("token ID должен быть целым числом")
                token = int(raw_token)
                if not 0 <= token < vocab_size or token > maximum:
                    raise ValueError(
                        f"token ID {token} вне диапазона vocabulary={vocab_size}"
                    )
                buffer.append(token)
                count += 1
                if len(buffer) >= chunk_tokens:
                    if sys.byteorder != "little":
                        buffer.byteswap()
                    stream.write(buffer.tobytes())
                    buffer = array(typecode)
            if buffer:
                if sys.byteorder != "little":
                    buffer.byteswap()
                stream.write(buffer.tobytes())
            if count < 2:
                raise ValueError("token shard должен содержать минимум 2 токена")
            stream.seek(0)
            stream.write(
                HEADER.pack(
                    MAGIC,
                    VERSION,
                    selected_item_size,
                    int(vocab_size),
                    count,
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def iter_document_tokens(
    tokenizer: object,
    documents: Iterable[str],
) -> Iterator[int]:
    """Кодирует документы по одному, не удерживая токены всего корпуса."""
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer должен предоставлять encode()")
    for text in documents:
        for token in encode(text, add_bos=True, add_eos=True):
            yield int(token)
