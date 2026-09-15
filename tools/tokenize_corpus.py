#!/usr/bin/env python3
"""Потоково переводит raw UTF-8 корпус mimiLLM в mmap `.mmtok` shards."""

from __future__ import annotations

import argparse
import json
import shutil
from array import array
from pathlib import Path

from mimillm.dataset import discover_text_files
from mimillm.token_shard import MappedTokenShard, tokenizer_fingerprint, write_token_shard
from mimillm.tokenizer import create_tokenizer
from mimillm.transformer import TransformerConfig


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _prepare_output(path: Path, *, force: bool) -> None:
    if path.exists():
        existing = list(path.iterdir()) if path.is_dir() else [path]
        if existing and not force:
            raise FileExistsError(
                f"output не пуст: {path}; используйте --force для пересборки"
            )
        if force:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _iter_encoded_chunks(
    files: list[Path],
    tokenizer: object,
    *,
    chunk_characters: int,
):
    encode = getattr(tokenizer, "encode")
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            while True:
                text = stream.read(chunk_characters)
                if not text:
                    break
                if not text.strip():
                    continue
                yield encode(text, add_bos=True, add_eos=True)


def build_shards(
    config_path: Path,
    *,
    split: str,
    output_dir: Path,
    shard_tokens: int,
    chunk_characters: int,
    force: bool,
) -> dict[str, object]:
    if shard_tokens < 2:
        raise ValueError("shard_tokens должен быть >= 2")
    if chunk_characters <= 0:
        raise ValueError("chunk_characters должен быть положительным")

    config_path = config_path.resolve()
    config = TransformerConfig.from_json(config_path)
    base = config_path.parent
    source = _resolve(
        base,
        config.text_train_path if split == "train" else config.text_validation_path,
    )
    files = discover_text_files(source)
    if not files:
        raise ValueError(f"raw корпус пуст: {source}")

    tokenizer_path = _resolve(base, config.tokenizer_path)
    tokenizer = create_tokenizer(
        config.tokenizer,
        path=tokenizer_path if config.tokenizer.strip().lower() == "bpe" else None,
    )
    if tokenizer.VOCAB_SIZE != config.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary={tokenizer.VOCAB_SIZE}, config={config.vocab_size}"
        )
    fingerprint = tokenizer_fingerprint(tokenizer)

    _prepare_output(output_dir, force=force)
    pending = array("I")
    shard_index = 0
    total_tokens = 0
    source_bytes = sum(path.stat().st_size for path in files)
    written: list[dict[str, object]] = []

    def flush(count: int) -> None:
        nonlocal pending, shard_index, total_tokens
        if count < 2:
            return
        shard_index += 1
        path = output_dir / f"{split}-{shard_index:05d}.mmtok"
        values = pending[:count]
        write_token_shard(
            path,
            values,
            vocab_size=config.vocab_size,
            tokenizer_sha256=fingerprint,
        )
        written.append({"file": path.name, "tokens": len(values)})
        total_tokens += len(values)
        pending = pending[count:]

    for encoded in _iter_encoded_chunks(
        files,
        tokenizer,
        chunk_characters=chunk_characters,
    ):
        offset = 0
        while offset < len(encoded):
            capacity = shard_tokens - len(pending)
            take = min(capacity, len(encoded) - offset)
            pending.extend(encoded[offset:offset + take])
            offset += take
            if len(pending) >= shard_tokens:
                flush(shard_tokens)

    if pending:
        if len(pending) == 1 and written:
            # Не создаём недопустимый однотокенный shard: переносим один токен
            # в последний файл через безопасную пересборку его небольшого хвоста.
            last = written.pop()
            last_path = output_dir / str(last["file"])
            with MappedTokenShard(
                last_path,
                expected_vocab_size=config.vocab_size,
                expected_tokenizer_sha256=fingerprint,
            ) as shard:
                merged = array("I", shard[:])
            last_path.unlink()
            total_tokens -= int(last["tokens"])
            merged.extend(pending)
            write_token_shard(
                last_path,
                merged,
                vocab_size=config.vocab_size,
                tokenizer_sha256=fingerprint,
            )
            written.append({"file": last_path.name, "tokens": len(merged)})
            total_tokens += len(merged)
            pending = array("I")
        else:
            flush(len(pending))

    if not written:
        raise ValueError("после токенизации корпус не содержит достаточно токенов")

    manifest = {
        "format": "mimiLLM-token-shards",
        "version": 1,
        "split": split,
        "vocab_size": config.vocab_size,
        "tokenizer_sha256": fingerprint,
        "source": str(source),
        "source_files": len(files),
        "source_bytes": source_bytes,
        "tokens": total_tokens,
        "shards": written,
        "shard_tokens_target": shard_tokens,
        "chunk_characters": chunk_characters,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build bounded-memory mimiLLM mmap token shards"
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-tokens", type=int, default=8_000_000)
    parser.add_argument("--chunk-characters", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = build_shards(
        args.config,
        split=args.split,
        output_dir=args.output.resolve(),
        shard_tokens=args.shard_tokens,
        chunk_characters=args.chunk_characters,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
