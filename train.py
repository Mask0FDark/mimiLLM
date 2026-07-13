#!/usr/bin/env python3
"""Обучает m0fdii предсказывать следующий UTF-8 byte token."""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from minillm.backend import get_backend
from minillm.checkpoint import load_checkpoint, save_checkpoint
from minillm.dataset import TokenDataset
from minillm.optim import AdamW
from minillm.transformer import DecoderTransformer, TransformerConfig
from minillm.utils import flatten, learning_rate_at


ROOT = Path(__file__).resolve().parent
DEFAULT_TEXT_TRAIN = ROOT / "data" / "text" / "train"
DEFAULT_TEXT_VALIDATION = ROOT / "data" / "text" / "validation"


def validation_loss(model: DecoderTransformer, dataset: TokenDataset, config: TransformerConfig) -> float:
    """Считает взвешенный QA/text validation на детерминированных batch."""
    total = 0.0
    for source, weight in dataset.source_weights():
        inputs, targets = dataset.deterministic_batch(
            config.batch_size, config.context_length, source=source
        )
        logits = model(inputs)
        loss = logits.reshape(-1, config.vocab_size).cross_entropy(flatten(targets)).item()
        total += weight * loss
    return total


def run_training(
    config: TransformerConfig, *, resume: Path | None = None,
    output: Path | None = None,
    text_train_paths: Sequence[Path] | None = None,
    text_validation_paths: Sequence[Path] | None = None,
) -> Path:
    """Выполняет весь цикл и возвращает путь последнего checkpoint."""
    rng = random.Random(config.seed)
    train_text = list(text_train_paths) if text_train_paths is not None else [DEFAULT_TEXT_TRAIN]
    validation_text = (
        list(text_validation_paths) if text_validation_paths is not None
        else [DEFAULT_TEXT_VALIDATION]
    )
    train_data = TokenDataset(
        ROOT / "data" / "train.txt", text_paths=train_text, text_ratio=config.text_ratio
    )
    validation_data = TokenDataset(
        ROOT / "data" / "validation.txt",
        text_paths=validation_text,
        text_ratio=config.text_ratio,
    )
    model = DecoderTransformer(config)
    optimizer = AdamW(
        model.parameters(), config.learning_rate, weight_decay=config.weight_decay
    )
    start_step = 0
    if resume is not None:
        loaded = load_checkpoint(resume, model, optimizer)
        start_step = loaded.step
        rng = random.Random(loaded.seed + start_step)
        print(f"Продолжение с шага {start_step}: {resume}")
    backend = get_backend()
    backend_name = getattr(backend, "name", "python")
    threads = getattr(backend, "num_threads", 1)
    destination = output or resume or ROOT / "checkpoints" / "debug.bin"
    print(f"m0fdii: parameters={model.parameter_count()}, backend={backend_name}, threads={threads}")
    print(
        f"data: qa_train={len(train_data.examples)} "
        f"text_train={len(train_data.text_documents)} text_ratio={config.text_ratio:.2f}"
    )
    last_saved = destination
    try:
        for step in range(start_step + 1, config.steps + 1):
            started = time.perf_counter()
            inputs, targets = train_data.sample_batch(
                config.batch_size, config.context_length, rng
            )
            optimizer.learning_rate = learning_rate_at(
                step, config.steps, config.learning_rate, config.warmup_steps
            )
            logits = model(inputs)
            loss = logits.reshape(-1, config.vocab_size).cross_entropy(flatten(targets))
            loss.backward()
            optimizer.clip_grad_norm(1.0)
            optimizer.step()
            optimizer.zero_grad()
            elapsed = time.perf_counter() - started
            tokens = sum(len(row) for row in inputs)
            message = (
                f"step={step} train_loss={loss.item():.5f} "
                f"tok/s={tokens / elapsed:.1f} time={elapsed:.3f}s "
                f"lr={optimizer.learning_rate:.6g} source={train_data.last_source} "
                f"backend={backend_name} threads={threads}"
            )
            if step % config.validation_interval == 0 or step == config.steps:
                message += f" validation_loss={validation_loss(model, validation_data, config):.5f}"
            print(message, flush=True)
            if step % config.checkpoint_interval == 0 or step == config.steps:
                last_saved = save_checkpoint(
                    destination, model, optimizer, config=config.to_dict(),
                    step=step, seed=config.seed,
                )
    except KeyboardInterrupt:
        emergency = destination.with_name(destination.stem + "_interrupted.bin")
        last_saved = save_checkpoint(
            emergency, model, optimizer, config=config.to_dict(),
            step=optimizer.step_count, seed=config.seed,
        )
        print(f"\nОбучение прервано. Аварийный checkpoint: {last_saved}")
    return last_saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение m0fdii")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "debug.json")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, help="Новый общий предел шагов, в том числе при resume")
    parser.add_argument(
        "--text-train", type=Path, action="append", metavar="PATH",
        help="UTF-8 файл или каталог train-текстов; параметр можно повторять",
    )
    parser.add_argument(
        "--text-validation", type=Path, action="append", metavar="PATH",
        help="отдельный UTF-8 файл или каталог validation-текстов",
    )
    parser.add_argument(
        "--text-ratio", type=float,
        help="доля batch из обычных текстов от 0 до 1; переопределяет config",
    )
    args = parser.parse_args()
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        config = TransformerConfig.from_dict(checkpoint.config)
    else:
        config = TransformerConfig.from_json(args.config)
    if args.steps is not None:
        if args.steps <= 0:
            parser.error("--steps должен быть положительным")
        config = replace(config, steps=args.steps)
    if args.text_ratio is not None:
        if not 0.0 <= args.text_ratio <= 1.0:
            parser.error("--text-ratio должен быть от 0 до 1")
        config = replace(config, text_ratio=args.text_ratio)
    path = run_training(
        config,
        resume=args.resume,
        output=args.output,
        text_train_paths=args.text_train,
        text_validation_paths=args.text_validation,
    )
    print(f"Checkpoint: {path}")


if __name__ == "__main__":
    main()
