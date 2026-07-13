#!/usr/bin/env python3
"""Обучает m0fdii предсказывать следующий UTF-8 byte token."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from minillm.backend import get_backend
from minillm.checkpoint import load_checkpoint, save_checkpoint
from minillm.dataset import TokenDataset
from minillm.optim import AdamW
from minillm.transformer import DecoderTransformer, TransformerConfig
from minillm.utils import flatten, learning_rate_at


ROOT = Path(__file__).resolve().parent


def validation_loss(model: DecoderTransformer, dataset: TokenDataset, config: TransformerConfig) -> float:
    """Считает validation на отдельном детерминированном batch."""
    inputs, targets = dataset.deterministic_batch(config.batch_size, config.context_length)
    logits = model(inputs)
    return logits.reshape(-1, config.vocab_size).cross_entropy(flatten(targets)).item()


def run_training(
    config: TransformerConfig, *, resume: Path | None = None,
    output: Path | None = None,
) -> Path:
    """Выполняет весь цикл и возвращает путь последнего checkpoint."""
    rng = random.Random(config.seed)
    train_data = TokenDataset(ROOT / "data" / "train.txt")
    validation_data = TokenDataset(ROOT / "data" / "validation.txt")
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
    destination = output or ROOT / "checkpoints" / "debug.bin"
    print(f"m0fdii: parameters={model.parameter_count()}, backend={backend_name}, threads={threads}")
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
            tokens = config.batch_size * config.context_length
            message = (
                f"step={step} train_loss={loss.item():.5f} "
                f"tok/s={tokens / elapsed:.1f} time={elapsed:.3f}s "
                f"lr={optimizer.learning_rate:.6g} backend={backend_name} threads={threads}"
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
    args = parser.parse_args()
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        config = TransformerConfig.from_dict(checkpoint.config)
    else:
        config = TransformerConfig.from_json(args.config)
    path = run_training(config, resume=args.resume, output=args.output)
    print(f"Checkpoint: {path}")


if __name__ == "__main__":
    main()

