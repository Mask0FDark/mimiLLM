#!/usr/bin/env python3
"""Benchmark a full mimiLLM training step with deterministic inputs."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimillm.backend import get_backend, reset_backend
from mimillm.optim import AdamW
from mimillm.transformer import DecoderTransformer, TransformerConfig
from mimillm.utils import flatten


def synthetic_batch(
    config: TransformerConfig, *, seed: int,
) -> tuple[list[list[int]], list[list[int]]]:
    rng = random.Random(seed)
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for _ in range(config.batch_size):
        row = [config.vocab_size - 3]
        row.extend(rng.randrange(0, min(256, config.vocab_size)) for _ in range(config.context_length))
        window = row[: config.context_length + 1]
        inputs.append(window[:-1])
        targets.append(window[1:])
    return inputs, targets


def _training_state(
    *,
    backend: str = "python",
    seed: int = 42,
    context_length: int = 16,
    d_model: int = 16,
    n_layers: int = 1,
    n_heads: int = 2,
    d_mlp: int = 32,
    batch_size: int = 2,
) -> tuple[
    Any, TransformerConfig, DecoderTransformer, AdamW,
    list[list[int]], list[list[int]],
]:
    os.environ["MIMILLM_BACKEND"] = backend
    reset_backend()
    selected_backend = get_backend()
    config = TransformerConfig(
        context_length=context_length,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        batch_size=batch_size,
        steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=0,
        validation_interval=1,
        checkpoint_interval=1,
        seed=seed,
    )
    model = DecoderTransformer(config)
    optimizer = AdamW(model.parameters(), config.learning_rate, weight_decay=0.0)
    inputs, targets = synthetic_batch(config, seed=seed + 1)
    return selected_backend, config, model, optimizer, inputs, targets


def _training_step(
    selected_backend: Any,
    config: TransformerConfig,
    model: DecoderTransformer,
    optimizer: AdamW,
    inputs: list[list[int]],
    targets: list[list[int]],
) -> dict[str, Any]:
    started = time.perf_counter()
    logits = model(inputs)
    loss = logits.reshape(-1, config.vocab_size).cross_entropy(flatten(targets))
    loss_value = loss.item()
    loss.backward()
    grad_norm = optimizer.clip_grad_norm(1.0)
    optimizer.step()
    optimizer.zero_grad()
    elapsed = time.perf_counter() - started
    checksum = sum(sum(parameter.data) for parameter in model.parameters())
    tokens = sum(len(row) for row in inputs)
    return {
        "backend": getattr(selected_backend, "name", "unknown"),
        "tokens": tokens,
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed if elapsed > 0.0 else float("inf"),
        "loss": loss_value,
        "grad_norm": grad_norm,
        "parameter_checksum": checksum,
        "parameters": model.parameter_count(),
    }


def training_step_snapshot(
    *,
    backend: str = "python",
    seed: int = 42,
    context_length: int = 16,
    d_model: int = 16,
    n_layers: int = 1,
    n_heads: int = 2,
    d_mlp: int = 32,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Run one timed step after backend initialization."""
    state = _training_state(
        backend=backend,
        seed=seed,
        context_length=context_length,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        batch_size=batch_size,
    )
    return _training_step(*state)


def benchmark_training_step(
    *, repeats: int = 5, warmup: int = 1, **options: Any,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    state = _training_state(**options)
    for _ in range(warmup):
        _training_step(*state)
    results = [_training_step(*state) for _ in range(repeats)]
    best = min(results, key=lambda item: item["seconds"])
    seconds = [item["seconds"] for item in results]
    mean_seconds = statistics.mean(seconds)
    median_seconds = statistics.median(seconds)
    return {
        **best,
        "repeats": repeats,
        "warmup": warmup,
        "best_seconds": best["seconds"],
        "mean_seconds": mean_seconds,
        "median_seconds": median_seconds,
        "mean_tokens_per_second": best["tokens"] / mean_seconds,
        "median_tokens_per_second": best["tokens"] / median_seconds,
        "samples_seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one mimiLLM training step")
    parser.add_argument("--backend", default=os.environ.get("MIMILLM_BACKEND", "auto"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    result = benchmark_training_step(
        backend=args.backend,
        repeats=args.repeats,
        warmup=args.warmup,
        context_length=args.context_length,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
