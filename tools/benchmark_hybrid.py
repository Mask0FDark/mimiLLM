#!/usr/bin/env python3
"""Compare CUDA-only and experimental CPU+GPU training on project batches."""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from mimillm import AdamW, DecoderTransformer, TransformerConfig, reset_backend
from mimillm.hybrid import HybridDataParallel
from mimillm.training import _datasets
from mimillm.utils import flatten


@dataclass(frozen=True)
class BenchmarkResult:
    model: DecoderTransformer
    seconds: float
    tokens: int
    losses: list[float]

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds


def cuda_run(
    config: TransformerConfig,
    batches: list[tuple[list[list[int]], list[list[int]], list[list[float]]]],
) -> BenchmarkResult:
    model = DecoderTransformer(config)
    optimizer = AdamW(
        model.parameters(), config.learning_rate, weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    tokens = 0
    started = time.perf_counter()
    for inputs, targets, weights in batches:
        logits = model(inputs)
        loss = logits.reshape(-1, config.vocab_size).cross_entropy(
            flatten(targets), weights=flatten(weights),  # type: ignore[arg-type]
        )
        losses.append(loss.item())
        loss.backward()
        optimizer.clip_grad_norm(1.0)
        optimizer.step()
        optimizer.zero_grad()
        tokens += sum(len(row) for row in inputs)
    return BenchmarkResult(model, time.perf_counter() - started, tokens, losses)


def hybrid_run(
    config: TransformerConfig,
    batches: list[tuple[list[list[int]], list[list[int]], list[list[float]]]],
    *,
    cpu_batch_size: int,
    cpu_threads: int,
) -> BenchmarkResult:
    model = DecoderTransformer(config)
    optimizer = AdamW(
        model.parameters(), config.learning_rate, weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    tokens = 0
    started = time.perf_counter()
    with HybridDataParallel(
        model,
        cpu_batch_size=cpu_batch_size,
        cpu_threads=cpu_threads,
    ) as hybrid:
        for inputs, targets, weights in batches:
            result = hybrid.forward_backward(inputs, targets, weights)
            losses.append(result.loss)
            optimizer.clip_grad_norm(1.0)
            optimizer.step()
            optimizer.zero_grad()
            hybrid.sync_replica()
            tokens += sum(len(row) for row in inputs)
    return BenchmarkResult(model, time.perf_counter() - started, tokens, losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="project config.json")
    parser.add_argument("--batches", type=int, default=6, help="number of real batches")
    parser.add_argument("--cpu-batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batches <= 0:
        parser.error("--batches must be positive")

    os.environ["MIMILLM_BACKEND"] = "cuda"
    reset_backend()
    config_path = args.config.resolve()
    config = TransformerConfig.from_json(config_path)
    if not 0 < args.cpu_batch_size < config.batch_size:
        parser.error("--cpu-batch-size must be between 1 and batch_size - 1")
    train_data, _ = _datasets(config, config_path.parent)
    rng = random.Random(config.seed + 10_000)
    batches = [
        train_data.sample_batch_with_loss_weights(
            config.batch_size, config.context_length, rng,
        )
        for _ in range(args.batches)
    ]

    cuda = cuda_run(config, batches)
    hybrid = hybrid_run(
        config,
        batches,
        cpu_batch_size=args.cpu_batch_size,
        cpu_threads=args.cpu_threads,
    )
    max_weight_difference = max(
        abs(left - right)
        for cuda_parameter, hybrid_parameter in zip(
            cuda.model.parameters(), hybrid.model.parameters(), strict=True,
        )
        for left, right in zip(cuda_parameter.data, hybrid_parameter.data, strict=True)
    )
    max_loss_difference = max(
        abs(left - right) for left, right in zip(cuda.losses, hybrid.losses, strict=True)
    )
    print(
        f"CUDA:   {cuda.seconds:.3f}s | {cuda.tokens_per_second:.1f} tok/s | "
        f"batch={config.batch_size}"
    )
    print(
        f"Hybrid: {hybrid.seconds:.3f}s | {hybrid.tokens_per_second:.1f} tok/s | "
        f"gpu_batch={config.batch_size - args.cpu_batch_size} | "
        f"cpu_batch={args.cpu_batch_size} | cpu_threads={args.cpu_threads}"
    )
    print(f"Speedup: {cuda.seconds / hybrid.seconds:.3f}x")
    print(f"Max loss difference:   {max_loss_difference:.8g}")
    print(f"Max weight difference: {max_weight_difference:.8g}")


if __name__ == "__main__":
    main()
