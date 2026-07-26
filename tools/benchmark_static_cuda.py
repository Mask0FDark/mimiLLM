#!/usr/bin/env python3
"""Benchmark mimiLLM CUDA Graph training without third-party packages."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimillm.backend import reset_backend
from mimillm.optim import AdamW
from mimillm.static_cuda import compile_static_cuda_training
from mimillm.tokenizer import BpeTokenizer
from mimillm.transformer import DecoderTransformer, TransformerConfig


def _batch(config: TransformerConfig, seed: int) -> tuple[
    list[list[int]], list[list[int]],
]:
    rng = random.Random(seed)
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for _ in range(config.batch_size):
        row = [config.vocab_size - 3]
        row.extend(
            rng.randrange(0, min(256, config.vocab_size))
            for _ in range(config.context_length)
        )
        inputs.append(row[:-1])
        targets.append(row[1:])
    return inputs, targets


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    os.environ["MIMILLM_BACKEND"] = "cuda"
    reset_backend()
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        tokenizer="byte" if args.vocab_size == 260 else "bpe",
        context_length=args.context_length,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        batch_size=args.batch_size,
        steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=0,
        validation_interval=1,
        checkpoint_interval=1,
        seed=args.seed,
    )
    tokenizer = (
        None if config.tokenizer == "byte"
        else BpeTokenizer.load(args.tokenizer_model)
    )
    model = DecoderTransformer(config, tokenizer_model=tokenizer)
    optimizer = AdamW(model.parameters(), 1e-3, weight_decay=0.0)
    inputs, targets = _batch(config, args.seed + 1)
    trainer = compile_static_cuda_training(
        model, optimizer, inputs, targets,
    )
    try:
        for _ in range(args.warmup):
            trainer.step(inputs, targets, clip_grad_norm=1.0)
        samples = [
            trainer.step(inputs, targets, clip_grad_norm=1.0)
            for _ in range(args.repeats)
        ]
    finally:
        trainer.close()
    seconds = [sample.seconds for sample in samples]
    graph_seconds = [sample.graph_seconds for sample in samples]
    optimizer_seconds = [sample.optimizer_seconds for sample in samples]
    tokens = config.batch_size * config.context_length
    median = statistics.median(seconds)
    mean = statistics.mean(seconds)
    return {
        "backend": "cuda_graph",
        "parameters": model.parameter_count(),
        "tokens": tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "median_seconds": median,
        "mean_seconds": mean,
        "median_tokens_per_second": tokens / median,
        "mean_tokens_per_second": tokens / mean,
        "median_graph_seconds": statistics.median(graph_seconds),
        "median_optimizer_seconds": statistics.median(optimizer_seconds),
        "samples_seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark mimiLLM fixed-shape CUDA Graph training",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-mlp", type=int, default=522)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup cannot be negative")
    if args.vocab_size != 260 and not args.tokenizer_model:
        parser.error("--tokenizer-model is required for BPE vocabularies")
    print(json.dumps(benchmark(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
