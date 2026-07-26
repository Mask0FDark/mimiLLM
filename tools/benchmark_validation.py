"""Benchmark eager or CUDA Graph validation on an existing mimiLLM model."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("cuda", "cpp", "python"), default="cuda",
    )
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    os.environ["MIMILLM_BACKEND"] = args.backend

    from mimillm import load_model
    from mimillm.training import _datasets, validation_loss

    model = load_model(args.weights.resolve())
    config = replace(
        model.config,
        cuda_graph_validation=not args.eager,
        batch_size=(
            args.batch_size
            if args.batch_size is not None else model.config.batch_size
        ),
    )
    _, dataset = _datasets(
        config, args.base_dir.resolve(), model.tokenizer,
    )
    supervised_tokens = 0.0
    for source, _ in dataset.source_weights():
        for _, _, weights in dataset.validation_batches(
            config.batch_size, config.context_length, source=source,
        ):
            supervised_tokens += sum(sum(row) for row in weights)

    durations: list[float] = []
    losses: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        losses.append(validation_loss(model, dataset, config))
        durations.append(time.perf_counter() - started)
    median_seconds = statistics.median(durations)
    print(
        f"mode={'eager' if args.eager else 'static'} | "
        f"loss={losses[-1]:.8f} | tokens={supervised_tokens:.0f} | "
        f"median={median_seconds:.4f}s | "
        f"tok/s={supervised_tokens / median_seconds:,.1f}"
    )
    print(
        "runs=" + ", ".join(f"{seconds:.4f}s" for seconds in durations)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
