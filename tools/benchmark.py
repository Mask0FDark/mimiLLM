#!/usr/bin/env python3
"""Измеряет реальные CPU-времена Python/C++ kernels и маленькой модели."""

from __future__ import annotations

import argparse
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimillm import backend_python
from mimillm.backend_cpp import CppBackend, is_available
from mimillm.optim import AdamW
from mimillm.transformer import DecoderTransformer, TransformerConfig


def measure(function, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mimiLLM")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rng = random.Random(42)
    left = [rng.uniform(-1, 1) for _ in range(args.size * args.size)]
    right = [rng.uniform(-1, 1) for _ in range(args.size * args.size)]
    python_time = measure(
        lambda: backend_python.matmul(left, right, args.size, args.size, args.size),
        args.repeats,
    )
    print(f"platform={platform.system()} arch={platform.machine()} python={sys.version.split()[0]}")
    print(f"python_matmul size={args.size} time={python_time:.6f}s")
    if is_available():
        cpp = CppBackend()
        cpp.set_num_threads(1)
        single = measure(lambda: cpp.matmul(left, right, args.size, args.size, args.size), args.repeats)
        cpp.set_num_threads(args.threads)
        multi = measure(lambda: cpp.matmul(left, right, args.size, args.size, args.size), args.repeats)
        print(f"compiler={cpp.compiler_info}")
        print(f"cpp_matmul threads=1 time={single:.6f}s speedup={python_time / single:.2f}x")
        print(f"cpp_matmul threads={cpp.num_threads} time={multi:.6f}s speedup={python_time / multi:.2f}x")
    else:
        print("cpp_matmul: backend не собран")

    config = TransformerConfig(
        context_length=8, d_model=8, n_layers=1, n_heads=2, d_mlp=16,
        batch_size=1, steps=1, validation_interval=1, checkpoint_interval=1,
    )
    model = DecoderTransformer(config)
    inputs = [[257, 10, 11, 12, 13, 14, 15, 16]]
    forward = measure(lambda: model(inputs), 1)
    optimizer = AdamW(model.parameters(), 1e-3)
    started = time.perf_counter()
    loss = model(inputs).reshape(-1, 260).cross_entropy([10, 11, 12, 13, 14, 15, 16, 258])
    loss.backward()
    optimizer.step()
    training = time.perf_counter() - started
    print(f"model_forward parameters={model.parameter_count()} time={forward:.6f}s")
    print(f"training_step tokens=8 time={training:.6f}s")


if __name__ == "__main__":
    main()
