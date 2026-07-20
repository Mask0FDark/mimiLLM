#!/usr/bin/env python3
"""Detect obvious host-memory growth during a repeated tiny training workload."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimillm.backend import reset_backend
from mimillm.checkpoint import save_checkpoint
from mimillm.generation import generate
from mimillm.optim import AdamW
from mimillm.tensor import no_grad
from mimillm.transformer import DecoderTransformer, TransformerConfig


def _rss_bytes() -> int:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _linear_slope(points: list[dict[str, int]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [point["step"] for point in points]
    ys = [point["rss_bytes"] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return 0.0 if denominator == 0.0 else sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny train/validation/generation/checkpoint RSS regression",
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "cpp", "python"), default="python")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--sample-interval", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("memory_regression.json"))
    args = parser.parse_args()
    if args.steps < 20 or args.sample_interval <= 0:
        parser.error("steps must be at least 20 and sample-interval must be positive")

    os.environ["MIMILLM_BACKEND"] = args.backend
    reset_backend()
    config = TransformerConfig(
        context_length=8, d_model=8, n_layers=1, n_heads=1, d_mlp=16,
        batch_size=1, steps=args.steps, learning_rate=0.001,
        weight_decay=0.0, warmup_steps=0, validation_interval=10,
        checkpoint_interval=10, seed=123,
    )
    model = DecoderTransformer(config)
    optimizer = AdamW(model.parameters(), config.learning_rate, weight_decay=0.0)
    inputs = [[model.tokenizer.BOS, 10, 11, 12, 13, 14, 15, 16]]
    targets = [10, 11, 12, 13, 14, 15, 16, model.tokenizer.EOS]
    samples: list[dict[str, int]] = []
    with tempfile.TemporaryDirectory(prefix="mimillm-memory-") as directory:
        checkpoint = Path(directory) / "checkpoint.bin"
        for step in range(1, args.steps + 1):
            optimizer.zero_grad()
            loss = model(inputs).reshape(-1, config.vocab_size).cross_entropy(targets)
            loss.backward()
            optimizer.clip_grad_norm(config.gradient_clip_norm)
            optimizer.step()
            del loss
            if step % args.sample_interval == 0 or step == args.steps:
                with no_grad():
                    validation = model(inputs).reshape(
                        -1, config.vocab_size,
                    ).cross_entropy(targets).item()
                    generated = generate(
                        model, inputs[0], max_new_tokens=4,
                        temperature=0.0, top_k=1,
                    )
                save_checkpoint(
                    checkpoint, model, optimizer, config=config.to_dict(),
                    step=step, seed=config.seed,
                )
                del generated
                gc.collect()
                samples.append({
                    "step": step,
                    "rss_bytes": _rss_bytes(),
                    "validation_microloss": int(validation * 1_000_000),
                })

    warmup_step = max(args.sample_interval, args.steps // 5)
    stable = [point for point in samples if point["step"] >= warmup_step]
    slope = _linear_slope(stable)
    net_growth = stable[-1]["rss_bytes"] - stable[0]["rss_bytes"]
    allowance = max(64 * 1024 * 1024, stable[0]["rss_bytes"] // 4)
    projected_growth = max(0.0, slope * (stable[-1]["step"] - stable[0]["step"]))
    passed = not (net_growth > allowance and projected_growth > allowance)
    report = {
        "format": "mimiLLM-memory-regression-v1",
        "passed": passed,
        "backend": args.backend,
        "steps": args.steps,
        "sample_interval": args.sample_interval,
        "warmup_step": warmup_step,
        "net_growth_bytes": net_growth,
        "allowed_growth_bytes": allowance,
        "slope_bytes_per_step": slope,
        "projected_growth_bytes": projected_growth,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in (
        "passed", "backend", "steps", "net_growth_bytes",
        "allowed_growth_bytes", "slope_bytes_per_step",
    )}, indent=2))
    print(f"report={args.output.resolve()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
