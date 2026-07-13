#!/usr/bin/env python3
"""Показывает безопасные metadata и формы checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minillm.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Инспекция checkpoint m0fdii")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    data = load_checkpoint(args.checkpoint)
    summary = {
        "step": data.step, "seed": data.seed, "config": data.config,
        "parameters": {name: tensor.shape for name, tensor in data.parameters.items()},
        "optimizer": None if data.optimizer_state is None else {
            key: value for key, value in data.optimizer_state.items()
            if key not in {"first_moments", "second_moments"}
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
