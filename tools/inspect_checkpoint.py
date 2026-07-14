#!/usr/bin/env python3
"""Display safe checkpoint metadata and parameter shapes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimillm.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a mimiLLM training checkpoint")
    parser.add_argument("checkpoint", type=Path, help="path to a training checkpoint")
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
