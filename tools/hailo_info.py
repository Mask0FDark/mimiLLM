#!/usr/bin/env python3
"""Inspect an optional HailoRT installation and a compiled HEF artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimillm.hailo import inspect_hailo_hef, inspect_hailo_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect HailoRT and optional HEF metadata")
    parser.add_argument("--hef", type=Path, help="path to a compiled .hef file")
    args = parser.parse_args()
    runtime = inspect_hailo_runtime()
    result: dict[str, object] = {"runtime": runtime.to_dict()}
    if args.hef is not None:
        result["hef"] = inspect_hailo_hef(args.hef).to_dict()
    print(json.dumps(result, indent=2))
    return 0 if runtime.available else 1


if __name__ == "__main__":
    raise SystemExit(main())
