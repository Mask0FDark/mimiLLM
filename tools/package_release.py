#!/usr/bin/env python3
"""Создаёт исходный ZIP-релиз без checkpoint, build и кэшей."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", "build", "__pycache__", "checkpoints", "logs", "dist"}
EXCLUDED_NAMES = {"hm.txt"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Упаковка исходников mimiLLM")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "mimiLLM-source.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if (
                path.is_file()
                and path.name not in EXCLUDED_NAMES
                and not any(part in EXCLUDED_PARTS for part in relative.parts)
                and not any(part.endswith(".egg-info") for part in relative.parts)
            ):
                archive.write(path, Path("mimiLLM") / relative)
    print(f"Создан релиз: {args.output}")


if __name__ == "__main__":
    main()
