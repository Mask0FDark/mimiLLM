#!/usr/bin/env python3
"""Runs the library training workflow with the repository debug config."""

from pathlib import Path

from mimillm.training import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    main(root / "configs" / "debug.json", root / "weights")
