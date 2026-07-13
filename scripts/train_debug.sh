#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py --config configs/debug.json --output checkpoints/debug.bin

