#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export MINILLM_BACKEND="${MINILLM_BACKEND:-cpp}"
export MINILLM_NUM_THREADS="${MINILLM_NUM_THREADS:-4}"
python train.py --config configs/mixed_demo.json --output checkpoints/mixed_demo.bin "$@"
