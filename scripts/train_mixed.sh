#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export MIMILLM_BACKEND="${MIMILLM_BACKEND:-cpp}"
export MIMILLM_NUM_THREADS="${MIMILLM_NUM_THREADS:-4}"
python train.py --config configs/mixed_demo.json --output-dir weights/mixed_demo "$@"
