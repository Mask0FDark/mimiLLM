#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py --config configs/qa_demo.json --output-dir weights/qa_demo
