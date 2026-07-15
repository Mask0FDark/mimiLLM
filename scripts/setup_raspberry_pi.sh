#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$ROOT/.venv-pi}"

python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install "$ROOT"

cd /tmp
MIMILLM_BACKEND=cpp "$VENV/bin/python" -c \
  'from mimillm import get_backend; backend = get_backend(); print(f"mimiLLM backend={backend.name} threads={backend.num_threads}")'
"$VENV/bin/python" "$ROOT/tools/hailo_info.py" || true
