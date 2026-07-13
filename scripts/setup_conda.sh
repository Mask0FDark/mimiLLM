#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda не найдена. Установите Miniconda или Anaconda и повторите запуск."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx minillm; then
  echo "Окружение minillm уже существует."
else
  echo "Создаю окружение minillm только из environment.yml..."
  conda env create -f environment.yml
fi

echo "Готово. Выполните:"
echo "  conda activate minillm"
echo "  python tools/build_backend.py --release"
echo "  python -m unittest discover -s tests -v"

