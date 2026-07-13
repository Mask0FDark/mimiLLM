#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda не найдена. Установите Miniconda или Anaconda и повторите запуск."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx mimillm; then
  echo "Окружение mimillm уже существует."
else
  echo "Создаю окружение mimillm только из environment.yml..."
  conda env create -f environment.yml
fi

echo "Устанавливаю mimiLLM в editable-режиме..."
conda run -n mimillm python -m pip install --no-deps --no-build-isolation -e .

echo "Готово. Выполните:"
echo "  conda activate mimillm"
echo "  python tools/build_backend.py --release"
echo "  python -m unittest discover -s tests -v"
