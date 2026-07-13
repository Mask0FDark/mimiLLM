# Changelog / История изменений

## 0.3.0 — 2026-07-13

- Added standard `config.json` + `model.safetensors` model directories.
- Added `save_model`, an expanded `load_model`, and a dependency-free F32 SafeTensors implementation.
- Added configurable paths for text and question train/validation data.
- Added reusable training APIs and separate train/use-weight examples.

- Добавлены стандартные каталоги моделей с `config.json` и `model.safetensors`.
- Добавлены `save_model`, расширенный `load_model` и независимая F32-реализация SafeTensors.
- Пути к четырём наборам данных теперь задаются в конфигурации.
- Цикл обучения вынесен в библиотеку, добавлены отдельные примеры обучения и загрузки весов.

## 0.2.0 — 2026-07-13

- The project became an installable `mimillm` library with a public API.
- Renamed the Python package, C++ namespace, C ABI, backend files, environment
  variables, and Conda environments to the mimiLLM brand.
- Added `create_model`, `load_model`, `generate_text`, and text-only datasets.
- Added `pyproject.toml`, editable installation, examples, and a bilingual README.
- Introduced the `MIMILLM1` checkpoint header.

- Проект оформлен как устанавливаемая библиотека `mimillm` с публичным API.
- Python-пакет, C++ namespace/C ABI, backend, переменные окружения и Conda
  переименованы в соответствии с брендом mimiLLM.
- Добавлены `create_model`, `load_model`, `generate_text` и text-only датасеты.
- Добавлены `pyproject.toml`, editable-установка, примеры и двуязычный README.
- Новый заголовок checkpoint — `MIMILLM1`.

## 0.1.0 — 2026-07-13

- Implemented a decoder-only Transformer, byte tokenizer, float32 Tensor,
  autograd, SGD, AdamW, checkpoints, mixed datasets, generation, and C++20 CPU
  acceleration.
- Added Windows/Linux build scripts, tests, benchmarks, and training examples.
