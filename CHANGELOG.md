# Changelog / История изменений

## 0.3.2 — 2026-07-13

- Added one user-facing response API for questions, commands, and writing requests using the same model weights.
- QA training now computes loss on answers while keeping requests in the attention context.
- Mixed-length batches use masked padding instead of truncating every sample to the shortest one.
- Added weighted cross-entropy and matching Python/C++ training tests.

- Добавлен единый интерфейс ответа на вопросы, команды и запросы на написание текста с одними весами модели.
- При QA-обучении loss теперь считается по ответу, а запрос остаётся в контексте attention.
- Примеры разной длины дополняются padding с нулевым весом вместо обрезания всего batch до кратчайшего примера.
- Добавлена weighted cross-entropy и проверки обучения на Python/C++ backend.

## 0.3.1 — 2026-07-13

- Windows x64 packages now include the threaded C++ backend automatically.
- A normal pip installation no longer requires a separate compiler or backend build step on Windows.
- Source checkouts prefer a freshly built library from `build/`, while installed packages use the bundled DLL.

- В пакет для Windows x64 теперь автоматически входит многопоточный C++ backend.
- После обычной установки через pip больше не требуется отдельно ставить компилятор и собирать DLL.
- При работе с исходниками используется свежая библиотека из `build/`, а установленный пакет загружает вложенную DLL.

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
