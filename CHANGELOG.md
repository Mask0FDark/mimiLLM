# Changelog / История изменений

## 0.7.1 — 2026-07-15

### English

- Linux source installs now build and package the native C++ backend automatically when a supported compiler is available, including on Raspberry Pi arm64.
- Added `inspect_hailo_runtime`, `hailo_is_available`, and `inspect_hailo_hef` for optional HailoRT device discovery and ready-HEF metadata inspection.
- Added a Raspberry Pi setup script that keeps access to the system HailoRT package while installing mimiLLM in a venv.
- Documented the tested boundary clearly: Hailo-8 runs compiled HEF artifacts, while mimiLLM SafeTensors currently run through the ARM64 C++ backend.

### Русский

- При установке из исходников на Linux нативный C++ backend теперь автоматически собирается и упаковывается при наличии компилятора, включая Raspberry Pi arm64.
- Добавлены `inspect_hailo_runtime`, `hailo_is_available` и `inspect_hailo_hef` для необязательной проверки устройств HailoRT и метаданных готового HEF.
- Добавлен скрипт установки для Raspberry Pi, который сохраняет доступ venv к системному пакету HailoRT.
- Чётко описана проверенная граница: Hailo-8 запускает скомпилированные HEF, а веса mimiLLM сейчас выполняются через ARM64 C++ backend.

## 0.7.0 — 2026-07-15

### English

- Added a deterministic training-step benchmark for comparing backend and optimizer changes with the same synthetic batch.
- Added one-call `tokenize`, `detokenize`, and `pretokenize` APIs that accept tokenizer names, BPE artifact paths, or tokenizer objects.
- New BPE artifacts use versioned Unicode-aware word, number, symbol, and leading-space segmentation. Existing version 1 artifacts retain their original behavior when loaded.
- The benchmark initializes its backend and model once, warms them up, and then measures consecutive training steps without including repeated CUDA compilation in the result.
- Validation loss now streams validation batches instead of materializing the full batch list before evaluation. This keeps validation memory usage lower on larger datasets.
- Added optional `save_validation_checkpoints` snapshots so every evaluated model can be compared later instead of losing an earlier generation-quality peak.
- Added dialogue `.jsonl` datasets with alternating `user`/`assistant` messages. Conversations are expanded turn by turn so each answer is trained with the preceding chat history in its attention context.
- Added tests that verify validation batch counting and benchmark reproducibility.

### Русский

- Добавлен детерминированный benchmark одного шага обучения, чтобы сравнивать изменения backend и optimizer на одинаковом synthetic batch.
- Добавлены простые функции `tokenize`, `detokenize` и `pretokenize`, которые принимают имя токенизатора, путь к BPE-файлу или готовый объект.
- Новые BPE-артефакты используют версионированное Unicode-aware разбиение слов, чисел, знаков и начальных пробелов. Старые BPE-файлы версии 1 сохраняют прежнее поведение при загрузке.
- Benchmark инициализирует backend и модель один раз, выполняет прогрев, а затем замеряет последовательные шаги обучения, не включая повторную компиляцию CUDA в результат.
- Validation loss теперь проходит validation batches потоково, без предварительного создания полного списка batches. На больших датасетах это снижает расход памяти во время validation.
- Добавлены опциональные снимки `save_validation_checkpoints`, чтобы после обучения можно было сравнить каждую проверенную модель и не потерять более удачный по качеству генерации этап.
- Добавлен диалоговый формат `.jsonl` с чередующимися сообщениями `user`/`assistant`. Разговор разворачивается по ходам, поэтому каждый ответ обучается с предыдущей историей чата в attention-контексте.
- Добавлены тесты для подсчёта validation batches и воспроизводимости benchmark.

## 0.6.0 — 2026-07-14

### English

- Added a trainable byte-level BPE tokenizer with UTF-8 byte fallback. It can learn subword tokens from project train data while still representing arbitrary text without unknown-token failures.
- Added `tokenizer.json` artifacts for BPE models. `save_model` writes the tokenizer next to `model.safetensors`, and `load_model` restores it automatically from a model directory or adjacent `.safetensors` file.
- Added `train_bpe_tokenizer`, `train_tokenizer_from_config`, `load_tokenizer`, and `save_tokenizer` to the public API.
- Added BPE tokenizer, model save/load, and one-step training tests. Existing `byte` and `unicode` models remain compatible.

### Русский

- Добавлен обучаемый byte-level BPE токенизатор с UTF-8 byte fallback. Он учит subword-токены на train-данных проекта и при этом может представить любой текст без неизвестных токенов.
- Для BPE-моделей добавлен artifact `tokenizer.json`. `save_model` сохраняет его рядом с `model.safetensors`, а `load_model` автоматически восстанавливает токенизатор из папки модели или рядом с `.safetensors`.
- В публичный API добавлены `train_bpe_tokenizer`, `train_tokenizer_from_config`, `load_tokenizer` и `save_tokenizer`.
- Добавлены тесты BPE-токенизации, сохранения/загрузки модели и короткого обучения. Старые модели с `byte` и `unicode` остаются совместимыми.

## 0.5.0 — 2026-07-14

### English

- Added the optional `unicode` tokenizer with a fixed 355-token vocabulary. Common Cyrillic characters use one token, while arbitrary text remains reversible through the UTF-8 byte fallback.
- Models now store their tokenizer choice in `config.json`, and generation automatically uses the tokenizer attached to the loaded model. Existing 260-token `byte` models remain compatible.
- Added `qa_prompt_weight`, `qa_answer_prefix_weight`, and `qa_answer_prefix_tokens` training controls. They can strengthen request representations and the first tokens that select a specific answer.
- Updated the bilingual documentation and QA/mixed example configurations, and added tokenizer, model-shape, and weighted-dataset tests.

### Русский

- Добавлен опциональный токенизатор `unicode` с фиксированным словарём из 355 токенов. Частые кириллические символы занимают один токен, а произвольный текст остаётся обратимым благодаря UTF-8 fallback.
- Выбор токенизатора теперь хранится в `config.json`, а генерация автоматически использует токенизатор загруженной модели. Старые `byte`-модели со словарём 260 токенов остаются совместимыми.
- Добавлены настройки обучения `qa_prompt_weight`, `qa_answer_prefix_weight` и `qa_answer_prefix_tokens`. Они позволяют усилить представление запроса и первые токены, определяющие конкретный ответ.
- Обновлены двуязычная документация и примеры QA/mixed; добавлены тесты токенизации, формы модели и весов QA-датасета.

## 0.4.0 — 2026-07-14

### English

- Added a complete NVIDIA CUDA backend for forward passes, autograd, attention, embeddings, softmax, masked cross-entropy, gradient clipping, and AdamW.
- CUDA kernels are compiled at runtime with NVRTC and loaded through the NVIDIA Driver API, so Windows does not require Visual Studio, `cl.exe`, PyTorch, or TensorFlow.
- `auto` now selects backends in the order CUDA, threaded C++, then Python; `MIMILLM_BACKEND` and the training API can force a specific backend.
- CUDA device name and VRAM are shown in the training header.
- Added matching CUDA numerical tests and CUDA benchmark output.

### Русский

- Добавлен полный NVIDIA CUDA backend для forward, autograd, attention, embedding, softmax, masked cross-entropy, gradient clipping и AdamW.
- CUDA-ядра компилируются во время запуска через NVRTC и загружаются через NVIDIA Driver API, поэтому на Windows не нужны Visual Studio, `cl.exe`, PyTorch или TensorFlow.
- Режим `auto` теперь выбирает backend в порядке CUDA, многопоточный C++, затем Python; конкретный backend можно задать через `MIMILLM_BACKEND` или API обучения.
- В заголовке обучения показываются название видеокарты и объём VRAM.
- Добавлены численные тесты CUDA и вывод CUDA в benchmark.

## 0.3.4 — 2026-07-14

- Library examples and command-line tools now use clear English help text.
- The weight-loading example explains deterministic generation, temperature, and top-k sampling.
- Sampling now defaults to `top_k=20` in the example, while deterministic generation remains the default with `temperature=0`.

- Примеры и консольные инструменты библиотеки теперь используют понятную английскую справку.
- Пример загрузки весов объясняет детерминированную генерацию, temperature и top-k sampling.
- В примере sampling использует `top_k=20`, а режим по умолчанию остаётся детерминированным благодаря `temperature=0`.

## 0.3.3 — 2026-07-14

- Validation now covers every supervised QA and text token instead of a single batch.
- Long validation runs report progress for every QA and text batch.
- The model with the lowest validation loss is kept in the main weights directory; final-step weights are stored in `last/`.
- Added persistent best-validation metadata so resumed training does not overwrite a better model.

- Validation теперь проверяет все supervised-токены QA и текстов вместо одного batch.
- Долгая validation показывает прогресс каждого QA- и text-batch.
- В основной папке весов сохраняется модель с минимальным validation loss, а веса последнего шага находятся в `last/`.
- Метаданные лучшей validation сохраняются между продолжениями обучения и защищают лучшую модель от перезаписи.

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
