# Changelog / История изменений

## 0.6.0.dev0 — CPU+GPU experiment

### English

- Added opt-in CPU+GPU data-parallel training on the `experiment/cpu-gpu` branch.
- A CUDA model and a threaded C++ replica process separate batch shards concurrently; gradients are combined by supervised-token weight before one AdamW update.
- Added thread-local backend selection and safe CUDA-context activation from worker threads.
- Added `backend="hybrid"`, CPU batch/thread controls, training progress details, and numerical gradient tests; a complete hybrid training smoke test was also performed.
- Benchmarks on the development RTX 3050 Laptop showed that hybrid execution is workload-dependent: some fixed long batches improved by about 10%, while repeated mixed-length m0fdii runs were about 6–11% slower than CUDA alone. The mode remains experimental and is not selected by `auto`.

### Русский

- В ветке `experiment/cpu-gpu` добавлено опциональное data-parallel обучение на CPU и GPU одновременно.
- CUDA-модель и копия на многопоточном C++ обрабатывают разные части batch; градиенты объединяются по весу supervised-токенов перед одним шагом AdamW.
- Добавлены потоко-локальный выбор backend и безопасная активация CUDA-контекста из рабочих потоков.
- Добавлены `backend="hybrid"`, настройки CPU-части batch и числа потоков, вывод параметров в прогрессе и численные тесты градиентов; также выполнен полный smoke-test обучения.
- Измерения на RTX 3050 Laptop показали зависимость от нагрузки: на некоторых фиксированных длинных batch получено около 10% ускорения, а повторяемые прогоны по batch m0fdii разной длины оказались примерно на 6–11% медленнее чистой CUDA. Режим остаётся экспериментальным и не включается через `auto`.

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
