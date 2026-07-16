# Changelog / История изменений

## 0.9.0 — 2026-07-16

### English

- Added `train_pipeline` and the `mimillm-train-pipeline` command for a validated linear `pretrain -> sft` curriculum. Later stages automatically receive the preceding best weights; SFT from scratch and silent output-directory overwrites are rejected by default.
- Added `lineage.json` and `pipeline_state.json` with stage kind, parent weights, effective configuration, model hash, and tokenizer hash. A resumed stage verifies immutable configuration and tokenizer fields before loading its checkpoint.
- Added preflight dataset audits for train/validation leakage, duplicate QA pairs, conflicting answers, cross-stage overlap, and configurable foreign-assistant phrases.
- BPE format v3 reserves atomic pieces for frequent multi-byte Unicode characters before frequency merges. Existing v1 and v2 artifacts remain loadable.
- Added `analyze_tokenizer`, `TokenizerReport`, automatic `tokenizer_report.json`, and configurable pipeline quality gates for compression, Unicode coverage, and exact round trips.
- Generation now masks PAD/BOS/SEP and invalid UTF-8 continuations, and cannot finish in the middle of a multi-byte character.
- Modern JSON configurations that explicitly select tied embeddings now receive the current AdamW/cosine defaults when optional optimizer fields are omitted; legacy untied configurations retain their historical defaults.
- Added a complete bilingual staged-training example and end-to-end regression tests.
- Added optional held-out generation gates for each pipeline stage. They test real deterministic single- and multi-turn responses, save `generation_report.json`, and stop the curriculum when model behavior misses the configured pass rate.

### Русский

- Добавлены `train_pipeline` и команда `mimillm-train-pipeline` для проверенной цепочки `pretrain -> sft`. Каждый следующий этап автоматически получает лучшие веса предыдущего; случайный SFT с нуля и незаметная перезапись каталога весов по умолчанию запрещены.
- Добавлены `lineage.json` и `pipeline_state.json` с типом этапа, родительскими весами, эффективной конфигурацией и хешами модели и токенизатора. Перед продолжением checkpoint проверяются неизменяемые поля конфигурации и токенизатор.
- Добавлен предварительный аудит данных: утечки train/validation, дубликаты QA, вопросы с конфликтующими ответами, пересечения между этапами и настраиваемые фразы чужих ассистентов.
- BPE формата v3 сначала создаёт цельные токены для частых многобайтовых Unicode-символов, а затем выполняет частотные слияния. Артефакты v1 и v2 продолжают загружаться.
- Добавлены `analyze_tokenizer`, `TokenizerReport`, автоматический `tokenizer_report.json` и пороги качества pipeline для сжатия, Unicode-покрытия и точного round trip.
- Генерация теперь маскирует PAD/BOS/SEP и недопустимые UTF-8-продолжения и не может завершиться посреди многобайтового символа.
- Современные JSON-конфигурации с явно включёнными связанными embeddings получают актуальные AdamW/cosine defaults, если необязательные поля оптимизатора пропущены; старые untied-конфигурации сохраняют прежнее поведение.
- Добавлены полный двуязычный пример поэтапного обучения и end-to-end регрессионные тесты.
- Добавлены опциональные проверки реальной генерации после этапа. Они детерминированно проверяют одно- и многоходовые диалоги, сохраняют `generation_report.json` и не запускают следующий этап, если модель не набрала заданную долю успешных сценариев.

## 0.8.0 — 2026-07-16

### English

- Added `init_from` to start a new training stage from compatible reusable weights while resetting AdamW, warmup, the learning-rate schedule, and the step counter.
- Added `tokenizer_path` so multiple curriculum-stage configs can explicitly reuse one BPE artifact.
- Added validation-based early stopping through `early_stopping_patience` and `early_stopping_min_delta`.
- Fixed QA window sampling so long assistant answers are trained beyond their first context-sized chunk.
- Added optional tied input/output token embeddings, enabled for newly constructed configurations while legacy JSON models keep their original output projection.
- Added configurable AdamW betas and epsilon, gradient clipping, and constant/linear/cosine learning-rate schedules. New configurations default to the LLM-oriented AdamW + warmup + cosine stack.
- Added a reproducible one-example QA overfit diagnostic and regression tests for staged initialization and early stopping.
- Documented the distinction between causal pretraining, answer-only supervised fine-tuning, checkpoint resume, and weight initialization.

### Русский

- Добавлен `init_from`: новый этап получает совместимые готовые веса, но начинает с нового AdamW, warmup, расписания learning rate и номера шага.
- Добавлен `tokenizer_path`, чтобы несколько конфигураций поэтапного обучения явно использовали один BPE-артефакт.
- Добавлена ранняя остановка по validation через `early_stopping_patience` и `early_stopping_min_delta`.
- Исправлен выбор QA-окон: длинные ответы ассистента теперь обучаются целиком, а не только до первого окна размером с контекст.
- Добавлены связанные входные и выходные token embeddings. Для новых конфигураций они включены по умолчанию, а старые JSON-модели сохраняют прежнюю отдельную выходную матрицу.
- Добавлены настройки beta и epsilon AdamW, gradient clipping и расписания learning rate `constant`, `linear`, `cosine`. Новые конфигурации по умолчанию используют ориентированный на LLM стек AdamW + warmup + cosine.
- Добавлены воспроизводимая диагностика переобучения на одном QA-примере и регрессионные тесты переноса весов и ранней остановки.
- В документации разделены causal pretraining, SFT только на ответах, продолжение checkpoint и начало нового этапа с готовых весов.

## 0.7.1 — 2026-07-15

### English

- Linux source installs now build and package the native C++ backend automatically when a supported compiler is available, including on Raspberry Pi arm64.
- Added a generic Raspberry Pi setup helper and verified the ARM64 C++ backend on Raspberry Pi 5 with Ubuntu Server 24.04 arm64.

### Русский

- При установке из исходников на Linux нативный C++ backend теперь автоматически собирается и упаковывается при наличии компилятора, включая Raspberry Pi arm64.
- Добавлен обычный установочный скрипт для Raspberry Pi; ARM64 C++ backend проверен на Raspberry Pi 5 с Ubuntu Server 24.04 arm64.

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
