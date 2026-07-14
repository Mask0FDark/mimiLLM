# mimiLLM

mimiLLM — открытая Python-библиотека для создания, обучения и исследования decoder-only языковых моделей.

Внутри есть собственные тензоры на `float32`, автоматическое вычисление градиентов, слои нейронной сети, causal attention, decoder-only Transformer, AdamW, токенизатор, обучение и сохранение весов. Всё это написано без NumPy, PyTorch, TensorFlow и других ML-фреймворков. Вычисления могут выполняться на NVIDIA GPU через CUDA, в многопоточном C++ или на чистом Python.

Архитектура, размеры модели, источники данных и параметры обучения задаются пользователем. Библиотеку можно использовать как основу собственного проекта, а её компактная реализация позволяет проследить весь путь от входного текста до обновления весов.

[English version](#english-version)

## Что с ней можно сделать

- создать decoder-only Transformer своего размера;
- обучить его на обычных текстах, вопросах и ответах или на смеси этих данных;
- сохранить модель в папку с `config.json` и `model.safetensors`;
- загрузить готовые веса одной функцией;
- продолжить прерванное обучение из checkpoint;
- переключаться между CUDA, многопоточным C++ и понятным Python backend;
- открыть реализацию любого шага — от умножения тензоров до attention и AdamW.

Если хочется увидеть готовый проект модели, посмотрите [m0fdii](https://github.com/Mask0FDark/m0fdii). Там уже лежат данные, конфигурация, обученные веса и короткие скрипты запуска.

## Установка

Требуется Python 3.12 или новее.

Установка последней версии с GitHub:

```powershell
python -m pip install "git+https://github.com/Mask0FDark/mimiLLM.git"
```

На Windows x64 многопоточный C++ backend входит в пакет и устанавливается этой же командой автоматически. Если в системе есть NVIDIA GPU, драйвер и CUDA Toolkit с NVRTC, библиотека также автоматически включает CUDA backend. Visual Studio и `cl.exe` для него не нужны.

Для разработки самой библиотеки:

```bash
git clone https://github.com/Mask0FDark/mimiLLM.git
cd mimiLLM
python -m pip install -e .
```

Можно создать отдельное окружение Conda:

```powershell
conda create -n my-llm python=3.12 -y
conda run -n my-llm python -m pip install "git+https://github.com/Mask0FDark/mimiLLM.git"
```

Для CPU-режимов сторонние Python runtime-зависимости не нужны. GPU-режим требует установленный NVIDIA CUDA Toolkit с NVRTC — от этой системной зависимости CUDA backend не скрывается. Без CUDA библиотека автоматически продолжит работу через C++ или Python.

## Загрузка готовых весов

Обычная папка модели выглядит так:

```text
weights/
├── config.json
└── model.safetensors
```

Загрузка и использование модели:

```python
from mimillm import generate_response, load_model

model = load_model("weights")
print(generate_response(model, "Кто ты?", max_new_tokens=100))
print(generate_response(model, "Напиши короткий рассказ", max_new_tokens=200))
```

Это одна модель и один набор весов. Любая введённая строка считается запросом
к LLM. Обычные тексты из `data/text` используются при обучении языка, а не
включают отдельный пользовательский режим продолжения.

Передавать можно как папку, так и сам файл:

```python
model = load_model("weights/model.safetensors")
```

Во втором случае рядом с весами всё равно должен находиться `config.json`: по нему библиотека узнаёт размеры embedding, количество слоёв, длину контекста и остальные параметры архитектуры.

Запускаемый пример: [examples/use_weights.py](examples/use_weights.py).

## Обучение своей модели

В отдельном проекте удобно держать конфигурацию, данные и скрипт запуска рядом:

```text
my_model/
├── config.json
├── train.py
└── data/
    ├── text/
    │   ├── train/
    │   └── validation/
    └── question/
        ├── train/
        └── validation/
```

Это не жёстко зашитая структура. Все четыре пути задаются в конфигурации и могут вести в другое место, в том числе по абсолютному пути.

### Обычные тексты

В `data/text/train` кладутся документы, на которых модель изучает язык через предсказание следующего токена. Поддерживаются UTF-8 файлы `.txt`, `.md` и `.text`; вложенные каталоги тоже просматриваются.

В `data/text/validation` должны лежать другие документы. Они не обновляют веса, а помогают увидеть, учится ли модель работать с незнакомым текстом или просто запоминает train-набор.

### Вопросы и ответы

В каталогах `data/question/train` и `data/question/validation` находятся `.txt` файлы с блоками такого вида:

```text
Вопрос: Что делает attention?
Ответ: Attention помогает модели учитывать другие токены в контексте.

Вопрос: What is a token?
Ответ: A token is a unit processed by a language model.
```

Вопрос занимает первую строку блока. Ответ может продолжаться на следующих строках до пустой строки.

### Конфигурация

Небольшой рабочий `config.json`:

```json
{
  "vocab_size": 355,
  "tokenizer": "unicode",
  "context_length": 96,
  "d_model": 64,
  "n_layers": 2,
  "n_heads": 4,
  "d_mlp": 192,
  "batch_size": 1,
  "batches_per_epoch": null,
  "steps": 2500,
  "learning_rate": 0.0008,
  "weight_decay": 0.01,
  "warmup_steps": 20,
  "validation_interval": 25,
  "checkpoint_interval": 50,
  "seed": 42,
  "text_ratio": 0.35,
  "qa_prompt_weight": 0.1,
  "qa_answer_prefix_weight": 3.0,
  "qa_answer_prefix_tokens": 24,
  "text_train_path": "data/text/train",
  "text_validation_path": "data/text/validation",
  "question_train_path": "data/question/train",
  "question_validation_path": "data/question/validation"
}
```

Главные параметры архитектуры:

- `tokenizer` выбирает `byte` или `unicode`; ему должен соответствовать `vocab_size` 260 или 355;
- `context_length` — сколько последних токенов помещается в контекст;
- `d_model` — размер внутреннего представления токена;
- `n_layers` — количество Transformer-блоков;
- `n_heads` — число attention-голов;
- `d_mlp` — размер скрытой части feed-forward слоя.

Параметры обучения:

- `batch_size` задаёт число примеров в одном batch, а `steps` — общее число обновлений весов; большой batch обрабатывает больше токенов за шаг, поэтому скорость сравнивают по `tok/s`, а не по длительности одного шага;
- `batches_per_epoch` задаёт длину эпохи; при `null` она рассчитывается по данным автоматически;
- `learning_rate`, `weight_decay` и `warmup_steps` управляют AdamW;
- `validation_interval` задаёт частоту проверки validation loss;
- `checkpoint_interval` задаёт частоту сохранения состояния;
- `seed` делает инициализацию и выбор batch воспроизводимыми.

`text_ratio` управляет смешиванием источников. При `0.0` модель учится только на вопросах и ответах, при `1.0` — только на обычных текстах. Значение `0.35` означает, что примерно 35% batch будут текстовыми.

В QA-batch вопрос всегда остаётся в attention-контексте. `qa_prompt_weight` задаёт его долю в loss. `qa_answer_prefix_tokens` выбирает начало ответа, а `qa_answer_prefix_weight` усиливает его. Это полезно для маленькой модели: первые слова чаще всего определяют, какой именно ответ был выбран. Значения по умолчанию `0.0`, `0` и `1.0` сохраняют прежнее поведение.

Относительные пути к данным считаются от каталога, в котором лежит `config.json`. Поэтому запуск не зависит от текущей папки терминала.

### Скрипт обучения

Минимальный `train.py` состоит из нескольких строк:

```python
from pathlib import Path

from mimillm import train_from_config


HERE = Path(__file__).resolve().parent
result = train_from_config(
    HERE / "config.json",
    output_dir=HERE / "weights",
    backend="auto",
)

print(result.weights_dir)
```

После `python train.py` в папке `weights` появятся:

```text
weights/
├── config.json
├── model.safetensors
├── best_validation.json
├── training_checkpoint.bin
└── last/
    ├── config.json
    └── model.safetensors
```

Главные `config.json` и `model.safetensors` содержат веса с минимальным validation loss — именно их загружает `load_model("weights")`. В `last` сохраняются веса последнего шага, даже если к этому моменту модель начала переобучаться. `training_checkpoint.bin` дополнительно содержит moments AdamW, номер шага и seed и нужен для возобновления обучения.

Validation проходит по всем проверочным ответам и текстам. Во время долгой проверки в терминале отдельно показывается прогресс `val-qa` и `val-text`.

```python
result = train_from_config(
    HERE / "config.json",
    output_dir=HERE / "weights",
    resume=HERE / "weights" / "training_checkpoint.bin",
)
```

Полный короткий пример находится в [examples/train_model.py](examples/train_model.py).

## Создание модели прямо в коде

Конфигурацию необязательно читать из JSON:

```python
from mimillm import ModelConfig, create_model, save_model

config = ModelConfig(
    context_length=128,
    d_model=96,
    n_layers=3,
    n_heads=6,
    d_mlp=288,
)

model = create_model(config)
print(f"Параметров: {model.parameter_count():,}")
save_model("weights", model)
```

Сразу после создания веса случайные. Чтобы модель генерировала осмысленный текст, её ещё нужно обучить.

## Как устроена библиотека

Код разделён по небольшим модулям, чтобы путь данных можно было проследить без прыжков по огромному фреймворку:

```text
mimillm/tensor.py          float32 Tensor и математические операции
mimillm/autograd.py        граф вычислений и backward
mimillm/layers.py          Linear, Embedding, RMSNorm и MLP
mimillm/attention.py       causal multi-head self-attention
mimillm/transformer.py     TransformerConfig и decoder-only модель
mimillm/optim.py           SGD и AdamW
mimillm/dataset.py         тексты, вопросы и формирование batch
mimillm/training.py        полный цикл обучения
mimillm/generation.py      авторегрессионная генерация
mimillm/safetensors.py     чтение и запись переносимых весов
mimillm/checkpoint.py      checkpoint для продолжения обучения
```

В библиотеке есть два фиксированных токенизатора. `byte` имеет словарь из 260 токенов и кодирует весь текст байтами UTF-8. `unicode` имеет 355 токенов: частые кириллические буквы и знаки занимают один токен, а всё остальное обратимо кодируется теми же UTF-8-байтами. Для русского текста это делает последовательности почти вдвое короче и не позволяет повторяющемуся первому UTF-8-байту искусственно улучшать loss.

Выбор токенизатора хранится в `config.json`, и `load_model` автоматически использует его при генерации. Менять `tokenizer` или `vocab_size` уже после обучения нельзя: формы embedding и выходного слоя будут другими, поэтому нужно начать новое обучение.

Transformer использует pre-norm блоки, causal mask, обучаемые позиционные embedding и отдельную выходную проекцию. Это настоящая авторегрессионная модель: во время генерации она много раз вычисляет logits и каждый раз выбирает следующий токен.

## Веса и checkpoint — не одно и то же

`model.safetensors` хранит только именованные `F32` тензоры модели. Формат состоит из JSON-заголовка и непрерывного бинарного буфера и совместим с документированной [спецификацией SafeTensors](https://github.com/huggingface/safetensors#format). mimiLLM читает и записывает его собственной реализацией без внешней зависимости.

`training_checkpoint.bin` — внутренний формат mimiLLM. Кроме параметров модели он содержит состояние оптимизатора и служебную информацию, поэтому занимает больше места. Для публикации готовой модели достаточно `config.json` и `model.safetensors`.

## CUDA, C++ и Python backend

По умолчанию используется режим `auto`. Библиотека выбирает CUDA, затем многопоточный C++, затем Python. Архитектура модели и формат весов при этом не меняются, поэтому обучение можно продолжить на другом backend.

### CUDA

Для GPU-режима нужны:

- NVIDIA GPU и совместимый драйвер;
- NVIDIA CUDA Toolkit с библиотекой NVRTC;
- переменная `CUDA_PATH` или доступный `nvcc`, чтобы библиотека нашла NVRTC и заголовки CUDA.

На Windows Visual Studio, `cl.exe`, PyTorch и TensorFlow не требуются. Ядра из `cuda/kernels.cu` компилируются NVRTC при первом выборе CUDA в новом процессе и напрямую загружаются через NVIDIA Driver API.

Принудительный запуск на GPU из PowerShell:

```powershell
$env:MIMILLM_BACKEND = "cuda"
python train.py
```

Или прямо через Python API:

```python
result = train_from_config("config.json", output_dir="weights", backend="cuda")
```

Если оставить `backend="auto"`, CUDA будет выбрана сама. В начале обучения появятся `backend=cuda`, название GPU и объём VRAM. Чтобы временно запретить автоматический выбор GPU, можно задать `MIMILLM_DISABLE_CUDA=1`.

CUDA backend реализует forward и backward операции, matmul, attention, softmax, embedding, masked cross-entropy, gradient clipping и AdamW. Граф autograd и цикл обучения остаются общими для всех backend.

### Экспериментальный CPU+GPU

В ветке `experiment/cpu-gpu` есть data-parallel режим `hybrid`. Он держит две одинаковые копии модели, считает большую часть batch на CUDA, один пример — в C++ CPU, затем объединяет градиенты с учётом числа supervised-токенов. AdamW по-прежнему делает одно обновление, поэтому формат весов и checkpoint не меняется.

PowerShell:

```powershell
$env:MIMILLM_BACKEND = "hybrid"
$env:MIMILLM_HYBRID_CPU_BATCH = "1"
$env:MIMILLM_HYBRID_CPU_THREADS = "4"
python train.py
```

Тот же режим через API:

```python
train_from_config(
    "config.json",
    output_dir="weights",
    backend="hybrid",
    hybrid_cpu_batch_size=1,
    hybrid_cpu_threads=4,
)
```

Режим требует одновременно CUDA и C++ backend, а `batch_size` должен быть не меньше 2. Больше CPU-потоков не всегда быстрее: они конкурируют с CUDA за память и за время Python-потока. На RTX 3050 Laptop фиксированный длинный batch иногда ускорялся примерно на 10%, но повторяемые прогоны по реальным batch m0fdii разной длины оказались примерно на 6–11% медленнее чистой CUDA. Поэтому это опциональный эксперимент, а не новый режим `auto`.

Перед долгим запуском можно проверить оба режима на настоящих batch своего проекта:

```powershell
python tools/benchmark_hybrid.py E:\m0fdii\config.json --batches 6
```

### C++ и Python

Установка через pip на Windows x64 содержит готовую C++ DLL. Ручная сборка нужна только при изменении C++-исходников:

```powershell
python tools/build_backend.py --release
$env:MIMILLM_BACKEND = "cpp"
```

Linux или WSL:

```bash
python3 tools/build_backend.py --release
export MIMILLM_BACKEND=cpp
```

Доступные переменные окружения:

- `MIMILLM_BACKEND=auto|hybrid|cuda|cpp|python` — выбор backend;
- `MIMILLM_DISABLE_CUDA=1` — пропустить CUDA только при автоматическом выборе;
- `MIMILLM_NUM_THREADS=N` — число рабочих потоков C++;
- `MIMILLM_CPP_LIBRARY=/path/to/library` — явный путь к DLL или `.so`.

C++ backend подключается через стабильный C ABI. Python backend является читаемой эталонной реализацией и работает без нативной сборки.

## Проверки

Все тесты на Python backend:

```powershell
$env:MIMILLM_BACKEND = "python"
python -m unittest discover -s tests -v
```

С C++ backend:

```powershell
python tools/build_backend.py --release
$env:MIMILLM_BACKEND = "cpp"
python -m unittest discover -s tests -v
```

С CUDA backend:

```powershell
$env:MIMILLM_BACKEND = "cuda"
python -m unittest discover -s tests -v
python tools/benchmark.py
```

Тесты проверяют тензорные операции, численные градиенты, autograd, слои, attention, оптимизаторы, checkpoint, SafeTensors, генерацию, работу с данными и короткий полный цикл обучения. CUDA-операции отдельно сравниваются с эталонным Python backend.

## English version

mimiLLM is an open Python library for building, training, and exploring decoder-only language models. It implements float32 tensors, autograd, neural-network layers, causal attention, a decoder-only Transformer, AdamW, datasets, training, generation, checkpoints, and SafeTensors weights without NumPy, PyTorch, or another ML runtime. Computation can run on NVIDIA CUDA, threaded C++, or pure Python.

Model dimensions, datasets, and training parameters are controlled by the user. The compact implementation makes the complete path from input text to updated weights available for inspection and extension.

### Install

Python 3.12 or newer is required:

```bash
python -m pip install "git+https://github.com/Mask0FDark/mimiLLM.git"
```

For editable local development:

```bash
git clone https://github.com/Mask0FDark/mimiLLM.git
cd mimiLLM
python -m pip install -e .
```

### Load a model

```python
from mimillm import generate_response, load_model

model = load_model("weights")
print(generate_response(model, "Who are you?", max_new_tokens=100))
print(generate_response(model, "Write a short story", max_new_tokens=200))
```

Both calls use the same model and the same weights. Plain text corpora teach
language during training; they do not create a separate user-facing mode.

The `weights` directory contains `config.json` and `model.safetensors`. You may also pass the `.safetensors` path directly when the config file is next to it.

### Train on your data

Create four data directories:

```text
data/text/train/
data/text/validation/
data/question/train/
data/question/validation/
```

Text directories accept recursive UTF-8 `.txt`, `.md`, and `.text` documents. Question directories contain `.txt` files made of `Вопрос:` and `Ответ:` blocks. Set their paths and `text_ratio` in `config.json`, then call:

```python
from pathlib import Path
from mimillm import train_from_config

HERE = Path(__file__).resolve().parent
result = train_from_config(
    HERE / "config.json", output_dir=HERE / "weights", backend="auto",
)
```

Relative data paths are resolved from the config directory. The root `config.json` and `model.safetensors` keep the lowest-validation-loss model, `last/` keeps the final-step weights, and `training_checkpoint.bin` stores the AdamW state for resuming. Validation covers every supervised token and reports separate `val-qa` and `val-text` batch progress.

See [examples/train_model.py](examples/train_model.py), [examples/use_weights.py](examples/use_weights.py), and the complete [m0fdii model project](https://github.com/Mask0FDark/m0fdii).

### Tokenization and QA loss

`tokenizer: "byte"` uses a 260-token vocabulary and can represent any UTF-8 text. `tokenizer: "unicode"` uses a 355-token vocabulary: common Cyrillic characters have one token, while every other character still has a reversible UTF-8 byte fallback. The Unicode mode gives Russian text much shorter sequences and a more meaningful loss. The selected tokenizer is stored in `config.json` and is used automatically after `load_model`.

For a Unicode model, set both fields together:

```json
{
  "tokenizer": "unicode",
  "vocab_size": 355,
  "qa_prompt_weight": 0.1,
  "qa_answer_prefix_weight": 3.0,
  "qa_answer_prefix_tokens": 24
}
```

The QA prompt always remains visible to attention. `qa_prompt_weight` optionally includes it in the language-model loss, while the two answer-prefix settings give the first answer tokens more influence. Defaults preserve the original answer-only objective. Changing the tokenizer or vocabulary size requires training new weights because the embedding and output shapes change.

### Backends and tests

The default `auto` mode selects CUDA first, then threaded C++, then pure Python. All backends use the same architecture, public API, checkpoints, and SafeTensors weights.

On Windows x64, the threaded C++ backend is bundled with the package and installed automatically by pip; no separate compiler or build command is required.

CUDA mode requires an NVIDIA GPU, a compatible driver, and the NVIDIA CUDA Toolkit with NVRTC. This system dependency is explicit; CPU modes do not require it. Visual Studio, `cl.exe`, PyTorch, and TensorFlow are not required. NVRTC compiles the bundled kernels when CUDA is first selected in a process, and the NVIDIA Driver API loads them directly.

The experimental `experiment/cpu-gpu` branch also provides `backend="hybrid"`. It runs one model replica on CUDA and another on the threaded C++ backend, splits each batch between them, and combines gradients by supervised-token weight before one AdamW update. The defaults assign one sample and four threads to CPU; they can be changed with `hybrid_cpu_batch_size`, `hybrid_cpu_threads`, `MIMILLM_HYBRID_CPU_BATCH`, and `MIMILLM_HYBRID_CPU_THREADS`.

This mode is hardware- and batch-shape-dependent. On the tested RTX 3050 Laptop system it was about 10% faster for some fixed long batches, but repeated runs over real mixed-length m0fdii batches were about 6–11% slower than CUDA alone. It is intentionally opt-in and does not replace CUDA in `auto` mode.

```bash
MIMILLM_BACKEND=cuda python train.py
```

The Python API can select it directly with `train_from_config(..., backend="cuda")`. Use `MIMILLM_DISABLE_CUDA=1` to skip CUDA in automatic mode.

```bash
python tools/build_backend.py --release
MIMILLM_BACKEND=cpp python -m unittest discover -s tests -v
```

CUDA tests and benchmark:

```bash
MIMILLM_BACKEND=cuda python -m unittest discover -s tests -v
python tools/benchmark.py
```

The code favors readability and deterministic tests over production-scale performance. MIT License.
