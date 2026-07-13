# mimiLLM

mimiLLM — открытая Python-библиотека для создания, обучения и исследования decoder-only языковых моделей.

Внутри есть собственные тензоры на `float32`, автоматическое вычисление градиентов, слои нейронной сети, causal attention, decoder-only Transformer, AdamW, токенизатор, обучение и сохранение весов. Всё это написано без NumPy, PyTorch, TensorFlow и других ML-фреймворков. Вычисления выполняются на чистом Python или через многопоточный C++ backend.

Архитектура, размеры модели, источники данных и параметры обучения задаются пользователем. Библиотеку можно использовать как основу собственного проекта, а её компактная реализация позволяет проследить весь путь от входного текста до обновления весов.

[English version](#english-version)

## Что с ней можно сделать

- создать decoder-only Transformer своего размера;
- обучить его на обычных текстах, вопросах и ответах или на смеси этих данных;
- сохранить модель в папку с `config.json` и `model.safetensors`;
- загрузить готовые веса одной функцией;
- продолжить прерванное обучение из checkpoint;
- переключаться между понятным Python backend и более быстрым C++ backend;
- открыть реализацию любого шага — от умножения тензоров до attention и AdamW.

Если хочется увидеть готовый проект модели, посмотрите [m0fdii](https://github.com/Mask0FDark/m0fdii). Там уже лежат данные, конфигурация, обученные веса и короткие скрипты запуска.

## Установка

Требуется Python 3.12 или новее.

Установка последней версии с GitHub:

```powershell
python -m pip install "git+https://github.com/Mask0FDark/mimiLLM.git"
```

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

Сторонние runtime-зависимости библиотеке не нужны. Без собранного C++ backend автоматически используется Python.

## Загрузка готовых весов

Обычная папка модели выглядит так:

```text
weights/
├── config.json
└── model.safetensors
```

Загрузка и генерация текста:

```python
from mimillm import generate_text, load_model

model = load_model("weights")
text = generate_text(
    model,
    "Однажды вечером",
    max_new_tokens=40,
    temperature=0.7,
    top_k=20,
)
print(text)
```

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
  "vocab_size": 260,
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
  "text_train_path": "data/text/train",
  "text_validation_path": "data/text/validation",
  "question_train_path": "data/question/train",
  "question_validation_path": "data/question/validation"
}
```

Главные параметры архитектуры:

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
)

print(result.weights_dir)
```

После `python train.py` в папке `weights` появятся:

```text
weights/
├── config.json
├── model.safetensors
└── training_checkpoint.bin
```

`config.json` и `model.safetensors` — переносимая модель для `load_model()`. В `training_checkpoint.bin` дополнительно лежат moments AdamW, номер шага и seed; этот файл нужен только для возобновления обучения.

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

Токенизатор byte-level: значения `0–255` соответствуют байтам UTF-8, а ещё четыре значения используются для `PAD`, `BOS`, `EOS` и `SEP`. Поэтому словарь всегда содержит 260 токенов и может представить любой русский, английский или другой UTF-8 текст. Обратная сторона простоты — один Unicode-символ часто занимает несколько токенов.

Transformer использует pre-norm блоки, causal mask, обучаемые позиционные embedding и отдельную выходную проекцию. Это настоящая авторегрессионная модель: во время генерации она много раз вычисляет logits и каждый раз выбирает следующий токен.

## Веса и checkpoint — не одно и то же

`model.safetensors` хранит только именованные `F32` тензоры модели. Формат состоит из JSON-заголовка и непрерывного бинарного буфера и совместим с документированной [спецификацией SafeTensors](https://github.com/huggingface/safetensors#format). mimiLLM читает и записывает его собственной реализацией без внешней зависимости.

`training_checkpoint.bin` — внутренний формат mimiLLM. Кроме параметров модели он содержит состояние оптимизатора и служебную информацию, поэтому занимает больше места. Для публикации готовой модели достаточно `config.json` и `model.safetensors`.

## Python и C++ backend

По умолчанию используется режим `auto`: если собранная библиотека найдена, тяжёлые операции выполняются в C++, иначе всё работает на Python.

Сборка на Windows:

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

- `MIMILLM_BACKEND=auto|python|cpp` — выбор backend;
- `MIMILLM_NUM_THREADS=N` — число рабочих потоков C++;
- `MIMILLM_CPP_LIBRARY=/path/to/library` — явный путь к DLL или `.so`.

C++ backend отвечает за вычислительные kernels и подключается через стабильный C ABI. Граф модели, autograd и цикл обучения при этом остаются в Python, поэтому оба backend используют одну архитектуру и один публичный API.

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

Тесты проверяют тензорные операции, численные градиенты, autograd, слои, attention, оптимизаторы, checkpoint, SafeTensors, генерацию, работу с данными и короткий полный цикл обучения. Проект проверялся на Windows и WSL/Linux.

## English version

mimiLLM is an open Python library for building, training, and exploring decoder-only language models. It implements float32 tensors, autograd, neural-network layers, causal attention, a decoder-only Transformer, AdamW, datasets, training, generation, checkpoints, and SafeTensors weights without NumPy, PyTorch, or another ML runtime.

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
from mimillm import generate_text, load_model

model = load_model("weights")
print(generate_text(model, "Once upon a time", max_new_tokens=40))
```

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
result = train_from_config(HERE / "config.json", output_dir=HERE / "weights")
```

Relative data paths are resolved from the config directory. Training exports `config.json` and `model.safetensors` for inference, plus `training_checkpoint.bin` for resuming AdamW state.

See [examples/train_model.py](examples/train_model.py), [examples/use_weights.py](examples/use_weights.py), and the complete [m0fdii model project](https://github.com/Mask0FDark/m0fdii).

### Backends and tests

The pure Python backend works on Windows and Linux. The optional threaded C++ backend accelerates core operations while keeping the model and training loop in Python.

```bash
python tools/build_backend.py --release
MIMILLM_BACKEND=cpp python -m unittest discover -s tests -v
```

The code favors readability and deterministic tests over production-scale performance. MIT License.
