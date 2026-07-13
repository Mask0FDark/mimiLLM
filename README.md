# mimiLLM

mimiLLM is a small Python library for building, training, saving, and studying decoder-only language models. Its tensor engine, automatic differentiation, Transformer layers, optimizer, tokenizer, data pipeline, and weight format are implemented directly in the project, without NumPy, PyTorch, or another ML framework.

The library is intended for learning and experiments. It can train on ordinary texts, question-answer examples, or a mixture of both. The implementation is deliberately readable; it is not a replacement for production GPU frameworks used to train billion-parameter models.

[Русская версия](#русский) · [English version](#english)

## Русский

### Установка

Python 3.12 или новее:

```powershell
conda create -n my-model python=3.12 -y
conda run -n my-model python -m pip install -e E:\mimiLLM
```

После публикации библиотеку также можно установить прямо из GitHub:

```powershell
python -m pip install "git+https://github.com/Mask0FDark/mimiLLM.git"
```

Чистая Python-реализация работает на Windows и Linux. Необязательный C++ backend ускоряет основные операции на CPU.

### Вариант 1: загрузить готовые веса

Папка готовой модели содержит два переносимых файла:

```text
weights/
├── config.json
└── model.safetensors
```

Использование:

```python
from mimillm import generate_text, load_model

model = load_model("weights")
continuation = generate_text(
    model,
    "Однажды вечером",
    max_new_tokens=40,
    temperature=0.7,
    top_k=20,
)
print(continuation)
```

Можно передать и сам файл весов: `load_model("weights/model.safetensors")`. В этом случае `config.json` должен лежать рядом. Полный запускаемый пример находится в [examples/use_weights.py](examples/use_weights.py).

### Вариант 2: обучить модель на своих данных

Рекомендуемая структура отдельного проекта:

```text
my_model/
├── config.json
├── train.py
└── data/
    ├── text/
    │   ├── train/
    │   │   └── books.txt
    │   └── validation/
    │       └── validation.txt
    └── question/
        ├── train/
        │   └── questions.txt
        └── validation/
            └── questions.txt
```

Пути задаются в `config.json`. Относительные пути считаются от каталога этого файла:

```json
{
  "vocab_size": 260,
  "context_length": 96,
  "d_model": 64,
  "n_layers": 2,
  "n_heads": 4,
  "d_mlp": 192,
  "batch_size": 1,
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

Минимальный `train.py`:

```python
from pathlib import Path

from mimillm import train_from_config


HERE = Path(__file__).resolve().parent
result = train_from_config(HERE / "config.json", output_dir=HERE / "weights")
print(result.weights_dir)
```

Запуск: `python train.py`. Готовый пример есть в [examples/train_model.py](examples/train_model.py). После обучения папка `weights` содержит:

- `config.json` и `model.safetensors` — файлы для загрузки и распространения модели;
- `training_checkpoint.bin` — внутреннее состояние модели и AdamW для продолжения обучения.

Продолжить обучение можно так:

```python
result = train_from_config(
    HERE / "config.json",
    output_dir=HERE / "weights",
    resume=HERE / "weights" / "training_checkpoint.bin",
)
```

### Форматы данных

В `data/text/train` и `data/text/validation` можно класть любое число UTF-8 файлов `.txt`, `.md` и `.text`, включая вложенные каталоги. Они используются для обычного предсказания следующего токена и изучения языка.

Файлы в `data/question/train` и `data/question/validation` используют простые блоки:

```text
Вопрос: Что такое нейронная сеть?
Ответ: Это модель, которая обучается на примерах.

Вопрос: What is a token?
Ответ: A token is a unit processed by a language model.
```

`text_ratio` задаёт долю учебных batch из обычных текстов:

- `0.0` — только вопросы и ответы;
- `1.0` — только обычные тексты;
- значение между ними — смешанное обучение.

Train и validation должны содержать разные данные. Validation не участвует в обновлении весов и нужна для проверки переобучения.

### Создать модель программно

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
save_model("weights", model)
```

### Формат весов

`model.safetensors` следует документированной структуре SafeTensors: JSON-заголовок описывает имена, формы, типы и смещения тензоров, после него идут непрерывные бинарные данные. mimiLLM записывает параметры как `F32` и может читать свои файлы без внешней зависимости. Подробнее: [официальная спецификация SafeTensors](https://github.com/huggingface/safetensors#format).

Имена `config.json` и `model.safetensors` выбраны так же, как в привычных каталогах опубликованных языковых моделей. Это делает структуру понятной и пригодной для внешних инструментов, но архитектуру mimiLLM всё равно должен поддерживать загрузчик, который читает веса.

### Разработка и проверки

```powershell
python -m unittest discover -s tests -v
python tools/build_backend.py
$env:MIMILLM_BACKEND = "cpp"
python -m unittest discover -s tests -v
```

Linux/WSL:

```bash
python3 -m unittest discover -s tests -v
python3 tools/build_backend.py
MIMILLM_BACKEND=cpp python3 -m unittest discover -s tests -v
```

Полезные переменные окружения:

- `MIMILLM_BACKEND=auto|python|cpp` выбирает backend;
- `MIMILLM_NUM_THREADS=N` задаёт число CPU-потоков C++ backend;
- `MIMILLM_CPP_LIBRARY=/path/to/library` задаёт явный путь к DLL или `.so`.

## English

### Install

Use Python 3.12 or newer:

```bash
python -m pip install "git+https://github.com/Mask0FDark/mimiLLM.git"
```

For editable local development: `python -m pip install -e E:\mimiLLM`.

The Python backend runs on Windows and Linux. An optional C++ backend accelerates core CPU operations.

### Option 1: load existing weights

A reusable model directory contains `config.json` and `model.safetensors`:

```python
from mimillm import generate_text, load_model

model = load_model("weights")
text = generate_text(model, "Once upon a time", max_new_tokens=40)
print(text)
```

`load_model("weights/model.safetensors")` is also supported when `config.json` is next to the weight file. See [examples/use_weights.py](examples/use_weights.py).

### Option 2: train on your own data

Create these four directories next to your project config:

```text
data/text/train/
data/text/validation/
data/question/train/
data/question/validation/
```

Put UTF-8 `.txt`, `.md`, or `.text` documents in the text directories. Put `.txt` files containing `Вопрос:`/`Ответ:` blocks in the question directories. All directories are scanned recursively.

Add the paths to `config.json`:

```json
{
  "text_ratio": 0.5,
  "text_train_path": "data/text/train",
  "text_validation_path": "data/text/validation",
  "question_train_path": "data/question/train",
  "question_validation_path": "data/question/validation"
}
```

This abbreviated JSON only highlights data fields; a real config also needs the model and training fields shown above, or can use `ModelConfig` defaults.

```python
from pathlib import Path
from mimillm import train_from_config

HERE = Path(__file__).resolve().parent
result = train_from_config(HERE / "config.json", output_dir=HERE / "weights")
```

Relative dataset paths are resolved from the directory containing `config.json`. `text_ratio=0` selects question-answer data only, `text_ratio=1` selects ordinary text only, and values in between mix both sources. See [examples/train_model.py](examples/train_model.py).

Training exports `config.json` and `model.safetensors` for inference, plus `training_checkpoint.bin` for resuming AdamW state. The portable weights use the documented [SafeTensors format](https://github.com/huggingface/safetensors#format) with float32 tensors.

### Scope

mimiLLM includes a byte tokenizer, tensors, autograd, layers, causal multi-head attention, a decoder-only Transformer, AdamW/SGD, mixed text and QA datasets, generation, checkpoints, SafeTensors weights, and an optional threaded C++ backend. The code favors clarity and deterministic tests over production-scale performance.

MIT License.
