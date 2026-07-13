# mimiLLM

[English](#english) · [Русский](#русский)

## English

mimiLLM is a small, readable toolkit for building decoder-only language models
from first principles. The package is imported as `mimillm`.

The repository contains the complete path from UTF-8 text to generated tokens:
a byte tokenizer, float32 tensors, automatic differentiation, Transformer
layers, optimizers, checkpoints, datasets, generation, and an optional C++20
CPU backend. The Python implementation has no runtime dependencies.

This is a library and a learning project, not a pretrained chatbot. You can
change the dimensions and build a larger network with the same API, but the
current CPU backend is intended for experiments and education. Training a
billion-parameter production LLM requires distributed accelerators, much more
memory, and a substantially larger software stack.

### What is included

- an installable Python package with a documented public API;
- a contiguous float32 `Tensor` and dynamic autograd engine;
- `Module`, `Parameter`, linear layers, embeddings, RMSNorm, ReLU, and MLP;
- causal multi-head attention and a configurable decoder-only Transformer;
- SGD and AdamW with gradient clipping;
- a reversible UTF-8 byte tokenizer with a fixed 260-token vocabulary;
- text-only, QA-only, and mixed QA/text training;
- versioned binary checkpoints without `pickle`;
- greedy, temperature, and top-k generation;
- a pure Python backend and an optional multithreaded C++20 backend;
- reproducible configs, tests, benchmarks, and small bilingual sample data.

### Install on Windows

Open PowerShell in the repository:

```powershell
conda env create -f environment-windows.yml
conda activate mimillm
python -m pip install --no-deps -e .
python tools/build_backend.py --release
python -m unittest discover -s tests -v
```

`environment-windows.yml` installs MinGW-w64 from Conda, so Visual Studio is
not required. The helper below creates or updates the same environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_conda.ps1
```

### Install on Linux

The same source works on x86-64 and is written to remain portable to AArch64:

```bash
conda env create -f environment.yml
conda activate mimillm
python -m pip install --no-deps -e .
python tools/build_backend.py --release
python -m unittest discover -s tests -v
```

The C++ backend is optional. To use only the reference implementation:

```bash
MIMILLM_BACKEND=python python examples/create_model.py
```

### Create a model from Python

```python
from mimillm import ByteTokenizer, ModelConfig, create_model

config = ModelConfig(
    context_length=128,
    d_model=64,
    n_layers=2,
    n_heads=4,
    d_mlp=192,
)

model = create_model(config)
tokenizer = ByteTokenizer()
tokens = tokenizer.encode("Language models predict what comes next.", add_bos=True)
logits = model([tokens])

print(model.parameter_count())
print(logits.shape)  # (batch, time, vocabulary)
```

You can also pass the configuration fields directly to `create_model(...)`.
Unknown fields, invalid head dimensions, and unsupported vocabulary sizes are
rejected early instead of being ignored.

### Train on your own text

For the command-line trainer, place UTF-8 `.txt`, `.md`, or `.text` files in:

```text
data/text/train/
data/text/validation/
```

Then run the mixed example:

```bash
python tools/make_dataset.py --seed 42
python train.py --config configs/mixed_demo.json --output checkpoints/mixed_demo.bin
python evaluate.py --checkpoint checkpoints/mixed_demo.bin \
  --data data/validation.txt --text-data data/text/validation
```

`text_ratio` is the probability of choosing an ordinary-text batch. Use `0`
for QA-only training, `1` for text-only training, or a value in between for a
mixture. External corpora can be supplied with repeatable `--text-train` and
`--text-validation` options.

The library API also supports a corpus with no QA file at all:

```python
import random
from mimillm import AdamW, ModelConfig, TokenDataset, create_model

config = ModelConfig(
    context_length=64,
    d_model=32,
    n_layers=2,
    n_heads=4,
    d_mlp=96,
    batch_size=1,
    steps=100,
)
model = create_model(config)
data = TokenDataset(text_paths="my_corpus", text_ratio=1.0)
optimizer = AdamW(model.parameters(), learning_rate=3e-4)
rng = random.Random(42)

for _ in range(config.steps):
    inputs, targets = data.sample_batch(config.batch_size, config.context_length, rng)
    flat_targets = [token for row in targets for token in row]
    logits = model(inputs)
    loss = logits.reshape(-1, config.vocab_size).cross_entropy(flat_targets)
    loss.backward()
    optimizer.clip_grad_norm(1.0)
    optimizer.step()
    optimizer.zero_grad()
```

Keep training and validation documents separate. A tiny sample corpus proves
that the pipeline works; it is not enough to teach a model a language. Check
licenses and remove private, duplicated, or low-quality data before training.

### Generate text and load checkpoints

```python
from mimillm import generate_text, load_model

model = load_model("checkpoints/mixed_demo.bin")
continuation = generate_text(
    model,
    "A useful program",
    max_new_tokens=80,
    temperature=0.7,
    top_k=20,
)
print(continuation)
```

The checkpoint header is `MIMILLM1`. A checkpoint stores model configuration,
weights, AdamW moments, step, and seed. Writes are atomic, and the loader checks
the schema, shapes, sizes, duplicate names, truncation, and trailing data.

### Public API

| Area | Main names |
| --- | --- |
| Model | `ModelConfig`, `LanguageModel`, `create_model`, `load_model` |
| Tensor | `Tensor`, `Parameter`, `no_grad` |
| Layers | `Module`, `Linear`, `Embedding`, `RMSNorm`, `ReLU`, `FeedForward` |
| Training | `AdamW`, `SGD`, `TokenDataset`, `cross_entropy` |
| Text | `ByteTokenizer`, `generate`, `generate_text`, `answer_question` |
| Storage | `save_checkpoint`, `load_checkpoint`, `CheckpointData` |
| Backend | `get_backend`, `reset_backend` |

The lower-level modules remain public and readable. This makes it practical to
replace attention, add a new optimizer, write another tokenizer, or implement a
GPU backend without rewriting the rest of the project.

### Backend selection

If a compiled library is available, `auto` uses it; otherwise mimiLLM warns and
falls back to Python.

```text
MIMILLM_BACKEND=auto|python|cpp
MIMILLM_NUM_THREADS=4
MIMILLM_BACKEND_LIBRARY=/absolute/path/to/the/shared/library
MIMILLM_NATIVE=1
```

`MIMILLM_NATIVE=1` enables `-march=native` for GCC/Clang. The resulting binary
may fail on a different CPU. The normal release build deliberately avoids
architecture-specific intrinsics and assembly.

### Repository map

```text
mimillm/       installable Python library
cpp/           portable C++20 kernels and C ABI
configs/       reproducible model and training settings
data/          QA examples and separate text train/validation corpora
examples/      small public-API examples
tests/         unit, gradient, backend, checkpoint, and smoke tests
tools/         build, benchmark, dataset, inspection, and release helpers
train.py       complete reference training loop
evaluate.py    checkpoint evaluation
chat.py        small terminal front end
```

Run `python tools/benchmark.py` to compare the Python backend, one C++ thread,
multiple C++ threads, model forward, and a training step. Small matrices may be
faster on one thread because scheduling work also has a cost.

## Русский

mimiLLM — небольшая и читаемая библиотека для создания decoder-only языковых
моделей с нуля. Имя Python-пакета — `mimillm`.

В репозитории есть весь путь от UTF-8 текста до сгенерированных токенов:
byte-токенизатор, float32 Tensor, автоматическое дифференцирование, слои
Transformer, оптимизаторы, checkpoint, датасеты, генерация и необязательный
C++20 backend для CPU. У Python-части нет runtime-зависимостей.

Это библиотека и учебный проект, а не готовый обученный чат-бот. Через тот же API
можно увеличить число слоёв и размеры модели, но текущий CPU backend рассчитан
на изучение и эксперименты. Для промышленной LLM с миллиардами параметров нужны
распределённые ускорители, намного больше памяти и другой масштаб инфраструктуры.

### Что входит в проект

- устанавливаемый Python-пакет с единым публичным API;
- собственные `Tensor`, autograd, `Module` и `Parameter`;
- Linear, Embedding, RMSNorm, ReLU, MLP и causal multi-head attention;
- настраиваемый decoder-only Transformer;
- SGD, AdamW и ограничение нормы градиента;
- обратимый UTF-8 byte tokenizer со словарём из 260 токенов;
- text-only, QA-only и смешанное QA/text обучение;
- безопасный версионированный checkpoint без `pickle`;
- greedy, temperature и top-k генерация;
- эталонный Python backend и ускоренный C++20 backend;
- тесты, benchmark, воспроизводимые конфигурации и небольшой корпус-пример.

### Установка в Windows

Откройте PowerShell в корне репозитория:

```powershell
conda env create -f environment-windows.yml
conda activate mimillm
python -m pip install --no-deps -e .
python tools/build_backend.py --release
python -m unittest discover -s tests -v
```

MinGW-w64 устанавливается из Conda, поэтому Visual Studio не обязательна.
Окружение также можно создать или обновить помощником:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_conda.ps1
```

### Установка в Linux

```bash
conda env create -f environment.yml
conda activate mimillm
python -m pip install --no-deps -e .
python tools/build_backend.py --release
python -m unittest discover -s tests -v
```

Исходники проверяются на x86-64 и не используют x86 intrinsics, поэтому их можно
собирать для AArch64. Физический запуск на ARM64 всё равно нужно проверять на
соответствующей машине.

### Создание своей модели

```python
from mimillm import ByteTokenizer, ModelConfig, create_model

config = ModelConfig(
    context_length=128,
    d_model=64,
    n_layers=2,
    n_heads=4,
    d_mlp=192,
)
model = create_model(config)

tokenizer = ByteTokenizer()
tokens = tokenizer.encode("Модель предсказывает продолжение.", add_bos=True)
logits = model([tokens])

print(f"Параметров: {model.parameter_count():,}")
print(f"Форма logits: {logits.shape}")
```

Меняйте `d_model`, `n_layers`, `n_heads`, `d_mlp` и `context_length`, чтобы
собрать другую архитектуру. `d_model` должен делиться на `n_heads`. Byte-level
подход не требует создания словаря для каждого языка, но русский символ обычно
занимает два токена, поэтому доступный текстовый контекст короче, чем кажется.

### Обучение на своих текстах

Положите UTF-8 файлы `.txt`, `.md` или `.text` в разные каталоги:

```text
data/text/train/         участвуют в optimizer step
data/text/validation/    используются только для независимой оценки
```

Запуск смешанного примера:

```powershell
python tools/make_dataset.py --seed 42
python train.py --config configs/mixed_demo.json --output checkpoints/mixed_demo.bin
python evaluate.py --checkpoint checkpoints/mixed_demo.bin `
  --data data/validation.txt --text-data data/text/validation
```

`text_ratio` задаёт вероятность text batch: `0` — только QA, `1` — только
обычные тексты, промежуточное значение — смесь. Свои каталоги можно подключить
повторяющимися параметрами `--text-train` и `--text-validation`. На уровне API
QA-файл вообще не обязателен:

```python
from mimillm import TokenDataset

dataset = TokenDataset(text_paths=r"D:\corpus", text_ratio=1.0)
```

Не смешивайте одинаковые документы между train и validation. Маленькие файлы из
репозитория проверяют механику, но не обучают полноценному языку. Для реального
результата нужны большой качественный корпус, права на его использование,
достаточная модель и вычисления.

### Готовые конфигурации

- `configs/debug.json` — четыре быстрых смешанных шага;
- `configs/qa_demo.json` — QA-only профиль;
- `configs/mixed_demo.json` — QA + обычные тексты;
- `configs/tiny.json` — более крупный учебный профиль.

Обучение можно продолжить без потери состояния AdamW:

```powershell
python train.py --resume checkpoints/mixed_demo.bin --steps 3000
```

`--steps` означает новый общий предел, а не число дополнительных шагов.

### Использование как библиотеки

Основные имена импортируются прямо из `mimillm`:

| Задача | API |
| --- | --- |
| Создать модель | `ModelConfig`, `LanguageModel`, `create_model` |
| Загрузить модель | `load_model` |
| Работать с Tensor | `Tensor`, `Parameter`, `Module`, `no_grad` |
| Обучать | `AdamW`, `SGD`, `TokenDataset`, `cross_entropy` |
| Генерировать | `generate`, `generate_text`, `answer_question` |
| Сохранять | `save_checkpoint`, `load_checkpoint` |

Низкоуровневые модули тоже остаются доступными. Можно заменить attention,
написать новый optimizer или tokenizer, добавить слой либо подключить другой
вычислительный backend, сохранив остальную часть проекта.

### Выбор backend

```powershell
$env:MIMILLM_BACKEND = "cpp"       # auto, python или cpp
$env:MIMILLM_NUM_THREADS = "4"
python train.py --config configs/debug.json
```

Для библиотеки в нестандартном месте задайте полный путь через
`MIMILLM_BACKEND_LIBRARY`. Без собранной DLL/`.so` режим `auto` выдаёт
предупреждение и использует Python. Это позволяет изучать код даже без C++
компилятора.

### Проверки и лицензия

```bash
python -m unittest discover -s tests -v
python tools/benchmark.py
python tools/package_release.py
```

Проект распространяется по лицензии MIT. Смотрите [LICENSE](LICENSE) и
[CHANGELOG.md](CHANGELOG.md).
