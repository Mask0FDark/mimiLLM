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
- безопасно выполнить цепочку языкового pretraining и диалогового SFT;
- проверить утечки данных, происхождение весов и качество BPE до долгого запуска;
- переключаться между CUDA, многопоточным C++ и понятным Python backend;
- открыть реализацию любого шага — от умножения тензоров до attention и AdamW.

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
├── model.safetensors
└── tokenizer.json      # только для tokenizer: "bpe"
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

Для настоящих многоходовых разговоров используйте `.jsonl`: каждая строка содержит один JSON-объект с чередующимися сообщениями `user` и `assistant`.

```json
{"messages":[{"role":"user","content":"Меня зовут Ира."},{"role":"assistant","content":"Приятно познакомиться, Ира."},{"role":"user","content":"Как меня зовут?"},{"role":"assistant","content":"Тебя зовут Ира."}]}
```

Библиотека разворачивает такой разговор по ходам. Первый ответ учится на первом вопросе, второй — на первом вопросе, первом ответе и новом вопросе, третий — на двух предыдущих парах и так далее. Loss вычисляется на текущем ответе, а предыдущие реплики остаются в attention-контексте. Благодаря этому формат обучения совпадает с обычным chat prompt; простой набор независимых QA-пар такого навыка не даёт.

### Конфигурация

Небольшой рабочий `config.json`:

```json
{
  "vocab_size": 355,
  "tokenizer": "unicode",
  "tie_word_embeddings": true,
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
  "adam_beta1": 0.9,
  "adam_beta2": 0.95,
  "adam_epsilon": 1e-8,
  "gradient_clip_norm": 1.0,
  "learning_rate_schedule": "cosine",
  "min_learning_rate_ratio": 0.1,
  "warmup_steps": 20,
  "validation_interval": 25,
  "checkpoint_interval": 50,
  "save_validation_checkpoints": false,
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

- `tokenizer` выбирает `byte`, `unicode` или `bpe`; фиксированным режимам соответствуют `vocab_size` 260 и 355, а `bpe` использует размер из `tokenizer.json`;
- `context_length` — сколько последних токенов помещается в контекст;
- `d_model` — размер внутреннего представления токена;
- `n_layers` — количество Transformer-блоков;
- `n_heads` — число attention-голов;
- `d_mlp` — размер скрытой части feed-forward слоя.
- `tie_word_embeddings: true` использует одну матрицу и для входных токенов, и для выходных logits. Это освобождает параметры под более крупный BPE-словарь; старые конфигурации без этого поля продолжают загружаться с отдельной выходной матрицей.

Параметры обучения:

- `batch_size` задаёт число примеров в одном batch, а `steps` — общее число обновлений весов; большой batch обрабатывает больше токенов за шаг, поэтому скорость сравнивают по `tok/s`, а не по длительности одного шага;
- `batches_per_epoch` задаёт длину эпохи; при `null` она рассчитывается по данным автоматически;
- `learning_rate`, `weight_decay`, `adam_beta1`, `adam_beta2` и `adam_epsilon` управляют AdamW;
- `gradient_clip_norm` ограничивает общую норму градиента и защищает обучение от редких резких скачков;
- `warmup_steps` линейно поднимает learning rate в начале, после чего `learning_rate_schedule: "cosine"` плавно уменьшает его до `learning_rate × min_learning_rate_ratio`; также доступны `linear` и `constant`;
- `validation_interval` задаёт частоту проверки validation loss;
- `checkpoint_interval` задаёт частоту сохранения состояния;
- `save_validation_checkpoints: true` сохраняет отдельную папку весов для каждой проверки в `weights/validation/step_XXXXXXXX`; это позволяет позже сравнить ответы нескольких этапов обучения, а не только `best` и `last`;
- `seed` делает инициализацию и выбор batch воспроизводимыми.

`batches_per_epoch` влияет только на группировку строк прогресса и номер
отображаемой эпохи. Её граница не запускает validation или checkpoint: они
выполняются строго по своим интервалам и один раз на последнем шаге. Поэтому
early stopping считает запланированные проверки, а не оформление лога.

`text_ratio` управляет смешиванием источников. При `0.0` модель учится только на вопросах и ответах, при `1.0` — только на обычных текстах. Значение `0.35` означает, что примерно 35% batch будут текстовыми.

По умолчанию новый `TransformerConfig` использует AdamW (`beta1=0.9`, `beta2=0.95`, `epsilon=1e-8`), global gradient clipping `1.0`, линейный warmup и cosine decay до 10% базового learning rate. CUDA и C++ backend автоматически выполняют AdamW и clipping нативными функциями, когда они доступны. Это один согласованный стек оптимизации: одновременно применять AdamW и SGD к одним весам нельзя. Для воспроизведения старых JSON-конфигураций без новых полей сохраняются прежние linear decay и `beta2=0.999`.

В QA-batch вопрос всегда остаётся в attention-контексте. `qa_prompt_weight` задаёт его долю в loss. `qa_answer_prefix_tokens` выбирает начало ответа, а `qa_answer_prefix_weight` усиливает его. Это полезно для маленькой модели: первые слова чаще всего определяют, какой именно ответ был выбран. Значения по умолчанию `0.0`, `0` и `1.0` сохраняют прежнее поведение.

Если в каталоге QA лежат наборы разного размера, их можно балансировать не по
числу строк, а явными вероятностями выбора файла:

```json
{
  "text_ratio": 0.05,
  "qa_source_weights": {
    "identity.jsonl": 0.55,
    "dialogues.jsonl": 0.35,
    "facts.jsonl": 0.10
  }
}
```

Ключи задаются относительно `question_train_path` и должны перечислять все
найденные QA-файлы. Значения нормализуются; вес `0` исключает файл из текущего
этапа. Оставшиеся 5% в примере занимает обычный текст. Та же смесь используется
для общего validation loss, а фактические вероятности печатаются как
`Source mix` перед первым шагом.

Относительные пути к данным считаются от каталога, в котором лежит `config.json`. Поэтому запуск не зависит от текущей папки терминала.

### Быстрый вызов токенизации

Для одного преобразования необязательно вручную создавать объект токенизатора:

```python
from mimillm import detokenize, tokenize

tokens = tokenize("Привет, mimiLLM!", "unicode", add_bos=True, add_eos=True)
text = detokenize(tokens, "unicode")
```

BPE-токенизатор можно передать готовым объектом или прямо путём к артефакту:

```python
tokens = tokenize("Привет!", "weights/tokenizer.json")
text = detokenize(tokens, "weights/tokenizer.json")
```

Для многократной работы эффективнее один раз вызвать `create_tokenizer` или `load_tokenizer`, а затем передавать готовый объект в `tokenize`.

### Subword BPE

Для новых моделей можно использовать byte-level BPE:

```json
{
  "tokenizer": "bpe",
  "vocab_size": 2048
}
```

Перед обучением модели нужно собрать `tokenizer.json` из train-данных:

```python
from mimillm import train_tokenizer_from_config

tokenizer = train_tokenizer_from_config(
    "config.json",
    vocab_size=2048,
    min_frequency=2,
)
print(tokenizer.VOCAB_SIZE)
```

Функция сохраняет не только `tokenizer.json`, но и `tokenizer_report.json`. В отчёте есть число токенов на UTF-8-байт и слово, использование byte fallback, покрытие Unicode и проверка точного обратного преобразования. Обратимость сама по себе больше не считается достаточной проверкой качества.

Новый формат BPE сначала создаёт цельные токены для частых многобайтовых Unicode-символов, а затем обучает обычные частотные слияния. Поэтому русская буква из обучающего корпуса не становится двумя независимыми byte-целями. Для модели в несколько миллионов параметров разумная начальная точка — словарь 2 000–4 000, но окончательный размер нужно выбирать по отчёту на конкретном корпусе.

Если корпус слишком маленький, фактический словарь может быть меньше запрошенного. Многоэтапный pipeline автоматически подставляет фактический размер во все этапы. При ручном обучении используйте напечатанное значение как `vocab_size` в `config.json`. После сохранения модели `save_model` кладёт `tokenizer.json` рядом с `model.safetensors`, а `load_model` загружает его автоматически.

Новый BPE разделяет Unicode-слова, числа и знаки, а обычный пробел прикрепляет к следующему слову. Это позволяет учить токены вида `" модель"` и лучше использовать словарь. Формат записан в `tokenizer.json`; старые BPE-файлы версии 1 продолжают загружаться с прежним поведением.

### Рекомендуемый способ: обучение по этапам

Для новой модели используйте `pipeline.json`, а не связывайте каталоги весов вручную. Готовый пример находится в [examples/staged_training](examples/staged_training).

Общая архитектура и optimizer-параметры лежат в `model.json`, указанном через
`base_config`. Файлы этапов содержат только отличия вроде `steps`,
`learning_rate`, `text_ratio` и путей к данным, поэтому одну архитектуру не
нужно копировать в несколько больших JSON-файлов.

```powershell
python train_pipeline.py examples/staged_training/pipeline.json --backend cuda
```

После установки пакета доступна та же команда:

```powershell
mimillm-train-pipeline examples/staged_training/pipeline.json --backend cuda
```

Pipeline выполняет последовательность:

```text
обучение общего BPE на train-источниках
        ↓
causal pretraining только на обычном тексте
        ↓
SFT на ответах ассистента с небольшим text replay при необходимости
```

Обычно первый этап имеет `kind: "pretrain"` и `text_ratio: 1.0`. Следующий
этап с `kind: "sft"` автоматически получает лучшие веса предыдущего этапа,
новый AdamW и новое расписание learning rate. Если качественные pretrained-веса
уже существуют, укажите верхнеуровневое поле `"initial_weights":
"weights/language"`: тогда первым этапом может быть SFT, но совместимость
архитектуры и токенизатора всё равно проверяется. Запустить SFT с нуля случайно
нельзя. `allow_sft_from_scratch: true` существует только для явных
диагностических экспериментов.

Несколько SFT-этапов — это не несколько независимых моделей: каждый этап
продолжает изменять одни и те же веса, но меняет состав данных, learning rate и
optimizer schedule. Так можно сначала закрепить базовые ответы, затем добавить
знания и только после этого усилить многоходовой диалог.

До первого optimizer step библиотека:

- ищет совпадения между train и validation даже между разными этапами;
- сообщает о дубликатах и одинаковых вопросах с разными ответами;
- отклоняет ответы с фразами из `dataset_checks.forbidden_phrases`, например с чужим именем ассистента;
- создаёт `tokenizer_report.json` и проверяет минимальное Unicode-покрытие и максимальное отношение tokens/byte;
- проверяет одинаковую архитектуру всех этапов.

Validation loss не проверяет, умеет ли модель реально отвечать. Поэтому этап
может дополнительно иметь поле `"evaluation": "dialogue_eval.json"`. После
обучения pipeline сравнивает модель с минимальным validation loss и модель
последнего шага, детерминированно выполняет независимые одно- и многоходовые
диалоги и записывает выбранный результат в `generation_report.json`, а оба
результата — в `generation_candidates.json`. Если последние веса лучше проходят
сценарии и достигают `min_pass_rate`, они становятся основными весами этапа, а
модель с минимальным loss сохраняется в `best_validation/`. Если ни один
кандидат не набрал требуемую долю сценариев,
этап получает статус `quality_failed`, а следующий этап не запускается. Формат
набора показан в
[dialogue_eval.json](examples/staged_training/dialogue_eval.json); его вопросы
не должны встречаться в train.

Помимо `exact`, `contains_all`, `contains_any` и `forbidden`, ожидание ответа
может задавать `min_characters`, `min_cyrillic_characters` и
`max_repeated_word_fraction`. Эти проверки полезны для маленьких моделей:
пустой ответ, повторяющиеся `Ответ:`, длинная строка разделителей или один и тот
же токен до конца контекста не пройдут gate как нормальная генерация.

Для предварительного обучения можно задать, например,
`"max_validation_loss": 5.0`. Если лучший validation loss выше порога, этап
получает статус `quality_failed`, а SFT не начинается. Это не универсальное
«число качества»: порог выбирают по базовому запуску с тем же корпусом,
токенизатором и словарём. Провал условия поднимает публичное исключение
`PipelineQualityError`; консольная команда выводит короткую причину без
внутреннего traceback.

В каждом каталоге весов появляется `lineage.json`: там записаны тип этапа, родительские веса, эффективная конфигурация и SHA-256 токенизатора и модели. Pipeline не записывает новые веса поверх непустого каталога. Прерванный этап продолжается так:

```powershell
python train_pipeline.py pipeline.json --backend cuda --resume-stage dialogue
```

При продолжении можно увеличить `steps` и изменить частоту validation/checkpoint. Изменение архитектуры, данных или токенизатора будет отклонено.

### Обучение одного этапа вручную

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
├── best/
│   ├── config.json
│   └── model.safetensors
└── last/
    ├── config.json
    └── model.safetensors
```

В `best` находятся веса с минимальным validation loss; путь и SHA-256 этой модели записаны в `best_validation.json`. Корневые `config.json` и `model.safetensors` остаются напрямую загружаемыми через `load_model("weights")`, а pipeline может поместить туда вариант, который лучше прошёл проверку реальной генерации. В `last` сохраняются веса последнего шага. `training_checkpoint.bin` дополнительно содержит moments AdamW, номер шага и seed и нужен для возобновления обучения.

Validation проходит по всем проверочным ответам и текстам. Во время долгой проверки в терминале отдельно показывается прогресс `val-qa` и `val-text`.

```python
result = train_from_config(
    HERE / "config.json",
    output_dir=HERE / "weights",
    resume=HERE / "weights" / "training_checkpoint.bin",
)
```

`resume` продолжает тот же этап: восстанавливает номер шага, optimizer и его
moments. Низкоуровневый `init_from` переносит веса в новый этап, но безопаснее
использовать `train_pipeline`, который проверяет и записывает всю цепочку.
При `init_from` создаётся новый optimizer и заново запускаются warmup и schedule:

```python
pretrain = train_from_config(
    HERE / "pretrain.json",
    output_dir=HERE / "pretrain_weights",
)
fine_tune = train_from_config(
    HERE / "sft.json",
    output_dir=HERE / "sft_weights",
    init_from=pretrain.weights_dir,
)
```

Архитектура и токенизатор этапов должны совпадать. Для BPE один и тот же файл
можно указать через `tokenizer_path`. `early_stopping_patience` задаёт число
validation-проверок без улучшения перед остановкой, а
`early_stopping_min_delta` — минимальное значимое улучшение. Хорошая практическая
схема: сначала causal pretraining на естественном тексте или полных диалогах,
затем SFT с `qa_prompt_weight: 0` и loss только на ответах ассистента.

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
mimillm/audit.py           утечки train/validation и загрязнение данных
mimillm/training.py        полный цикл обучения
mimillm/pipeline.py        проверенное многоэтапное обучение и lineage
mimillm/generation.py      авторегрессионная генерация
mimillm/safetensors.py     чтение и запись переносимых весов
mimillm/checkpoint.py      checkpoint для продолжения обучения
```

В библиотеке есть два фиксированных токенизатора и один обучаемый. `byte` имеет словарь из 260 токенов и кодирует весь текст байтами UTF-8. `unicode` имеет 355 токенов: частые кириллические буквы и знаки занимают один токен, а всё остальное обратимо кодируется теми же UTF-8-байтами. `bpe` обучает subword-словарь из train-корпуса, хранит его в `tokenizer.json` и также сохраняет byte fallback для любого UTF-8 текста.

Выбор токенизатора хранится в `config.json`, и `load_model` автоматически использует его при генерации. Для BPE рядом с весами должен лежать тот же `tokenizer.json`. Менять `tokenizer` или `vocab_size` уже после обучения нельзя: формы embedding и выходного слоя будут другими, поэтому нужно начать новое обучение.

Transformer использует pre-norm блоки, causal mask и обучаемые позиционные embedding. По умолчанию входная embedding-матрица связана с выходными logits. Во время генерации модель много раз вычисляет logits и каждый раз выбирает следующий токен; специальные токены и продолжения, нарушающие UTF-8, отфильтровываются до sampling.

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

### C++ и Python

Установка через pip на Windows x64 содержит готовую C++ DLL. На Linux, включая Raspberry Pi OS/Ubuntu arm64, `pip install` автоматически собирает `.so`, если установлен `g++`, `clang++` или `c++`. Ручная сборка нужна только при изменении C++-исходников:

```powershell
python tools/build_backend.py --release
$env:MIMILLM_BACKEND = "cpp"
```

Linux или WSL:

```bash
python3 tools/build_backend.py --release
export MIMILLM_BACKEND=cpp
```

Библиотека также проверялась на Raspberry Pi 5 с Ubuntu Server 24.04 arm64.
Готовый скрипт создаёт изолированный venv, устанавливает mimiLLM и проверяет
автоматически собранный ARM64 C++ backend:

```bash
./scripts/setup_raspberry_pi.sh
```

Доступные переменные окружения:

- `MIMILLM_BACKEND=auto|cuda|cpp|python` — выбор backend;
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

## Диагностика SFT и сохранения

Перед долгим обучением можно проверить весь путь на одной QA-паре:

```powershell
mimillm-check-sft --backend python --output-dir diagnostics/one_pair_cpu
mimillm-check-sft --backend cuda --output-dir diagnostics/one_pair_cuda
```

Команда выполняет только маленький тест: показывает и сохраняет токены и веса
loss, переобучает модель из 3,8 тысячи параметров на одном ответе, проверяет
greedy-генерацию и загружает веса в отдельном процессе. Итог находится в
`one_pair_sft_report.json`; код возврата ненулевой, если хотя бы одна проверка
не прошла.

Лучшие по validation веса хранятся в `weights/best/`. Файл
`best_validation.json` указывает на этот каталог и содержит SHA-256 модели.
Сначала полностью сохраняются новые веса, и только затем атомарно меняется
указатель. Корень `weights/` остаётся напрямую загружаемым и после pipeline
может содержать выбранный по проверке генерации вариант; `weights/last/`
всегда содержит последний шаг.

Если pipeline был остановлен до первого checkpoint, запустите его с
`--resume-stage`: незавершённый этап начнётся заново от проверенных
родительских весов. Если checkpoint уже существует, будут восстановлены шаг,
AdamW и расписание. Хеши точных родительских весов, конфигурации и токенизатора
записываются в `lineage.json`.

Для короткой проверки возможной утечки оперативной памяти:

```powershell
python tools/memory_regression.py --backend cuda --steps 500
```

Это диагностический tiny-прогон, а не обучение пользовательской модели.

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

The `weights` directory contains `config.json` and `model.safetensors`. BPE models also contain `tokenizer.json`. You may pass the `.safetensors` path directly when the config file and optional tokenizer file are next to it.

### Train on your data

Create four data directories:

```text
data/text/train/
data/text/validation/
data/question/train/
data/question/validation/
```

Text directories accept recursive UTF-8 `.txt`, `.md`, and `.text` documents. Question directories accept `.txt` files made of `Вопрос:` and `Ответ:` blocks and dialogue `.jsonl` files. Each JSONL line contains one `{"messages": [...]}` object with strictly alternating `user` and `assistant` roles. Set the paths and `text_ratio` in `config.json`, then call:

For a new model, the recommended entry point is the checked staged pipeline in
[examples/staged_training](examples/staged_training):

```bash
mimillm-train-pipeline examples/staged_training/pipeline.json --backend auto
```

From a source checkout, `python train_pipeline.py ...` runs the same command.
The pipeline trains one shared tokenizer from train sources, normally starts
with a text-only causal `pretrain` stage, and automatically initializes the next
answer-only `sft` stage from the preceding best weights. When compatible
pretrained weights already exist, top-level `"initial_weights":
"weights/language"` may initialize a pipeline whose first stage is SFT. The
architecture and tokenizer are still verified, while AdamW, warmup, and the
step counter start fresh. It refuses accidental SFT from scratch and refuses to
overwrite a non-empty stage directory.
Shared architecture and optimizer fields can live in one `base_config`; each
stage JSON then contains only its data/objective and schedule overrides.

Before the first optimizer step it audits train/validation leakage across all
stages, reports duplicates and conflicting answers, rejects configured foreign
assistant identities, measures tokenizer quality, and verifies that every
stage has the same model architecture. Each stage writes `lineage.json` with
its parent, effective configuration, and tokenizer/model hashes. Resume an
interrupted stage with:

Validation loss alone does not prove that a model can answer. A stage may set
`"evaluation": "dialogue_eval.json"` to run deterministic held-out single- and
multi-turn generation after training. The pipeline compares the
lowest-validation-loss and final-step weights, stores both results in
`generation_candidates.json`, and writes the selected result to
`generation_report.json`. If the final weights pass more scenarios and meet the
gate, they become the stage deployment weights while the loss-best model is
preserved in `best_validation/`. If neither candidate reaches `min_pass_rate`,
the stage becomes `quality_failed` and the next stage does not start. See the
[evaluation-suite example](examples/staged_training/dialogue_eval.json), and do
not reuse its prompts in train data.

Assistant expectations may also set `min_characters`,
`min_cyrillic_characters`, and `max_repeated_word_fraction` in addition to
`exact`, `contains_all`, `contains_any`, and `forbidden`. These shape checks
reject empty output, repeated role labels, separator runs, and one-token loops
instead of treating them as usable generation.

A pretraining stage may also set `"max_validation_loss": 5.0`. If its best
validation loss remains above the configured limit, the stage is marked
`quality_failed` and SFT does not start. This threshold is corpus- and
tokenizer-specific, not a universal quality score. Library callers receive
`PipelineQualityError`; the command-line entry point prints the concise reason
without an internal traceback.

```bash
mimillm-train-pipeline pipeline.json --backend auto --resume-stage dialogue
```

The lower-level one-stage API remains available:

```python
from pathlib import Path
from mimillm import train_from_config

HERE = Path(__file__).resolve().parent
result = train_from_config(
    HERE / "config.json", output_dir=HERE / "weights", backend="auto",
)
```

Relative data paths are resolved from the config directory. `best/` keeps the lowest-validation-loss model and `best_validation.json` points to it with a verified model SHA-256. The root remains directly loadable and a staged generation gate may promote a better final-step candidate there; `last/` always keeps the final-step weights. `training_checkpoint.bin` stores the AdamW state for resuming. Validation covers every supervised token and reports separate `val-qa` and `val-text` batch progress. Set `save_validation_checkpoints` to `true` to retain every evaluated model in `weights/validation/step_XXXXXXXX`, which is useful when generation quality and validation loss do not peak at the same step.

`batches_per_epoch` only groups progress output and determines the displayed
epoch number. Epoch boundaries do not trigger validation or checkpoints;
those run strictly at their configured intervals and once on the final step.

Use `resume` only to continue an interrupted stage: it restores the step and
AdamW moments. Use `init_from` for curriculum training or supervised fine-tuning:
it copies compatible model weights but starts a fresh optimizer, warmup, and
learning-rate schedule.

```python
pretrain = train_from_config("pretrain.json", output_dir="pretrain_weights")
sft = train_from_config(
    "sft.json", output_dir="sft_weights", init_from=pretrain.weights_dir,
)
```

Every stage must use the same architecture and exact tokenizer. A shared BPE
artifact can be selected with `tokenizer_path`. Configure
`early_stopping_patience` and `early_stopping_min_delta` to stop after repeated
validation checks without a meaningful improvement. A useful curriculum first
trains the causal language objective on natural text or complete conversations,
then runs answer-only SFT with `qa_prompt_weight: 0`.

### SFT and persistence diagnostics

Before a long run, verify the complete answer-only SFT path on one QA pair:

```bash
mimillm-check-sft --backend python --output-dir diagnostics/one_pair_cpu
mimillm-check-sft --backend cuda --output-dir diagnostics/one_pair_cuda
```

This is a tiny 3.8K-parameter acceptance run, not project training. Its JSON
report contains the formatted prompt, token IDs, shifted targets, per-target
loss weights, EOS position, loss before and after training, the greedy answer,
artifact hashes, and the answer produced after loading in a fresh process.

To sample process RSS during repeated tiny training, validation, generation,
and atomic checkpoint saves, run:

```bash
python tools/memory_regression.py --backend cuda --steps 500
```

New `TransformerConfig` instances use a complete AdamW training stack by
default: `beta1=0.9`, `beta2=0.95`, `epsilon=1e-8`, global gradient clipping at
`1.0`, linear warmup, and cosine learning-rate decay to 10% of the base rate.
Configure it with `adam_beta1`, `adam_beta2`, `adam_epsilon`,
`gradient_clip_norm`, `learning_rate_schedule`, and
`min_learning_rate_ratio`. The schedule can be `cosine`, `linear`, or
`constant`. CUDA and C++ backends automatically use their native AdamW and
clipping implementations when available. These mechanisms form one optimizer
stack; different optimizers such as AdamW and SGD are not applied to the same
weights simultaneously.

Set `tie_word_embeddings: true` to reuse the input embedding matrix for output
logits. This is the default for newly constructed configurations and is useful
for spending a fixed parameter budget on a larger BPE vocabulary. Legacy JSON
configs without the field retain their separate output projection.

Dialogue JSONL is expanded turn by turn. When the second assistant message is trained, the first user/assistant pair is already in its attention context; later targets receive all preceding complete turns that fit the configured context. The loss targets the current assistant answer. This is the appropriate format for teaching a model to use chat history—unrelated single-turn QA records cannot provide that supervision.

See [examples/train_model.py](examples/train_model.py), [examples/use_weights.py](examples/use_weights.py), and [examples/staged_training](examples/staged_training).

### Tokenization and QA loss

`tokenizer: "byte"` uses a 260-token vocabulary and can represent any UTF-8 text. `tokenizer: "unicode"` uses a 355-token vocabulary: common Cyrillic characters have one token, while every other character still has a reversible UTF-8 byte fallback. `tokenizer: "bpe"` trains a byte-level subword vocabulary from your train data and stores it in `tokenizer.json`. The selected tokenizer is stored in `config.json` and is used automatically after `load_model`.

For one-off conversions, use the convenience API with a tokenizer name, a BPE artifact path, or an already loaded tokenizer object:

```python
from mimillm import detokenize, tokenize

tokens = tokenize("Hello, mimiLLM!", "unicode", add_bos=True, add_eos=True)
text = detokenize(tokens, "unicode")

bpe_tokens = tokenize("Hello!", "weights/tokenizer.json")
```

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

For a manually managed BPE model, train the tokenizer before training weights:

```python
from mimillm import train_tokenizer_from_config

tokenizer = train_tokenizer_from_config("config.json", vocab_size=2048)
print(tokenizer.VOCAB_SIZE)
```

This also writes `tokenizer_report.json` with tokens per UTF-8 byte and word,
raw-byte use, exact round-trip checks, vocabulary utilization, and atomic
Unicode coverage. New BPE training first reserves complete pieces for frequent
multi-byte characters, then learns frequency merges. Common Cyrillic letters
therefore stop being independent partial-byte training targets. A 2,000–4,000
token vocabulary is a reasonable starting range for a several-million-parameter
model, but the report on the final corpus should decide it.

Use the printed value as `vocab_size` if the training corpus is too small to fill the requested vocabulary. The staged pipeline applies the actual size automatically. A saved BPE model directory contains `config.json`, `model.safetensors`, and `tokenizer.json`.

New BPE artifacts use a Unicode-aware pre-tokenizer that separates words, numbers, and symbols while attaching horizontal whitespace to the following piece. This allows useful leading-space tokens such as `" model"`. The selected behavior is stored in `tokenizer.json`; version 1 BPE artifacts remain loadable with their original segmentation.

Generation masks PAD/BOS/SEP and token continuations that would create invalid
UTF-8. EOS is not accepted in the middle of a multi-byte character, preventing
visible replacement characters caused solely by truncated byte sequences.

The QA prompt always remains visible to attention. `qa_prompt_weight` optionally includes it in the language-model loss, while the two answer-prefix settings give the first answer tokens more influence. Defaults preserve the original answer-only objective. Changing the tokenizer or vocabulary size requires training new weights because the embedding and output shapes change.

Use `qa_source_weights` when QA files have different roles or very different
sizes. Keys are paths relative to `question_train_path`, every discovered QA
file must be listed, and values are normalized into sampling probabilities. A
zero value excludes that file from the current stage. For example:

```json
{
  "text_ratio": 0.05,
  "qa_source_weights": {
    "identity.jsonl": 0.55,
    "dialogues.jsonl": 0.35,
    "facts.jsonl": 0.10
  }
}
```

The remaining 5% is text replay. Validation uses the same source mixture, and
training prints the effective probabilities in the `Source mix` line.

### Backends and tests

The default `auto` mode selects CUDA first, then threaded C++, then pure Python. All backends use the same architecture, public API, checkpoints, and SafeTensors weights.

On Windows x64, the threaded C++ backend is bundled with the package. On Linux, including Raspberry Pi arm64, `pip install` automatically builds the native `.so` when `g++`, `clang++`, or `c++` is available.

CUDA mode requires an NVIDIA GPU, a compatible driver, and the NVIDIA CUDA Toolkit with NVRTC. This system dependency is explicit; CPU modes do not require it. Visual Studio, `cl.exe`, PyTorch, and TensorFlow are not required. NVRTC compiles the bundled kernels when CUDA is first selected in a process, and the NVIDIA Driver API loads them directly.

```bash
MIMILLM_BACKEND=cuda python train.py
```

The Python API can select it directly with `train_from_config(..., backend="cuda")`. Use `MIMILLM_DISABLE_CUDA=1` to skip CUDA in automatic mode.

The library has also been tested on a Raspberry Pi 5 running Ubuntu Server
24.04 arm64. The setup helper creates an isolated venv, installs mimiLLM, and
verifies the automatically built ARM64 C++ backend:

```bash
./scripts/setup_raspberry_pi.sh
```

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
