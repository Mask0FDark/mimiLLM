# Data / Данные

## English

mimiLLM can train on two compatible next-token tasks:

1. `qa_seed.jsonl` is reproducibly split into `train.txt` and `validation.txt`;
2. ordinary documents live in `text/train` and `text/validation`.

Rebuild the QA split with:

```bash
python tools/make_dataset.py --seed 42
```

Each JSONL record contains `question`, `answer`, and an optional `variants`
array. The text corpus loader recursively reads UTF-8 `.txt`, `.md`, and `.text`
files. Each file stays a separate document.

The included Russian and English documents are deliberately small. They test
the pipeline; they do not teach a useful language model. Keep training and
validation documents disjoint, check licenses, and never add secrets or private
data to a corpus.

## Русский

mimiLLM обучается на двух совместимых задачах предсказания следующего токена:

1. `qa_seed.jsonl` воспроизводимо делится на `train.txt` и `validation.txt`;
2. обычные документы лежат отдельно в `text/train` и `text/validation`.

Пересоздание QA-файлов:

```bash
python tools/make_dataset.py --seed 42
```

Каждая JSONL-строка содержит `question`, `answer` и необязательный массив
`variants`. Загрузчик текстов рекурсивно читает UTF-8 файлы `.txt`, `.md` и
`.text`; каждый файл остаётся отдельным документом.

Русские и английские примеры в репозитории намеренно малы и нужны только для
проверки pipeline. Не смешивайте документы между train и validation, проверяйте
лицензии и не добавляйте секретные или персональные данные.
