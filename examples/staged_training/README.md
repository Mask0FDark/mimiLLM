# Staged training example / Пример поэтапного обучения

This example keeps shared model and optimizer fields in `model.json`.
`pretrain.json` and `sft.json` contain only stage-specific overrides, while
`pipeline.json` defines their order, shared tokenizer, data checks, and output
directories.

В этом примере общие размеры модели и настройки оптимизатора находятся в
`model.json`. Файлы `pretrain.json` и `sft.json` содержат только отличия этапов,
а `pipeline.json` задаёт порядок, общий токенизатор, проверки данных и каталоги
весов.

From the repository root / Из корня репозитория:

```powershell
python train_pipeline.py examples/staged_training/pipeline.json --backend auto
```

The bundled dataset is intentionally tiny and is suitable only for checking the
workflow. Replace all four data directories before treating the result as a
model-quality experiment.

`dialogue_eval.json` demonstrates a held-out generation gate. Add
`"evaluation": "dialogue_eval.json"` to the `dialogue` stage after replacing
its cases with prompts that do not occur in training. The tiny bundled model is
not expected to pass it.

Встроенный набор данных намеренно мал и подходит только для проверки механики.
Перед экспериментом с качеством модели замените данные во всех четырёх
каталогах.

`dialogue_eval.json` показывает проверку реальных ответов. После замены
сценариев на отсутствующие в train добавьте
`"evaluation": "dialogue_eval.json"` в этап `dialogue`. Встроенная крошечная
модель проходить эту проверку не обязана.
