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

The pretraining stage also demonstrates `max_validation_loss`. If its best
validation loss remains above this limit, the pipeline stops before SFT. The
limit must be chosen from a baseline for the same tokenizer and dataset; it is
not a universal model-quality score.

Встроенный набор данных намеренно мал и подходит только для проверки механики.
Перед экспериментом с качеством модели замените данные во всех четырёх
каталогах.

`dialogue_eval.json` показывает проверку реальных ответов. После замены
сценариев на отсутствующие в train добавьте
`"evaluation": "dialogue_eval.json"` в этап `dialogue`. Встроенная крошечная
модель проходить эту проверку не обязана.

Этап предварительного обучения также показывает `max_validation_loss`. Если
его лучший validation loss остаётся выше порога, pipeline останавливается до
SFT. Порог нужно выбирать по базовому запуску с тем же токенизатором и набором
данных: универсальной оценкой качества модели он не является.

For a later follow-up curriculum, an already validated model directory can be
set as top-level `initial_weights`; the first listed stage may then be SFT. The
pipeline verifies architecture and tokenizer compatibility and resets the
optimizer and learning-rate schedule.

Для следующей цепочки уже проверенный каталог модели можно указать в
верхнеуровневом `initial_weights`; тогда первым этапом в списке может быть SFT.
Pipeline проверит совместимость архитектуры и токенизатора и начнёт с нового
optimizer и нового расписания learning rate.
