# mimiLLM 0.11.0.dev5 — 2026-09-15

Дополнение к основному `CHANGELOG.md` для ветки исправления обучения. При следующем релизном обновлении этот раздел можно перенести в общий журнал без изменения содержания.

## Русский

### Корректность AdamW

- Добавлено настоящее `gradient_accumulation_steps`: каждый microbatch сначала копируется/суммируется в отдельный gradient accumulator, а Adam moments, clipping, weight decay и обновление параметров выполняются только на границе accumulation-окна.
- Перед optimizer update накопленный градиент усредняется по числу microbatch.
- `AdamW.step_count` теперь считает настоящие optimizer updates, а не предъявления данных.
- Добавлен выборочный `decay_mask`; `TransformerConfig.weight_decay_exclude_1d=true` исключает одномерные bias/RMSNorm параметры из weight decay.
- Старое поведение сохранено по умолчанию: `gradient_accumulation_steps=1`, `weight_decay_exclude_1d=false`.

### Checkpoint

- Формат checkpoint поднят до v2.
- Checkpoint v2 сохраняет `gradient_accumulation_steps`, текущий `accumulation_count`, decay mask и незавершённые gradient accumulator buffers.
- Resume посередине accumulation-окна восстанавливает точное состояние оптимизатора.
- Reader продолжает принимать старые checkpoint v1.

### Проверки конфигурации

- `steps`, `validation_interval` и `checkpoint_interval` должны попадать на границу accumulation.
- Добавлены regression tests усреднения градиентов, selective weight decay, checkpoint resume и передачи optimizer hints из `TransformerConfig`.
- Добавлена отдельная CUDA-регрессия, сравнивающая accumulation с эталонным усреднённым update. На runner без NVIDIA она корректно пропускается.
- Добавлен GitHub Actions workflow для CPU correctness suite.

### Большие корпуса

- Удалена неиспользуемая плоская копия `TokenDataset.tokens`, которая дублировала ссылки на все токены корпуса.
- Добавлен `.mmtok` — read-only memory-mapped формат token shards.
- Для vocabulary до 65 536 token ID хранятся как uint16; для большего vocabulary используется uint32.
- Header shard содержит vocab size, token count и SHA-256 fingerprint точной конфигурации tokenizer. Shard от другого BPE отклоняется даже при одинаковом `vocab_size`.
- `TokenDataset` умеет выбирать окна напрямую из mmap без загрузки всего token stream в Python RAM.
- Raw `.txt/.md/.text` режим сохранён без изменения; implicit mixing raw и `.mmtok` запрещён.
- Добавлен `tools/tokenize_corpus.py`, который читает raw-текст ограниченными кусками и собирает shards с ограниченным RAM buffer.
- Добавлены end-to-end тесты writer/reader, recursive discovery, повреждённого файла, tokenizer mismatch, TokenDataset mmap sampling и потокового конвертера.

## English

### AdamW correctness

- Added real `gradient_accumulation_steps`. Adam moments, clipping, weight decay, and parameter updates run only at accumulation boundaries.
- Accumulated gradients are averaged before the optimizer update.
- `AdamW.step_count` now counts real optimizer updates.
- Added selective decay masks and `weight_decay_exclude_1d` for bias/RMSNorm-style one-dimensional parameters.
- Legacy behavior remains the default for existing configs.

### Checkpoints

- Checkpoint format is now v2 and persists incomplete accumulation windows.
- Resume restores accumulation count, pending gradients, decay mask, and Adam moments.
- v1 checkpoints remain readable.

### Large corpora

- Removed the unused flattened `TokenDataset.tokens` duplicate.
- Added tokenizer-bound `.mmtok` mmap shards using uint16/uint32 token IDs.
- Added bounded-memory raw-corpus conversion and mmap dataset regression tests.
