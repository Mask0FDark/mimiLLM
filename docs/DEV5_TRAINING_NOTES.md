# mimiLLM dev5: заметки по корректному обучению

Краткая памятка для проектов, которые переходят с `0.11.0.dev4` на `0.11.0.dev5`.

## Gradient accumulation

`steps` в текущем high-level training loop остаётся числом microbatch-предъявлений. Реальный AdamW update выполняется раз в `gradient_accumulation_steps` microstep.

Например:

```json
{
  "steps": 600000,
  "batch_size": 1,
  "context_length": 1024,
  "gradient_accumulation_steps": 32
}
```

означает:

- 614,4 млн предъявленных токенов;
- 32 768 токенов на optimizer update;
- 18 750 настоящих AdamW updates.

`steps`, `validation_interval` и `checkpoint_interval` должны делиться на accumulation без остатка.

## Weight decay

Для Transformer рекомендуется включать:

```json
{
  "weight_decay_exclude_1d": true
}
```

Тогда одномерные bias/RMSNorm scale не получают decay. Старые конфиги без поля сохраняют прежнее поведение ради воспроизводимости.

## Resume

Checkpoint v2 сохраняет незавершённое accumulation-окно. Поэтому корректная остановка и resume могут происходить между optimizer updates без потери уже накопленных microbatch-gradient.

При смене `gradient_accumulation_steps` или decay semantics старый training checkpoint нельзя считать эквивалентным новому запуску.

## Большие корпуса

Для больших pretrain corpus используйте `.mmtok` вместо Python `list[int]`. Shard хранит token ID как uint16/uint32 и читается через `mmap`.

Каждый shard привязан SHA-256 fingerprint к точной конфигурации tokenizer. Совпадение одного `vocab_size` недостаточно.

Создание shards из source checkout:

```powershell
python .\tools\tokenize_corpus.py `
  --config .\config.json `
  --split train `
  --output .\tokens\train
```

После этого `text_train_path` можно направить на каталог, содержащий только `.mmtok` shards, если конкретный training pipeline не использует этот же путь для raw-text audit/tokenizer training.

## CUDA

CPU regression suite не заменяет CUDA parity. На NVIDIA отдельно запускайте:

```powershell
python -m unittest tests.test_cuda_accumulation -v
```

Для архитектурного VRAM probe важно измерять память после первого microstep при accumulation > 1: именно тогда создаётся persistent accumulation buffer.
