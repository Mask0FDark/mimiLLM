# Состояние mimiLLM

## Готово

- Проект оформлен как устанавливаемая библиотека `mimillm` версии 0.2.0.
- Публичный API экспортирует модель, конфигурацию, Tensor/autograd, слои,
  оптимизаторы, датасеты, генерацию и checkpoint-функции.
- `create_model()` создаёт свою конфигурацию, `load_model()` восстанавливает
  архитектуру и веса одной командой, `generate_text()` работает с обычной строкой.
- Поддерживаются text-only, QA-only и смешанные QA/text датасеты на UTF-8.
- Реализованы decoder-only Transformer, causal attention, RMSNorm, embedding,
  MLP, SGD, AdamW, gradient clipping и динамический autograd.
- C++20 backend использует стабильный C ABI `mimillm_*` и постоянный thread pool;
  Python fallback не требует сторонних runtime-библиотек.
- Checkpoint `MIMILLM1` хранит конфигурацию, веса, состояние AdamW, step и seed
  без небезопасного `pickle`.
- Есть Conda-окружения для Windows и Linux, `pyproject.toml`, примеры, benchmark,
  инструменты сборки/проверки и двуязычный README.

## Проверка платформ

- Windows 11 x86-64: новое Conda-окружение `mimillm`, Python 3.12,
  MinGW-w64 GCC 15.2, editable-установка и 71/71 тест с C++ backend.
- Windows Python fallback: 71/71 тест.
- WSL Ubuntu x86-64: Python 3.12, GCC 13.3 и 71/71 тест с C++ backend.
- Исходники не используют x86 intrinsics и рассчитаны на сборку под AArch64,
  но физический ARM64-запуск в этой сессии не выполнялся.

## Контрольное обучение

Смешанный профиль дошёл до шага 2500. Последний train loss — `0.72131`,
weighted validation loss — `1.48545` (`qa=0.81922`, `text=2.72273`).
Это подтверждает работу train/resume и
смешивания источников, но небольшой демонстрационный корпус не создаёт качество
готовой разговорной модели.

## Ограничения

- Текущий backend ориентирован на обучение и эксперименты на CPU. Конфигурацию
  можно увеличить, но для миллиардов параметров нужен новый GPU/distributed backend.
- Byte tokenizer прост и универсален, но расходует несколько токенов на многие
  Unicode-символы.
- Windows debug build MinGW не использует ASan/UBSan из-за отсутствующих runtime;
  Linux debug build поддерживает эти санитайзеры.
