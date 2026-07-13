# История изменений

## 0.1.0 — 2026-07-13

- Реализована маленькая decoder-only Transformer модель и byte tokenizer.
- Добавлены собственные float32 Tensor, autograd, SGD, AdamW и checkpoint.
- Добавлен C++20 backend через ctypes с постоянным thread pool.
- Проверены нативные Windows x86-64 и Linux x86-64 сборки.
- Добавлены QA dataset, train/resume/evaluate, генерация и интерактивный чат.
- Добавлено смешанное обучение на QA и обычных многоязычных UTF-8 документах,
  отдельная text validation и настраиваемый `text_ratio`.
- Добавлены unittest, benchmark, Conda-окружения и русская документация.
