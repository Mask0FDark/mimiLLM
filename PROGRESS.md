# Состояние проекта m0fdii

## Готово

- Создан Git-репозиторий и базовая структура проекта.
- Добавлены Conda-окружение, setup-скрипт, лицензия и начальная документация.
- Реализован UTF-8 byte-level токенизатор с 260 токенами.
- Добавлены 30 исходных QA-пар, варианты формулировок и детерминированное разделение.
- Реализован непрерывный float32 Tensor, reshape/transpose, broadcasting,
  редукции, matmul/batched matmul, softmax и базовые нелинейности.
- Добавлен эталонный набор Python-ядер без внешних зависимостей.
- Autograd проверен центральными конечными разностями для broadcasting,
  matmul, ReLU, cross-entropy, embedding и Linear.
- Реализованы Parameter, Module, Linear, ReLU, SGD и clipping общей нормы.
- Собран Linux x86-64 C++20 backend через C ABI/ctypes; Python fallback сохранён.
- Реализованы kernels matmul/batched matmul/softmax/ReLU/embedding/CE/AdamW
  и постоянный std::thread pool.
- Добавлены Embedding, RMSNorm, causal multi-head attention, pre-norm блок и
  настраиваемый DecoderTransformer с обучаемыми позициями.

## Сейчас выполняется

- Этап 6: AdamW, обучение, validation и checkpoint/resume.

## Осталось

- Обучение/checkpoint, генерация, benchmark и финальный прогон.

## Известные проблемы

- В установленном WSL Ubuntu пока не найден системный C++ компилятор.

## Последний результат тестов

- `python -m unittest tests.test_tokenizer -v`: 7 тестов, OK.
- `python -m unittest tests.test_tensor tests.test_tokenizer -v`: 16 тестов, OK.
- `python -m unittest tests.test_autograd tests.test_layers tests.test_optimizer -v`:
  12 тестов, OK; все численные gradient checks прошли.
- WSL Linux: release `.so` собрана GCC 13; `tests.test_cpp_backend`: 7 тестов, OK.
- `tests.test_transformer tests.test_layers tests.test_autograd`: 16 тестов, OK;
  causal mask доказана сравнением logits до изменённой будущей позиции.
- `python tools/make_dataset.py`: train=76, validation=14.
