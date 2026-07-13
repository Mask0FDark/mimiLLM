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

## Сейчас выполняется

- Этап 4: переносимые C++20 kernels, C ABI и ctypes backend.

## Осталось

- C++ backend, Transformer, обучение, checkpoint и генерация.

## Известные проблемы

- В установленном WSL Ubuntu пока не найден системный C++ компилятор.

## Последний результат тестов

- `python -m unittest tests.test_tokenizer -v`: 7 тестов, OK.
- `python -m unittest tests.test_tensor tests.test_tokenizer -v`: 16 тестов, OK.
- `python -m unittest tests.test_autograd tests.test_layers tests.test_optimizer -v`:
  12 тестов, OK; все численные gradient checks прошли.
- `python tools/make_dataset.py`: train=76, validation=14.
