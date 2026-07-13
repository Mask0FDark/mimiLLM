# Состояние проекта m0fdii

## Готово

- Создан Git-репозиторий и базовая структура проекта.
- Добавлены Conda-окружение, setup-скрипт, лицензия и начальная документация.
- Реализован UTF-8 byte-level токенизатор с 260 токенами.
- Добавлены 30 исходных QA-пар, варианты формулировок и детерминированное разделение.
- Реализован непрерывный float32 Tensor, reshape/transpose, broadcasting,
  редукции, matmul/batched matmul, softmax и базовые нелинейности.
- Добавлен эталонный набор Python-ядер без внешних зависимостей.

## Сейчас выполняется

- Этап 3: численная проверка autograd, Linear и SGD.

## Осталось

- Autograd-проверки, C++ backend, Transformer, обучение, checkpoint и генерация.

## Известные проблемы

- В установленном WSL Ubuntu пока не найден системный C++ компилятор.

## Последний результат тестов

- `python -m unittest tests.test_tokenizer -v`: 7 тестов, OK.
- `python -m unittest tests.test_tensor tests.test_tokenizer -v`: 16 тестов, OK.
- `python tools/make_dataset.py`: train=76, validation=14.
