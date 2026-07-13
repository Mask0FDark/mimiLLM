# Состояние проекта mimiLLM / m0fdii

## Готово

- Реализована маленькая decoder-only Transformer с UTF-8 byte tokenizer.
- Реализованы собственные непрерывные float32 Tensor, динамический autograd,
  Parameter/Module, слои, causal attention, SGD и AdamW без внешних библиотек.
- Добавлен переносимый C++20 backend через стабильный C ABI/ctypes: elementwise,
  matmul/batched matmul, softmax, ReLU, embedding, cross-entropy и AdamW.
- Исправлена синхронизация постоянного `std::thread` pool и добавлен stress test.
- Работают train/validation, gradient clipping, warmup/decay, атомарный бинарный
  checkpoint, полное resume, generation и интерактивный chat.
- QA-примеры выбираются как отдельные последовательности, поэтому модель видит
  соответствие полного вопроса началу ответа, а не случайные куски общего файла.
- Добавлено смешанное обучение на QA и обычных UTF-8 `.txt`/`.md`/`.text`
  документах. `text_ratio` управляет долей text batch; train и validation
  корпуса разделены; в логе указывается источник каждого шага.
- В репозитории есть небольшой русский и английский text-корпус для проверки.
  Внешние каталоги подключаются повторяющимися аргументами CLI.
- Добавлены Windows Conda/MinGW и Linux Conda окружения, PowerShell/bash setup,
  benchmark, инспектор checkpoint, упаковщик исходного ZIP и подробная документация.

## Проверенные платформы

- Windows 11 x86-64, Conda Python 3.12, MinGW-w64 GCC 15.2: release DLL собрана,
  61/61 тестов прошли с C++ backend и 61/61 с Python fallback.
- WSL Ubuntu x86-64, Python 3.12, GCC 13.3: release `.so` собрана, 61/61 тестов
  прошли с C++ backend.
- Linux AArch64 поддерживается переносимым исходным кодом и флагами, но в этой
  сессии физическая ARM64-машина не была доступна для запуска.

## Результаты обучения

- QA-обучение до шага 1000: train loss на последнем шаге `1.19328`,
  QA validation loss `1.07530`.
- Смешанный debug run (QA + русский/английский текст): train loss
  `5.48604 → 5.13350`, weighted validation loss `5.34125 → 5.32229` за 4 шага.
- Продолжение демонстрационного checkpoint на смешанном корпусе выполняется
  отдельно; checkpoint и логи исключены из Git и исходного релиза.

## Известные ограничения

- Модель на 122 948 параметров и демонстрационный корпус показывают механизм,
  но пока не дают качество современной LLM и не должны описываться как ChatGPT.
- Для реального изучения языков нужны существенно больший проверенный корпус,
  больше контекст, параметров, шагов и вычислений.
- MinGW Conda для Windows не содержит runtime ASan/UBSan; Windows debug build
  использует символы и проверки компилятора без санитайзеров. Linux debug build
  поддерживает ASan/UBSan.
