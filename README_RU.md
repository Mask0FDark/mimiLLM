# mimiLLM: m0fdii — маленькая языковая модель с нуля

`mimiLLM` — учебный проект для GitHub, а `m0fdii` — его первая эталонная
модель. Компоненты разделены так, чтобы читатель мог менять Tensor, слои,
архитектуру, датасет и training loop и на этой основе собирать собственные
нейросети, а не только запускать один готовый checkpoint.

`m0fdii` — генеративная языковая модель архитектуры decoder-only
Transformer. Она обучается предсказывать следующий byte-token и генерирует
продолжение по одному токену. В `chat.py` нет таблицы ответов и проверок текста
вопроса: единственные `if` там обрабатывают команды `/exit`, `/help`, `/reset` и
`/settings`.

Это настоящая autoregressive language model, но не «маленький ChatGPT».
Демонстрационная конфигурация содержит 122 948 параметров, тогда как у
промышленных LLM их миллиарды. Проект предназначен для изучения всего стека:
UTF-8 tokenization, float32 Tensor, динамический autograd, Transformer, AdamW,
бинарный checkpoint, C ABI, многопоточные CPU kernels и генерация.

## Ограничения и зависимости

Python-код использует только стандартную библиотеку: `array`, `ctypes`,
`struct`, `json`, `math`, `random` и другие встроенные модули. C++ backend
использует только C++20 и стандартную библиотеку. В проекте нет NumPy, PyTorch,
TensorFlow, BLAS, OpenMP, pybind11, CUDA и `pip install`.

Проверенные платформы:

- Windows 11 x86-64, Conda Python 3.12, MinGW-w64 GCC 15.2;
- Linux x86-64 (WSL Ubuntu), Python 3.12, GCC 13.3;
- Linux ARM64/AArch64 поддерживается исходным кодом и флагами сборки, но в
  текущей сессии физическая ARM-машина не была доступна для запуска.

## Быстрый старт в Windows PowerShell

Для Windows используется отдельный файл, потому что `cxx-compiler` в обычном
`environment.yml` ожидает внешнюю установку Visual Studio. Windows-вариант
устанавливает полноценный MinGW-w64 C++ compiler прямо в Conda.

```powershell
conda env create -f environment-windows.yml
conda activate minillm
python tools/build_backend.py --release
python -m unittest discover -s tests -v
python tools/make_dataset.py
python train.py --config configs/debug.json
python evaluate.py --checkpoint checkpoints/debug.bin --data data/validation.txt
python chat.py --checkpoint checkpoints/debug.bin
```

Эквивалентный помощник для создания или обновления окружения:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_conda.ps1
```

## Быстрый старт в Linux x86-64 или ARM64

```bash
conda env create -f environment.yml
conda activate minillm
python tools/build_backend.py --release
python -m unittest discover -s tests -v
python tools/make_dataset.py
python train.py --config configs/debug.json
python evaluate.py --checkpoint checkpoints/debug.bin --data data/validation.txt
python chat.py --checkpoint checkpoints/debug.bin
```

Помощник `bash scripts/setup_conda.sh` проверяет Conda и создаёт окружение, если
его нет. Он не устанавливает Python-пакеты из PyPI и не меняет системные файлы.

## Сборка C++ backend

```bash
python tools/build_backend.py --release
python tools/build_backend.py --debug
python tools/build_backend.py --clean
```

Linux release использует `-O3 -DNDEBUG -std=c++20 -fPIC -shared -pthread` и
предупреждения компилятора. Linux debug добавляет ASan/UBSan. Conda MinGW для
Windows не поставляет runtime этих санитайзеров, поэтому Windows debug использует
`-O0 -g -fno-omit-frame-pointer` и сообщает об этом явно. Release DLL статически
линкует GCC/libstdc++, чтобы `ctypes` не искал runtime DLL вне `build/`.

Скрипт уважает `CXX`. Без неё он ищет GCC, Clang, MinGW и MSVC. На Windows
поддерживаются как `x86_64-w64-mingw32-g++.exe`, так и `cl.exe`. По умолчанию
нет `-march=native`, x86 intrinsics и assembly, поэтому один и тот же исходник
собирается для x86-64 и AArch64. Опционально:

```bash
MINILLM_NATIVE=1 python tools/build_backend.py --release
```

Такая библиотека может не запуститься на другом процессоре.

## Выбор backend и потоков

Если переменная не задана, m0fdii пытается загрузить `.so`, `.dll` или `.dylib`
из `build/`, а при отсутствии выдаёт предупреждение и использует Python.

Linux/macOS:

```bash
MINILLM_BACKEND=python python train.py --config configs/debug.json
MINILLM_BACKEND=cpp MINILLM_NUM_THREADS=4 python train.py --config configs/debug.json
```

Windows PowerShell:

```powershell
$env:MINILLM_BACKEND = "cpp"
$env:MINILLM_NUM_THREADS = "4"
python train.py --config configs/debug.json
```

Постоянный C++ thread pool создаётся один раз. Малые операции выполняются одним
потоком, потому что очередь задач может стоить дороже вычисления. На больших
матрицах работа делится по строкам; batched matmul — по парам batch/row.

## Датасет

Обучение поддерживает два источника одновременно:

- QA-пары учат модель продолжать конкретный вопрос ответом;
- обычные документы учат порядку слов, грамматике, стилю и общим языковым
  закономерностям через ту же задачу предсказания следующего токена.

`data/qa_seed.jsonl` содержит русские вопросы, ответы и варианты формулировок.
Генератор детерминированно создаёт 76 train и 14 validation QA-примеров:

```bash
python tools/make_dataset.py --seed 42
```

Чтобы добавить знания, внесите новые JSON-строки в `qa_seed.jsonl`, затем снова
создайте dataset и переобучите модель. Ответы одной записи должны подходить ко
всем её `variants`; вопросы с ответами «да» и «нет» лучше хранить отдельно.
Validation отделена до tokenization и не участвует в optimizer step.

Обычные UTF-8 документы находятся отдельно:

```text
data/text/train/         тексты, участвующие в optimizer step
data/text/validation/    независимые тексты только для оценки
```

Загрузчик рекурсивно читает `.txt`, `.md` и `.text`. Один файл считается одним
документом; документы не склеиваются на границе. В репозитории есть небольшой
русско-английский корпус, который проверяет весь pipeline. Для осмысленного
изучения языка его нужно заменить или дополнить гораздо большим, качественным
и юридически пригодным корпусом. Не смешивайте одинаковые документы между
train и validation и не добавляйте секретные или персональные данные.

`text_ratio` в JSON-конфигурации задаёт вероятность text batch. Например,
`0.35` означает приблизительно 35% шагов с обычным текстом и 65% с QA.
`0` включает только QA, `1` — только обычные тексты. Источник выбирается для
всего batch, поэтому строки в нём имеют одинаковую природу; внутри текстового
корпуса документы выбираются пропорционально числу доступных окон.

Собственные каталоги и долю можно передать без изменения кода:

```powershell
python train.py --config configs/tiny.json `
  --text-train D:\corpus\ru --text-train D:\corpus\en `
  --text-validation D:\corpus-validation --text-ratio 0.7
```

В bash вместо обратных апострофов используйте `\`. Параметры `--text-train` и
`--text-validation` можно повторять; они принимают как каталог, так и отдельный
файл. Все входные файлы декодируются строго как UTF-8, чтобы повреждённые данные
не проходили незаметно.

Byte tokenizer использует:

- `0..255` — байты UTF-8;
- `256` — PAD;
- `257` — BOS;
- `258` — EOS;
- `259` — SEP.

QA-пример имеет вид `<BOS>Вопрос: ...\nОтвет: ...<EOS>`, обычный документ —
`<BOS>текст документа<EOS>`. Byte-level подход не требует внешнего словаря и
без потерь представляет русский, английский и другие UTF-8 языки. При этом один
русский символ обычно занимает два токена, поэтому контекст 96 короче, чем
кажется.

## Обучение и продолжение

```bash
python train.py --config configs/debug.json
python train.py --config configs/qa_demo.json --output checkpoints/qa_demo.bin
python train.py --config configs/mixed_demo.json --output checkpoints/mixed_demo.bin
python train.py --resume checkpoints/qa_demo.bin
python train.py --resume checkpoints/mixed_demo.bin --steps 3000
```

Цикл сначала выбирает QA или text согласно `text_ratio`, затем причинные окна,
считает logits и cross-entropy, запускает
`backward()`, ограничивает общую норму градиента, делает AdamW step, отдельно
считает validation и периодически атомарно сохраняет checkpoint. В логе есть
step, источник `qa`/`text`, train/validation loss, tokens/s, время, learning
rate, backend и потоки. Validation loss — взвешенная тем же `text_ratio` сумма
отдельных детерминированных QA и text batch.
Learning rate проходит линейный warmup, затем уменьшается до 10% начального.
При `Ctrl+C` создаётся файл `*_interrupted.bin`.
`--steps` задаёт новый общий предел и позволяет осознанно продлить готовый
checkpoint; это не число дополнительных шагов.
Validation и generation выполняются внутри собственного `no_grad()` и не
создают ненужный граф обратного распространения; следующий training step снова
автоматически включает autograd.

`configs/debug.json` нужен для быстрой смешанной проверки. `qa_demo.json` —
QA-only модель 96/64/2/4/192 на 1000 шагов. `mixed_demo.json` использует ту же
наглядную архитектуру, 2500 шагов и `text_ratio=0.35`; это рекомендуемый пример
для совместного изучения ответов и языка. `tiny.json` крупнее и автоматически в
тестах не запускается. Для увеличения модели меняйте `d_model`, `n_layers`, `n_heads`,
`d_mlp`, `context_length`, `batch_size`, `steps` и `text_ratio`. `d_model` обязан делиться на
`n_heads`; ошибочные и неизвестные поля конфигурации отклоняются.

## Checkpoint

Формат `M0FDIICK`, версия 1, состоит из фиксированного заголовка, UTF-8 JSON
схемы и последовательных little-endian float32-буферов. Он хранит конфигурацию,
имена и формы параметров, веса, оба момента AdamW, step и seed. `pickle` не
используется. Loader проверяет magic, версию, размеры, дубликаты имён, ранний и
лишний конец файла.

```bash
python tools/inspect_checkpoint.py checkpoints/debug.bin
```

## Оценка и чат

```bash
python evaluate.py --checkpoint checkpoints/debug.bin --data data/validation.txt \
  --text-data data/text/validation
python chat.py --checkpoint checkpoints/mixed_demo.bin --temperature 0.7 --top-k 20 --max-new-tokens 80
```

`temperature=0` или `top-k=1` включает greedy generation. Генерация использует
последнее доступное контекстное окно и останавливается на EOS. `/reset` очищает
только отображаемую историю, но не изменяет веса. В первой версии история не
добавляется в prompt, поэтому каждый вопрос независим.

## Как устроен Tensor и autograd

Tensor хранит плотно упакованный `array('f')`, форму, contiguous strides,
`requires_grad`, float32-градиент, родителей, имя операции и локальную backward
функцию. `transpose` и `permute` создают копию: это расходует память, зато каждый
C++ kernel всегда получает простой непрерывный указатель.

Операция создаёт новый Tensor и замыкание с локальной производной. `backward()`
обходит граф в топологическом порядке от родителей к результату, затем идёт в
обратном направлении. `_accumulate_grad` суммирует повторные пути, поэтому для
`x*x+x` получается правильный градиент. Matmul backward использует
`dA=dY·Bᵀ`, `dB=Aᵀ·dY`; embedding делает scatter-add для повторных индексов.
Тесты сравнивают эти производные с центральными конечными разностями float32.

Поддерживается только нужное модели broadcasting и только softmax по последней
оси. Это намеренное ограничение, а не попытка повторить NumPy.

## Как работает Transformer

Входной byte-token складывается с обучаемым positional embedding. Каждый
pre-norm блок выполняет:

```text
x = x + attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

Attention строит Q, K, V, разбивает embedding на головы и вычисляет
`softmax(QKᵀ / sqrt(head_dim) + causal_mask)V`. Значение `-1e9` над диагональю
не позволяет видеть будущие токены. Отдельный тест меняет будущие token id и
доказывает, что logits предыдущих позиций остаются теми же. После блоков идут
финальный RMSNorm и Linear в 260 logits.

Cross-entropy учит правильный следующий байт. При генерации из последней строки
logits выбирается следующий token, он добавляется ко входу, и процесс повторяется.

## Связь Python и C++

`backend_api.h` экспортирует только `extern "C"` функции с указателями,
фиксированными `int32/int64`, кодом возврата и `minillm_last_error()`. Исключения
не пересекают ABI. `backend_cpp.py` задаёт `ctypes.argtypes`, удерживает
`array('f')` живым на время синхронного вызова, проверяет размеры до C++ и
преобразует код ошибки в Python exception.

C++ ускоряет matmul, batched matmul, softmax, ReLU, embedding, cross-entropy и
AdamW. Эталонные Python kernels остаются для проверки и работы без компилятора.
GPU пока отсутствует: корректный CPU/autograd стек важнее преждевременной CUDA,
а CUDA нарушила бы цель «только стандартные средства».

## Тесты и benchmark

```bash
python -m unittest discover -s tests -v
python tools/benchmark.py --size 96 --threads 4 --repeats 3
```

Тесты охватывают tokenizer, Tensor, numerical gradients, C ABI, 1/несколько
потоков, слои, causal mask, SGD/AdamW, checkpoint corruption, overfit одного
batch, resume и EOS generation. Benchmark выводит реальные времена Python/C++
matmul, speedup, потоки, архитектуру, Python/compiler, forward и training step.
На маленьких матрицах несколько потоков закономерно могут быть медленнее одного.

## Возможные ошибки на ARM Linux

- `C++ compiler не найден`: активируйте Conda-окружение или задайте `CXX`;
- `wrong ELF class`: библиотека собрана для другой архитектуры — удалите
  `build/` и пересоберите непосредственно на ARM64;
- `illegal instruction`: была использована чужая сборка с `MINILLM_NATIVE=1`;
- мало памяти: уменьшите batch/context/d_model; transpose пока создаёт копии;
- медленнее ожидаемого: проверьте `MINILLM_BACKEND=cpp` и число потоков.

Бинарные `.so` между x86-64 и ARM64 не переносимы, checkpoint float32 переносим.

## Почему модель может отвечать неправильно

- датасет очень мал и тематически узок;
- параметров на порядки меньше, чем у большой LLM;
- нет предварительного обучения на большом корпусе;
- byte-контекст ограничен;
- генерация вероятностна и может создать некорректную UTF-8 последовательность;
- модель не подключена к интернету и не проверяет факты;
- даже низкий training loss не гарантирует обобщение;
- фактическая точность ответов не гарантируется.

Для осмысленного QA checkpoint нужны сотни полных шагов. `debug.bin` проверяет
технику, но после четырёх шагов закономерно генерирует бессмысленный текст.

## Структура проекта

```text
configs/              размеры модели и обучения
data/                 QA и раздельные многоязычные text train/validation
minillm/              Tensor, autograd, слои, Transformer, optimizer, generation
cpp/                  C ABI, kernels и постоянный thread pool
tools/                сборка, dataset, benchmark, inspection, release ZIP
scripts/              Linux shell и Windows PowerShell помощники
tests/                быстрые unittest без длительного QA-обучения
train.py              обучение и resume
evaluate.py           отдельная validation
chat.py               интерактивная генерация
```

Проект распространяется по лицензии MIT.
