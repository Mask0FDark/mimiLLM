# mimiLLM / m0fdii

`mimiLLM` is a GitHub-ready educational project; `m0fdii` is its first
reference model. It is a small decoder-only Transformer implemented from scratch with the
Python standard library and an optional C++20 CPU backend. See
[README_RU.md](README_RU.md) for the complete Russian documentation.

Training can mix structured question-answer examples with ordinary UTF-8 text
documents in any language. `text_ratio` controls the mixture, while separate
`data/text/train` and `data/text/validation` trees prevent accidental validation
leakage. Custom files or directories can be supplied with repeatable
`--text-train` and `--text-validation` options.

Verified on native Windows x86-64 (Conda MinGW-w64 GCC) and Linux x86-64;
the portable source also targets Linux AArch64. No NumPy, PyTorch, BLAS,
pybind11, OpenMP, CUDA, pip dependencies, or rule-based chat answers.

Windows quick start:

```powershell
conda env create -f environment-windows.yml
conda activate minillm
python tools/build_backend.py --release
python -m unittest discover -s tests -v
python train.py --config configs/mixed_demo.json --output checkpoints/mixed_demo.bin
```

Linux quick start:

```bash
conda env create -f environment.yml
conda activate minillm
python tools/build_backend.py --release
python -m unittest discover -s tests -v
python train.py --config configs/mixed_demo.json --output checkpoints/mixed_demo.bin
```
