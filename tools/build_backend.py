#!/usr/bin/env python3
"""Собирает переносимую shared library напрямую, без CMake."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


def library_name() -> str:
    if sys.platform == "win32":
        return "minillm_backend.dll"
    if sys.platform == "darwin":
        return "libminillm_backend.dylib"
    return "libminillm_backend.so"


def find_compiler() -> str:
    """Находит C++ compiler, отдавая приоритет переменной CXX."""
    requested = os.environ.get("CXX")
    if requested:
        path = shutil.which(requested) or (requested if Path(requested).exists() else None)
        if path:
            return str(path)
        raise SystemExit(f"CXX указывает на недоступный компилятор: {requested}")
    for candidate in ("g++", "clang++", "c++"):
        path = shutil.which(candidate)
        if path:
            return path
    raise SystemExit(
        "C++ compiler не найден. Активируйте Conda-окружение minillm "
        "или задайте CXX=/путь/к/g++."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка C++ backend m0fdii")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true")
    mode.add_argument("--release", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean:
        if BUILD.exists():
            shutil.rmtree(BUILD)
        print("Каталог build удалён.")
        if not args.debug and not args.release:
            return
    compiler = find_compiler()
    BUILD.mkdir(parents=True, exist_ok=True)
    output = BUILD / library_name()
    command = [
        compiler, "-std=c++20", "-shared", "-pthread", "-Wall", "-Wextra", "-Wpedantic",
        str(ROOT / "cpp" / "backend_api.cpp"), str(ROOT / "cpp" / "kernels.cpp"),
        str(ROOT / "cpp" / "thread_pool.cpp"),
        "-o", str(output),
    ]
    if sys.platform != "win32":
        command.insert(2, "-fPIC")
    if args.debug:
        command[1:1] = ["-O0", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    else:
        command[1:1] = ["-O3", "-DNDEBUG"]
    if os.environ.get("MINILLM_NATIVE") == "1":
        command.insert(1, "-march=native")
        print("ВНИМАНИЕ: -march=native делает библиотеку непереносимой между CPU.")
    print(f"Платформа: {platform.system()} {platform.machine()}")
    print("Команда:", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Собрано: {output}")


if __name__ == "__main__":
    main()
