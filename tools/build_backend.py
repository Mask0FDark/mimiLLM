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
        # Conda cxx-compiler на Windows может активировать cl.exe даже когда
        # внешний Visual Studio Build Tools ещё не установлен. Тогда даём
        # установленному в том же окружении MinGW реальный шанс на fallback.
        if Path(requested).name.lower() not in {"cl", "cl.exe"}:
            raise SystemExit(f"CXX указывает на недоступный компилятор: {requested}")
        print(f"Предупреждение: {requested} из CXX не найден; ищу другой toolchain.")
    candidates = ["g++", "clang++", "c++"]
    if sys.platform == "win32":
        candidates = ["x86_64-w64-mingw32-g++.exe", "g++.exe", "clang++.exe", "cl.exe"]
    for candidate in candidates:
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
    sources = [
        str(ROOT / "cpp" / "backend_api.cpp"), str(ROOT / "cpp" / "kernels.cpp"),
        str(ROOT / "cpp" / "thread_pool.cpp"),
    ]
    compiler_name = Path(compiler).name.lower()
    is_msvc = compiler_name in {"cl", "cl.exe"}
    if is_msvc:
        command = [compiler, "/nologo", "/std:c++20", "/EHsc", "/LD", "/W4"]
        command += ["/Od", "/Zi"] if args.debug else ["/O2", "/DNDEBUG"]
        command += [*sources, f"/Fe:{output}"]
        if os.environ.get("MINILLM_NATIVE") == "1":
            print("MINILLM_NATIVE игнорируется для MSVC: переносимый baseline сохраняется.")
    else:
        command = [
            compiler, "-std=c++20", "-shared", "-pthread", "-Wall", "-Wextra", "-Wpedantic",
            *sources, "-o", str(output),
        ]
        if sys.platform != "win32":
            command.insert(2, "-fPIC")
        else:
            # DLL остаётся самодостаточной без поиска conda runtime DLL при ctypes-load.
            command[1:1] = ["-static-libgcc", "-static-libstdc++"]
        if args.debug:
            debug_flags = ["-O0", "-g", "-fno-omit-frame-pointer"]
            if sys.platform != "win32":
                debug_flags.insert(2, "-fsanitize=address,undefined")
            else:
                print(
                    "Предупреждение: Conda MinGW не поставляет ASan/UBSan runtime; "
                    "Windows debug собирается с символами без санитайзеров."
                )
            command[1:1] = debug_flags
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
