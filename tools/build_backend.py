#!/usr/bin/env python3
"""Build the portable shared library directly, without CMake."""

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
        return "mimillm_backend.dll"
    if sys.platform == "darwin":
        return "libmimillm_backend.dylib"
    return "libmimillm_backend.so"


def find_compiler() -> str:
    """Find a C++ compiler, giving priority to the CXX environment variable."""
    requested = os.environ.get("CXX")
    if requested:
        path = shutil.which(requested) or (requested if Path(requested).exists() else None)
        if path:
            return str(path)
        # Conda's Windows cxx-compiler can select cl.exe even when Visual Studio
        # Build Tools is missing. In that case, allow MinGW from the same
        # environment to act as a fallback.
        if Path(requested).name.lower() not in {"cl", "cl.exe"}:
            raise SystemExit(f"CXX points to an unavailable compiler: {requested}")
        print(f"Warning: {requested} from CXX was not found; trying another toolchain.")
    candidates = ["g++", "clang++", "c++"]
    if sys.platform == "win32":
        candidates = ["x86_64-w64-mingw32-g++.exe", "g++.exe", "clang++.exe", "cl.exe"]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    raise SystemExit(
        "No C++ compiler was found. Activate the mimillm Conda environment "
        "or set CXX=/path/to/g++."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the mimiLLM C++ backend")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true", help="build with debug symbols")
    mode.add_argument("--release", action="store_true", help="build an optimized release library")
    parser.add_argument("--clean", action="store_true", help="remove the build directory before building")
    parser.add_argument(
        "--output", type=Path,
        help="output shared-library path (defaults to build/<platform name>)",
    )
    args = parser.parse_args()
    if args.clean:
        if BUILD.exists():
            shutil.rmtree(BUILD)
        print("Removed the build directory.")
        if not args.debug and not args.release:
            return
    compiler = find_compiler()
    BUILD.mkdir(parents=True, exist_ok=True)
    output = args.output or (BUILD / library_name())
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
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
        if os.environ.get("MIMILLM_NATIVE") == "1":
            print("MIMILLM_NATIVE is ignored for MSVC to keep a portable baseline.")
    else:
        command = [
            compiler, "-std=c++20", "-shared", "-pthread", "-Wall", "-Wextra", "-Wpedantic",
            *sources, "-o", str(output),
        ]
        if sys.platform != "win32":
            command.insert(2, "-fPIC")
        else:
            # Keep the DLL self-contained so ctypes does not need Conda runtime DLLs.
            command[1:1] = ["-static-libgcc", "-static-libstdc++"]
        if args.debug:
            debug_flags = ["-O0", "-g", "-fno-omit-frame-pointer"]
            if sys.platform != "win32":
                debug_flags.insert(2, "-fsanitize=address,undefined")
            else:
                print(
                    "Warning: Conda MinGW does not provide the ASan/UBSan runtime; "
                    "the Windows debug build includes symbols without sanitizers."
                )
            command[1:1] = debug_flags
        else:
            command[1:1] = ["-O3", "-DNDEBUG"]
        if os.environ.get("MIMILLM_NATIVE") == "1":
            command.insert(1, "-march=native")
            print("Warning: -march=native makes the library non-portable across CPUs.")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print("Command:", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Built: {output}")


if __name__ == "__main__":
    main()
