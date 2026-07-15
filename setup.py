"""Build platform wheels containing native CPU and runtime CUDA sources."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from shutil import copy2

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel


ROOT = Path(__file__).resolve().parent


def native_library_name() -> str:
    if sys.platform == "win32":
        return "mimillm_backend.dll"
    if sys.platform == "darwin":
        return "libmimillm_backend.dylib"
    return "libmimillm_backend.so"


def find_unix_compiler() -> str | None:
    requested = os.environ.get("CXX")
    if requested:
        return shutil.which(requested) or (
            requested if Path(requested).is_file() else None
        )
    for candidate in ("g++", "clang++", "c++"):
        compiler = shutil.which(candidate)
        if compiler:
            return compiler
    return None


class BuildPythonWithNativeBackend(build_py):
    """Include CUDA source and build the portable CPU library on Unix."""

    def run(self) -> None:
        super().run()
        native_dir = Path(self.build_lib) / "mimillm" / "_native"
        native_dir.mkdir(parents=True, exist_ok=True)
        copy2(ROOT / "cuda" / "kernels.cu", native_dir / "mimillm_cuda_kernels.cu")
        if sys.platform == "win32" or os.environ.get("MIMILLM_SKIP_NATIVE_BUILD") == "1":
            return
        compiler = find_unix_compiler()
        if compiler is None:
            self.announce(
                "No C++ compiler found; the wheel will use the Python backend.", level=2,
            )
            return
        output = native_dir / native_library_name()
        sources = [
            ROOT / "cpp" / "backend_api.cpp",
            ROOT / "cpp" / "kernels.cpp",
            ROOT / "cpp" / "thread_pool.cpp",
        ]
        command = [
            compiler,
            "-O3",
            "-DNDEBUG",
            "-std=c++20",
            "-fPIC",
            "-shared",
            "-pthread",
            *(str(source) for source in sources),
            "-o",
            str(output),
        ]
        if os.environ.get("MIMILLM_NATIVE") == "1":
            command.insert(1, "-march=native")
            self.announce(
                "Building with -march=native; this wheel is specific to the build CPU.",
                level=2,
            )
        self.announce(
            f"Building mimiLLM native backend: {' '.join(command)}", level=2,
        )
        subprocess.run(command, cwd=ROOT, check=True)


class NativeBinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


class NativeBinaryWheel(bdist_wheel):
    def get_tag(self) -> tuple[str, str, str]:
        _python, _abi, platform = super().get_tag()
        return "py3", "none", platform


options = {
    "distclass": NativeBinaryDistribution,
    "cmdclass": {
        "bdist_wheel": NativeBinaryWheel,
        "build_py": BuildPythonWithNativeBackend,
    },
}
if sys.platform == "win32":
    setup(
        **options,
    )
else:
    setup(
        exclude_package_data={"mimillm": ["_native/*.dll"]},
        **options,
    )
