"""Load the CUDA backend with Windows CUDA 13 DLL-layout compatibility."""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from . import backend_cuda as _backend_cuda


_ORIGINAL_NVRTC_LOAD_LIBRARY = _backend_cuda._Nvrtc._load_library
_ORIGINAL_CUBLAS_LOAD_LIBRARY = _backend_cuda._Cublas._load_library
_DLL_DIRECTORIES: list[Any] = []


def _windows_cuda_roots() -> list[Path]:
    """Return CUDA roots in priority order without dropping duplicates unsafely."""
    roots: list[Path] = []
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        if root := os.environ.get(variable):
            roots.append(Path(root))
    if nvcc := shutil.which("nvcc.exe"):
        roots.append(Path(nvcc).resolve().parent.parent)
    roots.append(Path(sys.prefix) / "Library")

    toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_root.is_dir():
        roots.extend(sorted(toolkit_root.glob("v*"), reverse=True))

    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = os.path.normcase(os.path.abspath(str(root)))
        if key not in seen:
            result.append(root)
            seen.add(key)
    return result


def _windows_cuda_bin_directories(root: Path) -> tuple[Path, Path]:
    """Search the CUDA 13 layout first, then the CUDA 12-and-older layout."""
    return root / "bin" / "x64", root / "bin"


def _load_nvrtc_windows(self: Any) -> ctypes.CDLL:
    candidates: list[Path] = []
    for root in _windows_cuda_roots():
        include_dir = root / "include"
        if include_dir.is_dir() and include_dir not in self.include_directories:
            self.include_directories.append(include_dir)
        for bin_dir in _windows_cuda_bin_directories(root):
            if not bin_dir.is_dir():
                continue
            self._dll_directories.append(os.add_dll_directory(str(bin_dir)))
            candidates.extend(sorted(bin_dir.glob("nvrtc64_*.dll"), reverse=True))

    for candidate in candidates:
        try:
            return ctypes.CDLL(str(candidate))
        except OSError:
            continue
    raise FileNotFoundError(
        "NVRTC was not found. Install the NVIDIA CUDA Toolkit and set CUDA_PATH."
    )


def _load_cublas_windows() -> ctypes.CDLL:
    candidates: list[Path] = []
    for root in _windows_cuda_roots():
        for bin_dir in _windows_cuda_bin_directories(root):
            if not bin_dir.is_dir():
                continue
            _DLL_DIRECTORIES.append(os.add_dll_directory(str(bin_dir)))
            candidates.extend(sorted(bin_dir.glob("cublas64_*.dll"), reverse=True))

    for candidate in candidates:
        try:
            return ctypes.WinDLL(str(candidate))
        except OSError:
            continue
    for name in ("cublas64_13.dll", "cublas64_12.dll"):
        try:
            return ctypes.WinDLL(name)
        except OSError:
            continue
    raise FileNotFoundError("cuBLAS was not found")


if sys.platform == "win32":
    # CUDA 13 moved Windows runtime DLLs from ``bin`` to ``bin\\x64``.
    # Keep support for both layouts without requiring users to copy vendor DLLs.
    _backend_cuda._Nvrtc._load_library = _load_nvrtc_windows
    _backend_cuda._Cublas._load_library = staticmethod(_load_cublas_windows)


CudaBackend = _backend_cuda.CudaBackend
is_available = _backend_cuda.is_available

__all__ = ["CudaBackend", "is_available"]
