"""Select the CUDA, C++, or reference Python backend."""

from __future__ import annotations

import os
import warnings
from typing import Any

from . import backend_python


_backend: Any | None = None


def get_backend() -> Any:
    """Select a backend lazily according to MIMILLM_BACKEND."""
    global _backend
    if _backend is not None:
        return _backend
    requested = os.environ.get("MIMILLM_BACKEND", "auto").lower()
    if requested not in {"auto", "python", "cpp", "cuda"}:
        raise ValueError("MIMILLM_BACKEND must be auto, cuda, cpp, or python")
    cuda_error: Exception | None = None
    cuda_disabled = os.environ.get("MIMILLM_DISABLE_CUDA", "0") == "1"
    if requested == "cuda" or (requested == "auto" and not cuda_disabled):
        try:
            from .backend_cuda import CudaBackend

            _backend = CudaBackend()
            return _backend
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            cuda_error = exc
            if requested == "cuda":
                raise RuntimeError(f"The requested CUDA backend is unavailable: {exc}") from exc
    if requested in {"auto", "cpp"}:
        try:
            from .backend_cpp import CppBackend
            _backend = CppBackend()
            return _backend
        except (FileNotFoundError, OSError) as exc:
            if requested == "cpp":
                raise RuntimeError(f"The requested C++ backend is unavailable: {exc}") from exc
            cuda_detail = f"; CUDA: {cuda_error}" if cuda_error is not None else ""
            warnings.warn(
                f"Native backends are unavailable (C++: {exc}{cuda_detail}); using Python.",
                RuntimeWarning,
            )
    _backend = backend_python
    if not hasattr(_backend, "name"):
        _backend.name = "python"
    return _backend


def reset_backend() -> None:
    """Reset the selection cache; used by tests and command-line tools."""
    global _backend
    _backend = None
