"""Выбор эталонного Python или ускоренного C++ backend."""

from __future__ import annotations

import os
import warnings
from typing import Any

from . import backend_python


_backend: Any | None = None


def get_backend() -> Any:
    """Лениво выбирает backend согласно MINILLM_BACKEND."""
    global _backend
    if _backend is not None:
        return _backend
    requested = os.environ.get("MINILLM_BACKEND", "auto").lower()
    if requested not in {"auto", "python", "cpp"}:
        raise ValueError("MINILLM_BACKEND должен быть python, cpp или не задан")
    if requested != "python":
        try:
            from .backend_cpp import CppBackend
            _backend = CppBackend()
            return _backend
        except (FileNotFoundError, OSError) as exc:
            if requested == "cpp":
                raise RuntimeError(f"Запрошенный C++ backend недоступен: {exc}") from exc
            warnings.warn(f"C++ backend недоступен ({exc}); используется Python backend.", RuntimeWarning)
    _backend = backend_python
    if not hasattr(_backend, "name"):
        _backend.name = "python"
    return _backend


def reset_backend() -> None:
    """Сбрасывает кэш выбора; используется тестами и CLI."""
    global _backend
    _backend = None

