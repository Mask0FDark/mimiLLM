"""Безопасная ctypes-обёртка над C ABI библиотеки m0fdii."""

from __future__ import annotations

import ctypes
import os
import sys
from array import array
from collections.abc import Sequence
from pathlib import Path


FloatPointer = ctypes.POINTER(ctypes.c_float)
IntPointer = ctypes.POINTER(ctypes.c_int32)


def default_library_path() -> Path:
    root = Path(__file__).resolve().parents[1] / "build"
    if sys.platform == "win32":
        return root / "minillm_backend.dll"
    if sys.platform == "darwin":
        return root / "libminillm_backend.dylib"
    return root / "libminillm_backend.so"


class CppBackend:
    """Преобразует Python array в указатели только на время синхронного вызова."""

    name = "cpp"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_library_path()
        if not self.path.is_file():
            raise FileNotFoundError(f"C++ backend не найден: {self.path}")
        self.library = ctypes.CDLL(str(self.path))
        self._configure()
        threads = int(os.environ.get("MINILLM_NUM_THREADS", "0"))
        if threads > 0:
            self.set_num_threads(threads)

    def _configure(self) -> None:
        library = self.library
        library.minillm_last_error.restype = ctypes.c_char_p
        library.minillm_compiler_info.restype = ctypes.c_char_p
        library.minillm_set_num_threads.argtypes = [ctypes.c_int32]
        library.minillm_get_num_threads.restype = ctypes.c_int32
        library.minillm_add_f32.argtypes = [FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64]
        library.minillm_mul_f32.argtypes = [FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64]
        library.minillm_scalar_mul_f32.argtypes = [FloatPointer, ctypes.c_float, FloatPointer, ctypes.c_int64]
        library.minillm_matmul_f32.argtypes = [FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        library.minillm_batched_matmul_f32.argtypes = [FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        library.minillm_softmax_rows_f32.argtypes = [FloatPointer, FloatPointer, ctypes.c_int64, ctypes.c_int64]
        library.minillm_relu_f32.argtypes = [FloatPointer, FloatPointer, ctypes.c_int64]
        library.minillm_relu_backward_f32.argtypes = [FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64]

    @staticmethod
    def _float_buffer(values: Sequence[float]) -> tuple[array, ctypes.Array]:
        storage = values if isinstance(values, array) and values.typecode == "f" else array("f", values)
        pointer = (ctypes.c_float * len(storage)).from_buffer(storage)
        return storage, pointer

    def _check(self, status: int) -> None:
        if status != 0:
            raw = self.library.minillm_last_error()
            message = raw.decode("utf-8", errors="replace") if raw else "неизвестная ошибка C++"
            raise RuntimeError(f"C++ backend: {message}")

    @property
    def compiler_info(self) -> str:
        return self.library.minillm_compiler_info().decode("utf-8", errors="replace")

    @property
    def num_threads(self) -> int:
        return int(self.library.minillm_get_num_threads())

    def set_num_threads(self, threads: int) -> None:
        self._check(self.library.minillm_set_num_threads(threads))

    def _binary(self, function_name: str, left: Sequence[float], right: Sequence[float]) -> array:
        if len(left) != len(right):
            raise ValueError(f"{function_name}: длины буферов не совпадают")
        left_store, left_ptr = self._float_buffer(left)
        right_store, right_ptr = self._float_buffer(right)
        output = array("f", [0.0]) * len(left)
        output_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        function = getattr(self.library, function_name)
        self._check(function(left_ptr, right_ptr, output_ptr, len(output)))
        _ = left_store, right_store
        return output

    def add(self, left: Sequence[float], right: Sequence[float]) -> array:
        return self._binary("minillm_add_f32", left, right)

    def multiply(self, left: Sequence[float], right: Sequence[float]) -> array:
        return self._binary("minillm_mul_f32", left, right)

    def scalar_multiply(self, values: Sequence[float], scalar: float) -> array:
        source, source_ptr = self._float_buffer(values)
        output = array("f", [0.0]) * len(values)
        output_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.minillm_scalar_mul_f32(source_ptr, scalar, output_ptr, len(output)))
        _ = source
        return output

    def matmul(self, left: Sequence[float], right: Sequence[float], rows: int, inner: int, columns: int) -> array:
        if len(left) != rows * inner or len(right) != inner * columns:
            raise ValueError("matmul: размер буфера не соответствует форме")
        a, a_ptr = self._float_buffer(left)
        b, b_ptr = self._float_buffer(right)
        output = array("f", [0.0]) * (rows * columns)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.minillm_matmul_f32(a_ptr, b_ptr, out_ptr, rows, inner, columns))
        _ = a, b
        return output

    def batched_matmul(self, left: Sequence[float], right: Sequence[float], batches: int, rows: int, inner: int, columns: int) -> array:
        if len(left) != batches * rows * inner or len(right) != batches * inner * columns:
            raise ValueError("batched_matmul: размер буфера не соответствует форме")
        a, a_ptr = self._float_buffer(left)
        b, b_ptr = self._float_buffer(right)
        output = array("f", [0.0]) * (batches * rows * columns)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.minillm_batched_matmul_f32(a_ptr, b_ptr, out_ptr, batches, rows, inner, columns))
        _ = a, b
        return output

    def softmax_rows(self, values: Sequence[float], rows: int, columns: int) -> array:
        if len(values) != rows * columns:
            raise ValueError("softmax: размер буфера не соответствует форме")
        source, source_ptr = self._float_buffer(values)
        output = array("f", [0.0]) * len(values)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.minillm_softmax_rows_f32(source_ptr, out_ptr, rows, columns))
        _ = source
        return output

    def relu(self, values: Sequence[float]) -> array:
        source, source_ptr = self._float_buffer(values)
        output = array("f", [0.0]) * len(values)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.minillm_relu_f32(source_ptr, out_ptr, len(values)))
        _ = source
        return output


def is_available() -> bool:
    return default_library_path().is_file()

