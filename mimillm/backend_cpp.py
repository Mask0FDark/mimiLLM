"""Безопасная ctypes-обёртка над C ABI библиотеки mimiLLM."""

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
    explicit = os.environ.get("MIMILLM_BACKEND_LIBRARY")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if sys.platform == "win32":
        name = "mimillm_backend.dll"
    elif sys.platform == "darwin":
        name = "libmimillm_backend.dylib"
    else:
        name = "libmimillm_backend.so"
    package_native = Path(__file__).resolve().parent / "_native" / name
    source_build = Path(__file__).resolve().parents[1] / "build" / name
    return package_native if package_native.is_file() else source_build


class CppBackend:
    """Преобразует Python array в указатели только на время синхронного вызова."""

    name = "cpp"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path).expanduser().resolve()
            if path is not None
            else default_library_path()
        )
        if not self.path.is_file():
            raise FileNotFoundError(f"C++ backend не найден: {self.path}")
        self.library = ctypes.CDLL(str(self.path))
        self._configure()
        threads = int(os.environ.get("MIMILLM_NUM_THREADS", "0"))
        if threads > 0:
            self.set_num_threads(threads)

    def _configure(self) -> None:
        library = self.library
        library.mimillm_last_error.restype = ctypes.c_char_p
        library.mimillm_compiler_info.restype = ctypes.c_char_p
        library.mimillm_set_num_threads.argtypes = [ctypes.c_int32]
        library.mimillm_get_num_threads.restype = ctypes.c_int32
        library.mimillm_add_f32.argtypes = [FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64]
        library.mimillm_mul_f32.argtypes = [FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64]
        library.mimillm_scalar_mul_f32.argtypes = [FloatPointer, ctypes.c_float, FloatPointer, ctypes.c_int64]
        library.mimillm_matmul_f32.argtypes = [FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        library.mimillm_batched_matmul_f32.argtypes = [FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        library.mimillm_softmax_rows_f32.argtypes = [FloatPointer, FloatPointer, ctypes.c_int64, ctypes.c_int64]
        library.mimillm_relu_f32.argtypes = [FloatPointer, FloatPointer, ctypes.c_int64]
        library.mimillm_relu_backward_f32.argtypes = [FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64]
        library.mimillm_embedding_gather_f32.argtypes = [FloatPointer, IntPointer, FloatPointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        library.mimillm_embedding_scatter_add_f32.argtypes = [IntPointer, FloatPointer, FloatPointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        library.mimillm_cross_entropy_f32.argtypes = [FloatPointer, IntPointer, FloatPointer, ctypes.c_int64, ctypes.c_int64]
        library.mimillm_cross_entropy_backward_f32.argtypes = [FloatPointer, IntPointer, FloatPointer, ctypes.c_int64, ctypes.c_int64]
        library.mimillm_adamw_f32.argtypes = [
            FloatPointer, FloatPointer, FloatPointer, FloatPointer, ctypes.c_int64,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_int64,
        ]

    @staticmethod
    def _float_buffer(values: Sequence[float]) -> tuple[array, ctypes.Array]:
        storage = values if isinstance(values, array) and values.typecode == "f" else array("f", values)
        pointer = (ctypes.c_float * len(storage)).from_buffer(storage)
        return storage, pointer

    @staticmethod
    def _int_buffer(values: Sequence[int]) -> tuple[array, ctypes.Array]:
        storage = array("i", values)
        if storage.itemsize != 4:
            raise RuntimeError("платформа не предоставляет 32-bit array('i')")
        pointer = (ctypes.c_int32 * len(storage)).from_buffer(storage)
        return storage, pointer

    def _check(self, status: int) -> None:
        if status != 0:
            raw = self.library.mimillm_last_error()
            message = raw.decode("utf-8", errors="replace") if raw else "неизвестная ошибка C++"
            raise RuntimeError(f"C++ backend: {message}")

    @property
    def compiler_info(self) -> str:
        return self.library.mimillm_compiler_info().decode("utf-8", errors="replace")

    @property
    def num_threads(self) -> int:
        return int(self.library.mimillm_get_num_threads())

    def set_num_threads(self, threads: int) -> None:
        self._check(self.library.mimillm_set_num_threads(threads))

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
        return self._binary("mimillm_add_f32", left, right)

    def multiply(self, left: Sequence[float], right: Sequence[float]) -> array:
        return self._binary("mimillm_mul_f32", left, right)

    def scalar_multiply(self, values: Sequence[float], scalar: float) -> array:
        source, source_ptr = self._float_buffer(values)
        output = array("f", [0.0]) * len(values)
        output_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.mimillm_scalar_mul_f32(source_ptr, scalar, output_ptr, len(output)))
        _ = source
        return output

    def matmul(self, left: Sequence[float], right: Sequence[float], rows: int, inner: int, columns: int) -> array:
        if len(left) != rows * inner or len(right) != inner * columns:
            raise ValueError("matmul: размер буфера не соответствует форме")
        a, a_ptr = self._float_buffer(left)
        b, b_ptr = self._float_buffer(right)
        output = array("f", [0.0]) * (rows * columns)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.mimillm_matmul_f32(a_ptr, b_ptr, out_ptr, rows, inner, columns))
        _ = a, b
        return output

    def batched_matmul(self, left: Sequence[float], right: Sequence[float], batches: int, rows: int, inner: int, columns: int) -> array:
        if len(left) != batches * rows * inner or len(right) != batches * inner * columns:
            raise ValueError("batched_matmul: размер буфера не соответствует форме")
        a, a_ptr = self._float_buffer(left)
        b, b_ptr = self._float_buffer(right)
        output = array("f", [0.0]) * (batches * rows * columns)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.mimillm_batched_matmul_f32(a_ptr, b_ptr, out_ptr, batches, rows, inner, columns))
        _ = a, b
        return output

    def softmax_rows(self, values: Sequence[float], rows: int, columns: int) -> array:
        if len(values) != rows * columns:
            raise ValueError("softmax: размер буфера не соответствует форме")
        source, source_ptr = self._float_buffer(values)
        output = array("f", [0.0]) * len(values)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.mimillm_softmax_rows_f32(source_ptr, out_ptr, rows, columns))
        _ = source
        return output

    def relu(self, values: Sequence[float]) -> array:
        source, source_ptr = self._float_buffer(values)
        output = array("f", [0.0]) * len(values)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.mimillm_relu_f32(source_ptr, out_ptr, len(values)))
        _ = source
        return output

    def relu_backward(self, values: Sequence[float], grad_output: Sequence[float]) -> array:
        if len(values) != len(grad_output):
            raise ValueError("relu_backward: длины буферов не совпадают")
        source, source_ptr = self._float_buffer(values)
        gradient, gradient_ptr = self._float_buffer(grad_output)
        output = array("f", [0.0]) * len(values)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.mimillm_relu_backward_f32(source_ptr, gradient_ptr, out_ptr, len(output)))
        _ = source, gradient
        return output

    def embedding_gather(self, table: Sequence[float], indices: Sequence[int], vocab: int, width: int) -> array:
        if len(table) != vocab * width:
            raise ValueError("embedding_gather: размер таблицы не соответствует форме")
        values, values_ptr = self._float_buffer(table)
        ids, ids_ptr = self._int_buffer(indices)
        output = array("f", [0.0]) * (len(indices) * width)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.mimillm_embedding_gather_f32(values_ptr, ids_ptr, out_ptr, vocab, width, len(indices)))
        _ = values, ids
        return output

    def embedding_scatter_add(self, indices: Sequence[int], grad_output: Sequence[float], vocab: int, width: int) -> array:
        if len(grad_output) != len(indices) * width:
            raise ValueError("embedding_scatter_add: неверная форма градиента")
        ids, ids_ptr = self._int_buffer(indices)
        gradient, gradient_ptr = self._float_buffer(grad_output)
        output = array("f", [0.0]) * (vocab * width)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.mimillm_embedding_scatter_add_f32(ids_ptr, gradient_ptr, out_ptr, vocab, width, len(indices)))
        _ = ids, gradient
        return output

    def cross_entropy(self, logits: Sequence[float], targets: Sequence[int], rows: int, classes: int) -> float:
        if len(logits) != rows * classes or len(targets) != rows:
            raise ValueError("cross_entropy: размеры буферов не соответствуют форме")
        values, values_ptr = self._float_buffer(logits)
        ids, ids_ptr = self._int_buffer(targets)
        loss = ctypes.c_float()
        self._check(self.library.mimillm_cross_entropy_f32(values_ptr, ids_ptr, ctypes.byref(loss), rows, classes))
        _ = values, ids
        return float(loss.value)

    def cross_entropy_backward(self, logits: Sequence[float], targets: Sequence[int], rows: int, classes: int) -> array:
        if len(logits) != rows * classes or len(targets) != rows:
            raise ValueError("cross_entropy_backward: размеры не соответствуют форме")
        values, values_ptr = self._float_buffer(logits)
        ids, ids_ptr = self._int_buffer(targets)
        output = array("f", [0.0]) * len(logits)
        out_ptr = (ctypes.c_float * len(output)).from_buffer(output)
        self._check(self.library.mimillm_cross_entropy_backward_f32(values_ptr, ids_ptr, out_ptr, rows, classes))
        _ = values, ids
        return output

    def adamw_update(
        self, parameter: array, gradient: Sequence[float], first_moment: array,
        second_moment: array, *, learning_rate: float, beta1: float, beta2: float,
        epsilon: float, weight_decay: float, step: int,
    ) -> None:
        """Обновляет переданные array('f') непосредственно через C++ kernel."""
        if not all(isinstance(values, array) and values.typecode == "f" for values in (parameter, first_moment, second_moment)):
            raise TypeError("parameter и moments должны быть array('f')")
        if not (len(parameter) == len(gradient) == len(first_moment) == len(second_moment)):
            raise ValueError("AdamW: длины буферов не совпадают")
        parameter_store, parameter_ptr = self._float_buffer(parameter)
        gradient_store, gradient_ptr = self._float_buffer(gradient)
        first_store, first_ptr = self._float_buffer(first_moment)
        second_store, second_ptr = self._float_buffer(second_moment)
        self._check(self.library.mimillm_adamw_f32(
            parameter_ptr, gradient_ptr, first_ptr, second_ptr, len(parameter),
            learning_rate, beta1, beta2, epsilon, weight_decay, step,
        ))
        _ = parameter_store, gradient_store, first_store, second_store


def is_available() -> bool:
    return default_library_path().is_file()
