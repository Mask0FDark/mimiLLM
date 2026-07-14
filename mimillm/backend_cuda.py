"""NVIDIA CUDA backend loaded directly through the Driver API and NVRTC."""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from array import array
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


CUDA_SUCCESS = 0
BLOCK_SIZE = 256
TILE_SIZE = 16
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


def kernel_source_path() -> Path:
    """Return the development or installed-package CUDA kernel source."""
    source_tree = Path(__file__).resolve().parents[1] / "cuda" / "kernels.cu"
    packaged = Path(__file__).resolve().parent / "_native" / "mimillm_cuda_kernels.cu"
    return source_tree if source_tree.is_file() else packaged


def _product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _contiguous_strides(shape: Sequence[int]) -> list[int]:
    result = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        result[axis] = result[axis + 1] * int(shape[axis + 1])
    return result


def _broadcast_strides(shape: Sequence[int], output_shape: Sequence[int]) -> list[int]:
    if len(shape) > len(output_shape):
        raise ValueError("operand has more dimensions than the broadcast output")
    source = _contiguous_strides(shape)
    result = [0] * len(output_shape)
    offset = len(output_shape) - len(shape)
    for axis, dimension in enumerate(shape):
        output_dimension = output_shape[offset + axis]
        if dimension not in (1, output_dimension):
            raise ValueError("incompatible broadcast shapes")
        if dimension != 1:
            result[offset + axis] = source[axis]
    return result


def _float_storage(values: Sequence[float]) -> array:
    return values if isinstance(values, array) and values.typecode == "f" else array("f", values)


def _int32_storage(values: Sequence[int]) -> array:
    storage = array("i", values)
    if storage.itemsize != 4:
        raise RuntimeError("the platform does not provide a 32-bit array('i')")
    return storage


def _int64_storage(values: Sequence[int]) -> array:
    storage = array("q", values)
    if storage.itemsize != 8:
        raise RuntimeError("the platform does not provide a 64-bit array('q')")
    return storage


class _Nvrtc:
    """Small NVRTC wrapper used once when the CUDA backend is initialized."""

    def __init__(self) -> None:
        self._dll_directories: list[Any] = []
        self.include_directories: list[Path] = []
        self.library = self._load_library()
        self._configure()

    def _load_library(self) -> ctypes.CDLL:
        candidates: list[Path] = []
        roots: list[Path] = []
        for variable in ("CUDA_PATH", "CUDA_HOME"):
            if root := os.environ.get(variable):
                roots.append(Path(root))
        if nvcc := shutil.which("nvcc.exe" if sys.platform == "win32" else "nvcc"):
            roots.append(Path(nvcc).resolve().parent.parent)
        roots.append(Path(sys.prefix) / "Library" if sys.platform == "win32" else Path(sys.prefix))
        for root_path in roots:
            bin_dir = root_path / "bin"
            include_dir = root_path / "include"
            if sys.platform == "win32":
                if include_dir.is_dir() and include_dir not in self.include_directories:
                    self.include_directories.append(include_dir)
                if bin_dir.is_dir():
                    self._dll_directories.append(os.add_dll_directory(str(bin_dir)))
                candidates.extend(sorted(bin_dir.glob("nvrtc64_*.dll"), reverse=True))
            else:
                if include_dir.is_dir() and include_dir not in self.include_directories:
                    self.include_directories.append(include_dir)
                candidates.extend(sorted((root_path / "lib64").glob("libnvrtc.so*"), reverse=True))
                candidates.extend(sorted((root_path / "lib").glob("libnvrtc.so*"), reverse=True))
        if sys.platform != "win32":
            for candidate in candidates:
                try:
                    return ctypes.CDLL(str(candidate))
                except OSError:
                    pass
            for name in ("libnvrtc.so", "libnvrtc.so.12"):
                try:
                    return ctypes.CDLL(name)
                except OSError:
                    pass
        for candidate in candidates:
            try:
                return ctypes.CDLL(str(candidate))
            except OSError:
                pass
        raise FileNotFoundError(
            "NVRTC was not found. Install the NVIDIA CUDA Toolkit and set CUDA_PATH."
        )

    def _configure(self) -> None:
        library = self.library
        library.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p),
        ]
        library.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
        ]
        library.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        library.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        library.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        library.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        library.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    def compile(self, source: str, architecture: str) -> bytes:
        program = ctypes.c_void_p()
        encoded = source.encode("utf-8")
        status = self.library.nvrtcCreateProgram(
            ctypes.byref(program), encoded, b"mimillm_cuda_kernels.cu", 0, None, None,
        )
        if status != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed with status {status}")
        options = [b"--std=c++17", f"--gpu-architecture={architecture}".encode("ascii")]
        options.extend(
            f"--include-path={directory}".encode("utf-8")
            for directory in self.include_directories
        )
        option_array = (ctypes.c_char_p * len(options))(*options)
        try:
            status = self.library.nvrtcCompileProgram(program, len(options), option_array)
            log_size = ctypes.c_size_t()
            self.library.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
            log = ctypes.create_string_buffer(max(1, log_size.value))
            self.library.nvrtcGetProgramLog(program, log)
            if status != 0:
                message = log.value.decode("utf-8", errors="replace")
                raise RuntimeError(f"NVRTC compilation failed:\n{message}")
            ptx_size = ctypes.c_size_t()
            if self.library.nvrtcGetPTXSize(program, ctypes.byref(ptx_size)) != 0:
                raise RuntimeError("nvrtcGetPTXSize failed")
            ptx = ctypes.create_string_buffer(ptx_size.value)
            if self.library.nvrtcGetPTX(program, ptx) != 0:
                raise RuntimeError("nvrtcGetPTX failed")
            return bytes(ptx.raw)
        finally:
            self.library.nvrtcDestroyProgram(ctypes.byref(program))


class _CudaRuntime:
    """Own a CUDA context, compiled module, kernel handles, and device buffers."""

    def __init__(self) -> None:
        driver_name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
        loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
        try:
            self.driver = loader(driver_name)
        except OSError as exc:
            raise FileNotFoundError("the NVIDIA CUDA driver is not installed") from exc
        self._configure_driver()
        self._check(self.driver.cuInit(0), "cuInit")
        self.device = ctypes.c_int()
        self._check(self.driver.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")
        self.context = ctypes.c_void_p()
        self._check(
            self.driver.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device),
            "cuCtxCreate",
        )
        major = self.device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        minor = self.device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        source_path = kernel_source_path()
        if not source_path.is_file():
            raise FileNotFoundError(f"CUDA kernel source was not found: {source_path}")
        ptx = _Nvrtc().compile(source_path.read_text(encoding="utf-8"), f"compute_{major}{minor}")
        self.module = ctypes.c_void_p()
        ptx_buffer = ctypes.create_string_buffer(ptx)
        self._check(
            self.driver.cuModuleLoadDataEx(
                ctypes.byref(self.module), ctypes.cast(ptx_buffer, ctypes.c_void_p),
                0, None, None,
            ),
            "cuModuleLoadDataEx",
        )
        self._functions: dict[str, ctypes.c_void_p] = {}
        self._pool: dict[int, list[int]] = defaultdict(list)

    def _configure_driver(self) -> None:
        driver = self.driver
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        driver.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        driver.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        driver.cuModuleLoadDataEx.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        driver.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p,
        ]
        driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        driver.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        driver.cuMemsetD8_v2.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]
        driver.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        driver.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        driver.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ]
        driver.cuCtxSynchronize.argtypes = []
        driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def _check(self, status: int, operation: str) -> None:
        if status == CUDA_SUCCESS:
            return
        message = ctypes.c_char_p()
        self.driver.cuGetErrorString(status, ctypes.byref(message))
        detail = message.value.decode("utf-8", errors="replace") if message.value else f"status {status}"
        raise RuntimeError(f"{operation}: {detail}")

    def device_attribute(self, attribute: int) -> int:
        result = ctypes.c_int()
        self._check(
            self.driver.cuDeviceGetAttribute(ctypes.byref(result), attribute, self.device),
            "cuDeviceGetAttribute",
        )
        return int(result.value)

    @property
    def device_name(self) -> str:
        output = ctypes.create_string_buffer(256)
        self._check(self.driver.cuDeviceGetName(output, len(output), self.device), "cuDeviceGetName")
        return output.value.decode("utf-8", errors="replace")

    @property
    def device_memory(self) -> int:
        output = ctypes.c_size_t()
        self._check(
            self.driver.cuDeviceTotalMem_v2(ctypes.byref(output), self.device),
            "cuDeviceTotalMem",
        )
        return int(output.value)

    def function(self, name: str) -> ctypes.c_void_p:
        if name not in self._functions:
            result = ctypes.c_void_p()
            self._check(
                self.driver.cuModuleGetFunction(
                    ctypes.byref(result), self.module, name.encode("ascii"),
                ),
                f"cuModuleGetFunction({name})",
            )
            self._functions[name] = result
        return self._functions[name]

    def allocate(self, size: int) -> int:
        size = max(1, int(size))
        if self._pool[size]:
            return self._pool[size].pop()
        pointer = ctypes.c_uint64()
        self._check(self.driver.cuMemAlloc_v2(ctypes.byref(pointer), size), "cuMemAlloc")
        return int(pointer.value)

    def release(self, pointer: int, size: int) -> None:
        self._pool[max(1, int(size))].append(pointer)

    @contextmanager
    def buffers(self, *sizes: int) -> Iterator[tuple[int, ...]]:
        pointers: list[int] = []
        try:
            pointers = [self.allocate(size) for size in sizes]
            yield tuple(pointers)
        finally:
            for pointer, size in zip(pointers, sizes):
                self.release(pointer, size)

    def upload(self, pointer: int, storage: array) -> None:
        size = len(storage) * storage.itemsize
        if not size:
            return
        view = (ctypes.c_ubyte * size).from_buffer(storage)
        self._check(
            self.driver.cuMemcpyHtoD_v2(pointer, ctypes.cast(view, ctypes.c_void_p), size),
            "cuMemcpyHtoD",
        )

    def download(self, storage: array, pointer: int) -> None:
        size = len(storage) * storage.itemsize
        if not size:
            return
        view = (ctypes.c_ubyte * size).from_buffer(storage)
        self._check(
            self.driver.cuMemcpyDtoH_v2(ctypes.cast(view, ctypes.c_void_p), pointer, size),
            "cuMemcpyDtoH",
        )

    def zero(self, pointer: int, size: int) -> None:
        self._check(self.driver.cuMemsetD8_v2(pointer, 0, size), "cuMemsetD8")

    def launch(
        self, name: str, grid: tuple[int, int, int], block: tuple[int, int, int],
        arguments: Sequence[ctypes._SimpleCData], *, shared_memory: int = 0,
    ) -> None:
        pointers = (ctypes.c_void_p * len(arguments))(*(
            ctypes.cast(ctypes.byref(argument), ctypes.c_void_p) for argument in arguments
        ))
        self._check(
            self.driver.cuLaunchKernel(
                self.function(name), *grid, *block, shared_memory, None, pointers, None,
            ),
            f"cuLaunchKernel({name})",
        )
        self._check(self.driver.cuCtxSynchronize(), "cuCtxSynchronize")


_runtime_instance: _CudaRuntime | None = None


def _runtime() -> _CudaRuntime:
    global _runtime_instance
    if _runtime_instance is None:
        _runtime_instance = _CudaRuntime()
    return _runtime_instance


class CudaBackend:
    """Run mimiLLM tensor kernels on an NVIDIA GPU."""

    name = "cuda"
    supports_native_broadcast = True

    def __init__(self) -> None:
        self.runtime = _runtime()

    @property
    def device_name(self) -> str:
        return self.runtime.device_name

    @property
    def device_memory(self) -> int:
        return self.runtime.device_memory

    @property
    def multiprocessors(self) -> int:
        return self.runtime.device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

    @property
    def compiler_info(self) -> str:
        return f"NVRTC CUDA | {self.device_name}"

    @staticmethod
    def _pointer(value: int) -> ctypes.c_uint64:
        return ctypes.c_uint64(value)

    @staticmethod
    def _count(value: int) -> ctypes.c_int64:
        return ctypes.c_int64(value)

    def _elementwise_binary(
        self, kernel: str, left: Sequence[float], right: Sequence[float],
    ) -> array:
        if len(left) != len(right):
            raise ValueError(f"{kernel}: buffer lengths do not match")
        left_values, right_values = _float_storage(left), _float_storage(right)
        output = array("f", [0.0]) * len(left_values)
        size = len(output) * output.itemsize
        with self.runtime.buffers(size, size, size) as (a, b, result):
            self.runtime.upload(a, left_values)
            self.runtime.upload(b, right_values)
            self.runtime.launch(
                kernel, ((len(output) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1),
                (BLOCK_SIZE, 1, 1),
                [self._pointer(a), self._pointer(b), self._pointer(result), self._count(len(output))],
            )
            self.runtime.download(output, result)
        return output

    def add(self, left: Sequence[float], right: Sequence[float]) -> array:
        return self._elementwise_binary("mimillm_add", left, right)

    def multiply(self, left: Sequence[float], right: Sequence[float]) -> array:
        return self._elementwise_binary("mimillm_multiply", left, right)

    def scalar_multiply(self, values: Sequence[float], scalar: float) -> array:
        source = _float_storage(values)
        output = array("f", [0.0]) * len(source)
        size = len(source) * source.itemsize
        with self.runtime.buffers(size, size) as (input_pointer, output_pointer):
            self.runtime.upload(input_pointer, source)
            self.runtime.launch(
                "mimillm_scalar_multiply",
                ((len(source) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(input_pointer), ctypes.c_float(scalar), self._pointer(output_pointer), self._count(len(source))],
            )
            self.runtime.download(output, output_pointer)
        return output

    def permute(
        self, values: Sequence[float], shape: Sequence[int], axes: Sequence[int],
    ) -> array:
        if len(shape) != len(axes) or sorted(axes) != list(range(len(shape))):
            raise ValueError("permute axes must be a permutation of all dimensions")
        count = _product(shape)
        if len(values) != count:
            raise ValueError("permute buffer size does not match its shape")
        source = _float_storage(values)
        source_strides = _int64_storage(_contiguous_strides(shape))
        output_shape = [shape[axis] for axis in axes]
        output_shape_values = _int64_storage(output_shape)
        output_strides = _int64_storage(_contiguous_strides(output_shape))
        axes_values = _int64_storage(axes)
        output = array("f", [0.0]) * count
        metadata_size = len(shape) * 8
        with self.runtime.buffers(
            count * 4, count * 4, metadata_size, metadata_size, metadata_size, metadata_size,
        ) as buffers:
            source_pointer, output_pointer, source_stride_pointer, shape_pointer, output_stride_pointer, axes_pointer = buffers
            for pointer, storage in (
                (source_pointer, source), (source_stride_pointer, source_strides),
                (shape_pointer, output_shape_values), (output_stride_pointer, output_strides),
                (axes_pointer, axes_values),
            ):
                self.runtime.upload(pointer, storage)
            self.runtime.launch(
                "mimillm_permute", ((count + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1),
                (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(source_pointer), self._pointer(output_pointer),
                    self._pointer(source_stride_pointer), self._pointer(shape_pointer),
                    self._pointer(output_stride_pointer), self._pointer(axes_pointer),
                    self._count(len(shape)), self._count(count),
                ],
            )
            self.runtime.download(output, output_pointer)
        return output

    @staticmethod
    def _operation_code(operation: str) -> int:
        try:
            return {"add": 0, "sub": 1, "mul": 2, "div": 3}[operation]
        except KeyError as exc:
            raise ValueError(f"unknown binary operation: {operation}") from exc

    def _broadcast_metadata(
        self, left_shape: Sequence[int], right_shape: Sequence[int],
        output_shape: Sequence[int],
    ) -> tuple[array, array, array, array]:
        return (
            _int64_storage(_broadcast_strides(left_shape, output_shape)),
            _int64_storage(_broadcast_strides(right_shape, output_shape)),
            _int64_storage(output_shape),
            _int64_storage(_contiguous_strides(output_shape)),
        )

    def broadcast_binary(
        self, left: Sequence[float], right: Sequence[float],
        left_shape: Sequence[int], right_shape: Sequence[int],
        output_shape: Sequence[int], operation: str,
    ) -> array:
        left_values, right_values = _float_storage(left), _float_storage(right)
        metadata = self._broadcast_metadata(left_shape, right_shape, output_shape)
        output = array("f", [0.0]) * _product(output_shape)
        metadata_size = len(output_shape) * 8
        sizes = [len(left_values) * 4, len(right_values) * 4, len(output) * 4, *([metadata_size] * 4)]
        with self.runtime.buffers(*sizes) as buffers:
            left_pointer, right_pointer, output_pointer, *metadata_pointers = buffers
            self.runtime.upload(left_pointer, left_values)
            self.runtime.upload(right_pointer, right_values)
            for pointer, storage in zip(metadata_pointers, metadata):
                self.runtime.upload(pointer, storage)
            self.runtime.launch(
                "mimillm_broadcast_binary",
                ((len(output) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(left_pointer), self._pointer(right_pointer), self._pointer(output_pointer),
                    *(self._pointer(pointer) for pointer in metadata_pointers),
                    self._count(len(output_shape)), self._count(len(output)),
                    ctypes.c_int32(self._operation_code(operation)),
                ],
            )
            self.runtime.download(output, output_pointer)
        return output

    def broadcast_binary_backward(
        self, left: Sequence[float], right: Sequence[float], grad_output: Sequence[float],
        left_shape: Sequence[int], right_shape: Sequence[int],
        output_shape: Sequence[int], operation: str,
    ) -> tuple[array, array]:
        left_values, right_values, gradient = map(_float_storage, (left, right, grad_output))
        metadata = self._broadcast_metadata(left_shape, right_shape, output_shape)
        grad_left = array("f", [0.0]) * len(left_values)
        grad_right = array("f", [0.0]) * len(right_values)
        metadata_size = len(output_shape) * 8
        sizes = [
            len(left_values) * 4, len(right_values) * 4, len(gradient) * 4,
            len(grad_left) * 4, len(grad_right) * 4, *([metadata_size] * 4),
        ]
        with self.runtime.buffers(*sizes) as buffers:
            left_pointer, right_pointer, gradient_pointer, grad_left_pointer, grad_right_pointer, *metadata_pointers = buffers
            for pointer, storage in ((left_pointer, left_values), (right_pointer, right_values), (gradient_pointer, gradient)):
                self.runtime.upload(pointer, storage)
            self.runtime.zero(grad_left_pointer, len(grad_left) * 4)
            self.runtime.zero(grad_right_pointer, len(grad_right) * 4)
            for pointer, storage in zip(metadata_pointers, metadata):
                self.runtime.upload(pointer, storage)
            self.runtime.launch(
                "mimillm_broadcast_backward",
                ((len(gradient) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(left_pointer), self._pointer(right_pointer), self._pointer(gradient_pointer),
                    self._pointer(grad_left_pointer), self._pointer(grad_right_pointer),
                    *(self._pointer(pointer) for pointer in metadata_pointers),
                    self._count(len(output_shape)), self._count(len(gradient)),
                    ctypes.c_int32(self._operation_code(operation)),
                ],
            )
            self.runtime.download(grad_left, grad_left_pointer)
            self.runtime.download(grad_right, grad_right_pointer)
        return grad_left, grad_right

    def matmul(
        self, left: Sequence[float], right: Sequence[float],
        rows: int, inner: int, columns: int,
    ) -> array:
        return self.batched_matmul(left, right, 1, rows, inner, columns)

    def batched_matmul(
        self, left: Sequence[float], right: Sequence[float], batches: int,
        rows: int, inner: int, columns: int,
    ) -> array:
        if len(left) != batches * rows * inner or len(right) != batches * inner * columns:
            raise ValueError("matmul buffer size does not match its shape")
        left_values, right_values = _float_storage(left), _float_storage(right)
        output = array("f", [0.0]) * (batches * rows * columns)
        with self.runtime.buffers(len(left_values) * 4, len(right_values) * 4, len(output) * 4) as pointers:
            left_pointer, right_pointer, output_pointer = pointers
            self.runtime.upload(left_pointer, left_values)
            self.runtime.upload(right_pointer, right_values)
            self.runtime.launch(
                "mimillm_matmul",
                ((columns + TILE_SIZE - 1) // TILE_SIZE, (rows + TILE_SIZE - 1) // TILE_SIZE, batches),
                (TILE_SIZE, TILE_SIZE, 1),
                [
                    self._pointer(left_pointer), self._pointer(right_pointer), self._pointer(output_pointer),
                    self._count(batches), self._count(rows), self._count(inner), self._count(columns),
                ],
            )
            self.runtime.download(output, output_pointer)
        return output

    def softmax_rows(self, values: Sequence[float], rows: int, columns: int) -> array:
        return self._rows_operation("mimillm_softmax_rows", values, rows, columns)

    def _rows_operation(self, kernel: str, values: Sequence[float], rows: int, columns: int) -> array:
        source = _float_storage(values)
        if len(source) != rows * columns:
            raise ValueError(f"{kernel}: buffer size does not match its shape")
        output = array("f", [0.0]) * len(source)
        size = len(source) * 4
        with self.runtime.buffers(size, size) as (source_pointer, output_pointer):
            self.runtime.upload(source_pointer, source)
            self.runtime.launch(
                kernel, (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(source_pointer), self._pointer(output_pointer), self._count(rows), self._count(columns)],
                shared_memory=BLOCK_SIZE * 4,
            )
            self.runtime.download(output, output_pointer)
        return output

    def softmax_backward(
        self, output_values: Sequence[float], grad_output: Sequence[float],
        rows: int, columns: int,
    ) -> array:
        values, gradient = _float_storage(output_values), _float_storage(grad_output)
        if len(values) != rows * columns or len(gradient) != len(values):
            raise ValueError("softmax backward buffers do not match their shape")
        output = array("f", [0.0]) * len(values)
        size = len(values) * 4
        with self.runtime.buffers(size, size, size) as pointers:
            values_pointer, gradient_pointer, output_pointer = pointers
            self.runtime.upload(values_pointer, values)
            self.runtime.upload(gradient_pointer, gradient)
            self.runtime.launch(
                "mimillm_softmax_backward", (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(values_pointer), self._pointer(gradient_pointer), self._pointer(output_pointer), self._count(rows), self._count(columns)],
                shared_memory=BLOCK_SIZE * 4,
            )
            self.runtime.download(output, output_pointer)
        return output

    def sum_rows(self, values: Sequence[float], rows: int, columns: int) -> array:
        source = _float_storage(values)
        output = array("f", [0.0]) * rows
        with self.runtime.buffers(len(source) * 4, rows * 4) as (source_pointer, output_pointer):
            self.runtime.upload(source_pointer, source)
            self.runtime.launch(
                "mimillm_sum_rows", (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(source_pointer), self._pointer(output_pointer), self._count(rows), self._count(columns)],
                shared_memory=BLOCK_SIZE * 4,
            )
            self.runtime.download(output, output_pointer)
        return output

    def sum_rows_backward(self, grad_output: Sequence[float], rows: int, columns: int) -> array:
        gradient = _float_storage(grad_output)
        output = array("f", [0.0]) * (rows * columns)
        with self.runtime.buffers(len(gradient) * 4, len(output) * 4) as (gradient_pointer, output_pointer):
            self.runtime.upload(gradient_pointer, gradient)
            self.runtime.launch(
                "mimillm_sum_rows_backward", ((len(output) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1),
                (BLOCK_SIZE, 1, 1),
                [self._pointer(gradient_pointer), self._pointer(output_pointer), self._count(len(output)), self._count(columns)],
            )
            self.runtime.download(output, output_pointer)
        return output

    def relu(self, values: Sequence[float]) -> array:
        return self._unary("mimillm_relu", values)

    def _unary(self, kernel: str, values: Sequence[float]) -> array:
        source = _float_storage(values)
        output = array("f", [0.0]) * len(source)
        size = len(source) * 4
        with self.runtime.buffers(size, size) as (source_pointer, output_pointer):
            self.runtime.upload(source_pointer, source)
            self.runtime.launch(
                kernel, ((len(source) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(source_pointer), self._pointer(output_pointer), self._count(len(source))],
            )
            self.runtime.download(output, output_pointer)
        return output

    def relu_backward(self, values: Sequence[float], grad_output: Sequence[float]) -> array:
        source, gradient = _float_storage(values), _float_storage(grad_output)
        output = array("f", [0.0]) * len(source)
        size = len(source) * 4
        with self.runtime.buffers(size, size, size) as pointers:
            source_pointer, gradient_pointer, output_pointer = pointers
            self.runtime.upload(source_pointer, source)
            self.runtime.upload(gradient_pointer, gradient)
            self.runtime.launch(
                "mimillm_relu_backward", ((len(source) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(source_pointer), self._pointer(gradient_pointer), self._pointer(output_pointer), self._count(len(source))],
            )
            self.runtime.download(output, output_pointer)
        return output

    def embedding_gather(
        self, table: Sequence[float], indices: Sequence[int], vocab: int, width: int,
    ) -> array:
        values, ids = _float_storage(table), _int32_storage(indices)
        output = array("f", [0.0]) * (len(ids) * width)
        with self.runtime.buffers(len(values) * 4, len(ids) * 4, len(output) * 4) as pointers:
            table_pointer, ids_pointer, output_pointer = pointers
            self.runtime.upload(table_pointer, values)
            self.runtime.upload(ids_pointer, ids)
            self.runtime.launch(
                "mimillm_embedding_gather", ((len(output) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(table_pointer), self._pointer(ids_pointer), self._pointer(output_pointer), self._count(width), self._count(len(output))],
            )
            self.runtime.download(output, output_pointer)
        return output

    def embedding_scatter_add(
        self, indices: Sequence[int], grad_output: Sequence[float], vocab: int, width: int,
    ) -> array:
        ids, gradient = _int32_storage(indices), _float_storage(grad_output)
        output = array("f", [0.0]) * (vocab * width)
        with self.runtime.buffers(len(ids) * 4, len(gradient) * 4, len(output) * 4) as pointers:
            ids_pointer, gradient_pointer, output_pointer = pointers
            self.runtime.upload(ids_pointer, ids)
            self.runtime.upload(gradient_pointer, gradient)
            self.runtime.zero(output_pointer, len(output) * 4)
            self.runtime.launch(
                "mimillm_embedding_scatter", ((len(gradient) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(ids_pointer), self._pointer(gradient_pointer), self._pointer(output_pointer), self._count(width), self._count(len(gradient))],
            )
            self.runtime.download(output, output_pointer)
        return output

    def _cross_entropy(
        self, logits: Sequence[float], targets: Sequence[int], rows: int, classes: int,
        weights: Sequence[float] | None, *, gradient: bool,
    ) -> tuple[float, array | None]:
        values, ids = _float_storage(logits), _int32_storage(targets)
        weight_values = _float_storage(weights) if weights is not None else None
        weight_sum = float(sum(weight_values)) if weight_values is not None else float(rows)
        output_gradient = array("f", [0.0]) * len(values) if gradient else None
        sizes = [len(values) * 4, len(ids) * 4, 4]
        if weight_values is not None:
            sizes.append(len(weight_values) * 4)
        if gradient:
            sizes.append(len(values) * 4)
        with self.runtime.buffers(*sizes) as pointers:
            logits_pointer, ids_pointer, loss_pointer, *remaining = pointers
            weights_pointer = remaining.pop(0) if weight_values is not None else 0
            gradient_pointer = remaining.pop(0) if gradient else 0
            self.runtime.upload(logits_pointer, values)
            self.runtime.upload(ids_pointer, ids)
            self.runtime.zero(loss_pointer, 4)
            if weight_values is not None:
                self.runtime.upload(weights_pointer, weight_values)
            self.runtime.launch(
                "mimillm_cross_entropy_loss", (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(logits_pointer), self._pointer(ids_pointer), self._pointer(weights_pointer),
                    ctypes.c_float(weight_sum), self._pointer(loss_pointer), self._count(rows), self._count(classes),
                ], shared_memory=BLOCK_SIZE * 4,
            )
            loss_storage = array("f", [0.0])
            self.runtime.download(loss_storage, loss_pointer)
            if gradient:
                self.runtime.launch(
                    "mimillm_softmax_rows", (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                    [self._pointer(logits_pointer), self._pointer(gradient_pointer), self._count(rows), self._count(classes)],
                    shared_memory=BLOCK_SIZE * 4,
                )
                self.runtime.launch(
                    "mimillm_cross_entropy_gradient", ((len(values) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                    [
                        self._pointer(gradient_pointer), self._pointer(ids_pointer), self._pointer(weights_pointer),
                        ctypes.c_float(weight_sum), self._count(len(values)), self._count(classes),
                    ],
                )
                assert output_gradient is not None
                self.runtime.download(output_gradient, gradient_pointer)
        return float(loss_storage[0]), output_gradient

    def cross_entropy(self, logits: Sequence[float], targets: Sequence[int], rows: int, classes: int) -> float:
        return self._cross_entropy(logits, targets, rows, classes, None, gradient=False)[0]

    def cross_entropy_backward(
        self, logits: Sequence[float], targets: Sequence[int], rows: int, classes: int,
    ) -> array:
        result = self._cross_entropy(logits, targets, rows, classes, None, gradient=True)[1]
        assert result is not None
        return result

    def weighted_cross_entropy(
        self, logits: Sequence[float], targets: Sequence[int], weights: Sequence[float],
        rows: int, classes: int, *, compute_gradient: bool = True,
    ) -> tuple[float, array | None]:
        loss, gradient = self._cross_entropy(
            logits, targets, rows, classes, weights, gradient=compute_gradient,
        )
        return loss, gradient

    def adamw_update(
        self, parameter: array, gradient: Sequence[float], first_moment: array,
        second_moment: array, *, learning_rate: float, beta1: float, beta2: float,
        epsilon: float, weight_decay: float, step: int,
    ) -> None:
        gradient_values = _float_storage(gradient)
        count = len(parameter)
        size = count * 4
        with self.runtime.buffers(size, size, size, size) as pointers:
            parameter_pointer, gradient_pointer, first_pointer, second_pointer = pointers
            for pointer, storage in (
                (parameter_pointer, parameter), (gradient_pointer, gradient_values),
                (first_pointer, first_moment), (second_pointer, second_moment),
            ):
                self.runtime.upload(pointer, storage)
            correction1 = 1.0 - beta1 ** step
            correction2 = 1.0 - beta2 ** step
            self.runtime.launch(
                "mimillm_adamw", ((count + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(parameter_pointer), self._pointer(gradient_pointer),
                    self._pointer(first_pointer), self._pointer(second_pointer), self._count(count),
                    ctypes.c_float(learning_rate), ctypes.c_float(beta1), ctypes.c_float(beta2),
                    ctypes.c_float(epsilon), ctypes.c_float(weight_decay),
                    ctypes.c_float(correction1), ctypes.c_float(correction2),
                ],
            )
            self.runtime.download(parameter, parameter_pointer)
            self.runtime.download(first_moment, first_pointer)
            self.runtime.download(second_moment, second_pointer)

    def sum_squares(self, values: Sequence[float]) -> float:
        source = _float_storage(values)
        with self.runtime.buffers(len(source) * 4, 4) as (source_pointer, output_pointer):
            self.runtime.upload(source_pointer, source)
            self.runtime.zero(output_pointer, 4)
            self.runtime.launch(
                "mimillm_sum_squares", ((len(source) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(source_pointer), self._pointer(output_pointer), self._count(len(source))],
            )
            output = array("f", [0.0])
            self.runtime.download(output, output_pointer)
        return float(output[0])

    def scale_inplace(self, values: array, scalar: float) -> None:
        size = len(values) * 4
        with self.runtime.buffers(size) as (pointer,):
            self.runtime.upload(pointer, values)
            self.runtime.launch(
                "mimillm_scalar_multiply", ((len(values) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(pointer), ctypes.c_float(scalar), self._pointer(pointer), self._count(len(values))],
            )
            self.runtime.download(values, pointer)


def is_available() -> bool:
    """Return whether the NVIDIA driver, NVRTC, and bundled kernels are usable."""
    try:
        _runtime()
        return True
    except (FileNotFoundError, OSError, RuntimeError):
        return False
