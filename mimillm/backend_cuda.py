"""NVIDIA CUDA backend loaded directly through the Driver API and NVRTC."""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import weakref
from array import array
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


CUDA_SUCCESS = 0
CUDA_ERROR_OUT_OF_MEMORY = 2
BLOCK_SIZE = 256
TILE_SIZE = 16
DEFAULT_POOL_LIMIT_BYTES = 256 * 1024 * 1024
DEFAULT_POOL_BLOCKS_PER_SIZE = 2
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


class _CudaArray(array):
    """Host mirror whose matching device allocation may be reused by CUDA ops."""

    _mimillm_cuda_array = True

    def _ensure_host(self) -> None:
        if getattr(self, "_host_current", True):
            return
        runtime_reference = getattr(self, "_runtime_reference", None)
        runtime = runtime_reference() if runtime_reference is not None else None
        if runtime is None:
            raise RuntimeError("CUDA tensor lost its runtime before host synchronization")
        runtime.materialize(self)

    def __iter__(self):
        self._ensure_host()
        return super().__iter__()

    def __getitem__(self, key):
        self._ensure_host()
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self._ensure_host()
        super().__setitem__(key, value)
        self._host_current = True
        self._device_current = False

    def __repr__(self) -> str:
        self._ensure_host()
        return super().__repr__()

    def tolist(self) -> list[float]:
        self._ensure_host()
        return super().tolist()

    def tobytes(self) -> bytes:
        self._ensure_host()
        return super().tobytes()


def _output_storage(count: int) -> _CudaArray:
    storage = _CudaArray("f")
    storage.frombytes(bytes(count * 4))
    return storage


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


class _Cublas:
    """Minimal cuBLAS wrapper sharing mimiLLM's current Driver API context."""

    def __init__(self) -> None:
        self.library = self._load_library()
        self._configure()
        self.handle = ctypes.c_void_p()
        self._check(self.library.cublasCreate_v2(ctypes.byref(self.handle)), "cublasCreate")

    @staticmethod
    def _load_library() -> ctypes.CDLL:
        loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
        candidates: list[Path] = []
        if sys.platform == "win32":
            roots = [
                Path(value) for value in (
                    os.environ.get("CUDA_PATH"),
                    os.environ.get("CUDA_HOME"),
                ) if value
            ]
            toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
            if toolkit_root.is_dir():
                roots.extend(sorted(toolkit_root.glob("v*"), reverse=True))
            for root in roots:
                candidates.extend(sorted((root / "bin").glob("cublas64_*.dll"), reverse=True))
        else:
            for root in filter(None, (os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME"))):
                candidates.extend(sorted((Path(root) / "lib64").glob("libcublas.so*"), reverse=True))
            candidates.extend(Path("/usr/local/cuda/lib64").glob("libcublas.so*"))
        for candidate in candidates:
            try:
                return loader(str(candidate))
            except OSError:
                continue
        for name in (
            ("cublas64_13.dll", "cublas64_12.dll")
            if sys.platform == "win32"
            else ("libcublas.so", "libcublas.so.13", "libcublas.so.12")
        ):
            try:
                return loader(name)
            except OSError:
                continue
        raise FileNotFoundError("cuBLAS was not found")

    def _configure(self) -> None:
        library = self.library
        library.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.cublasCreate_v2.restype = ctypes.c_int
        library.cublasSgemm_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_int,
        ]
        library.cublasSgemm_v2.restype = ctypes.c_int
        library.cublasSgemmStridedBatched.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
            ctypes.c_longlong, ctypes.c_int,
        ]
        library.cublasSgemmStridedBatched.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed with cuBLAS status {status}")

    def matmul(
        self, left: int, right: int, output: int,
        batches: int, rows: int, inner: int, columns: int,
        *,
        transpose_left: bool = False,
        transpose_right: bool = False,
    ) -> None:
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        operation_right = 1 if transpose_right else 0
        operation_left = 1 if transpose_left else 0
        right_leading = inner if transpose_right else columns
        left_leading = rows if transpose_left else inner
        arguments = (
            self.handle, operation_right, operation_left,
            columns, rows, inner,
            ctypes.byref(alpha), ctypes.c_void_p(right), right_leading,
            ctypes.c_void_p(left), left_leading, ctypes.byref(beta),
            ctypes.c_void_p(output), columns,
        )
        if batches == 1:
            self._check(self.library.cublasSgemm_v2(*arguments), "cublasSgemm")
            return
        self._check(
            self.library.cublasSgemmStridedBatched(
                self.handle, operation_right, operation_left,
                columns, rows, inner,
                ctypes.byref(alpha), ctypes.c_void_p(right), right_leading,
                inner * columns,
                ctypes.c_void_p(left), left_leading, rows * inner,
                ctypes.byref(beta), ctypes.c_void_p(output), columns,
                rows * columns, batches,
            ),
            "cublasSgemmStridedBatched",
        )


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
        self._pool_bytes = 0
        configured_pool_mb = os.environ.get("MIMILLM_CUDA_POOL_MB")
        if configured_pool_mb is None:
            self._pool_limit_bytes = min(
                DEFAULT_POOL_LIMIT_BYTES,
                max(32 * 1024 * 1024, self.device_memory // 16),
            )
        else:
            try:
                pool_mb = int(configured_pool_mb)
            except ValueError as exc:
                raise ValueError(
                    "MIMILLM_CUDA_POOL_MB must be a non-negative integer"
                ) from exc
            if pool_mb < 0:
                raise ValueError(
                    "MIMILLM_CUDA_POOL_MB must be a non-negative integer"
                )
            self._pool_limit_bytes = pool_mb * 1024 * 1024
        self._pool_blocks_per_size = DEFAULT_POOL_BLOCKS_PER_SIZE
        self._tensor_cache: dict[int, tuple[weakref.ReferenceType[array], int, int]] = {}
        self._persistent_arrays: set[int] = set()
        try:
            self.cublas: _Cublas | None = _Cublas()
        except (FileNotFoundError, OSError, RuntimeError):
            self.cublas = None

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

    @staticmethod
    def _allocation_size(size: int) -> int:
        """Round temporary allocations into reusable size classes."""
        size = max(1, int(size))
        quantum = 64 * 1024 if size <= 1024 * 1024 else 1024 * 1024
        return ((size + quantum - 1) // quantum) * quantum

    def allocate(self, size: int) -> int:
        allocation_size = self._allocation_size(size)
        if self._pool[allocation_size]:
            self._pool_bytes -= allocation_size
            return self._pool[allocation_size].pop()
        pointer = ctypes.c_uint64()
        status = self.driver.cuMemAlloc_v2(
            ctypes.byref(pointer), allocation_size
        )
        if status == CUDA_ERROR_OUT_OF_MEMORY and self._pool_bytes:
            self.empty_cache()
            status = self.driver.cuMemAlloc_v2(
                ctypes.byref(pointer), allocation_size
            )
        self._check(status, "cuMemAlloc")
        return int(pointer.value)

    def release(self, pointer: int, size: int) -> None:
        if not pointer:
            return
        allocation_size = self._allocation_size(size)
        bucket = self._pool[allocation_size]
        should_cache = (
            allocation_size <= self._pool_limit_bytes
            and self._pool_bytes + allocation_size <= self._pool_limit_bytes
            and len(bucket) < self._pool_blocks_per_size
        )
        if should_cache:
            bucket.append(pointer)
            self._pool_bytes += allocation_size
            return
        self._check(self.driver.cuMemFree_v2(pointer), "cuMemFree")

    def empty_cache(self) -> None:
        """Return all inactive pooled allocations to the CUDA driver."""
        first_error: RuntimeError | None = None
        for bucket in self._pool.values():
            while bucket:
                pointer = bucket.pop()
                try:
                    self._check(self.driver.cuMemFree_v2(pointer), "cuMemFree")
                except RuntimeError as exc:
                    if first_error is None:
                        first_error = exc
        self._pool.clear()
        self._pool_bytes = 0
        if first_error is not None:
            raise first_error

    def memory_stats(self) -> dict[str, int]:
        """Report owned CUDA allocations without materializing any tensors."""
        cached_bytes = sum(
            size for reference, _, size in self._tensor_cache.values()
            if reference() is not None
        )
        return {
            "pool_bytes": self._pool_bytes,
            "pool_limit_bytes": self._pool_limit_bytes,
            "pool_blocks": sum(len(bucket) for bucket in self._pool.values()),
            "tensor_cache_bytes": cached_bytes,
            "tensor_cache_entries": sum(
                1
                for reference, _, _ in self._tensor_cache.values()
                if reference() is not None
            ),
            "persistent_entries": len(self._persistent_arrays),
        }

    def _release_cached(
        self, identity: int, reference: weakref.ReferenceType[array],
    ) -> None:
        entry = self._tensor_cache.get(identity)
        if entry is None or entry[0] is not reference:
            return
        _, pointer, size = self._tensor_cache.pop(identity)
        self._persistent_arrays.discard(identity)
        self.release(pointer, size)

    def _cached_pointer(self, storage: array, size: int) -> int | None:
        entry = self._tensor_cache.get(id(storage))
        if entry is None:
            return None
        reference, pointer, cached_size = entry
        if reference() is storage and cached_size == size:
            return pointer
        self._tensor_cache.pop(id(storage), None)
        self._persistent_arrays.discard(id(storage))
        if reference() is not None:
            self.release(pointer, cached_size)
        return None

    def register_persistent(self, storage: array) -> None:
        """Keep an authoritative device mirror until the host array is released."""
        size = max(1, len(storage) * storage.itemsize)
        self._persistent_arrays.add(id(storage))
        cached = self._cached_pointer(storage, size)
        if cached is not None:
            if (
                getattr(storage, "_mimillm_cuda_array", False)
                and not getattr(storage, "_device_current", True)
            ):
                self.upload(cached, storage)
            return
        pointer = self.allocate(size)
        self.upload(pointer, storage)
        self._cache_pointer(storage, pointer, size)

    def _take_cached_pointer(self, storage: array, size: int) -> int | None:
        pointer = self._cached_pointer(storage, size)
        if pointer is not None:
            self._tensor_cache.pop(id(storage), None)
        return pointer

    def _cache_pointer(self, storage: array, pointer: int, size: int) -> None:
        identity = id(storage)
        existing = self._tensor_cache.pop(identity, None)
        if existing is not None and existing[0]() is not storage:
            self.release(existing[1], existing[2])
        reference = weakref.ref(
            storage,
            lambda ref, key=identity: self._release_cached(key, ref),
        )
        self._tensor_cache[identity] = (reference, pointer, size)

    @contextmanager
    def tensor_buffers(
        self, *specifications: tuple[array, str],
    ) -> Iterator[tuple[int, ...]]:
        """Borrow cached device mirrors for input, output, or in-place arrays."""
        pointers: list[int] = []
        temporary: list[tuple[int, int]] = []
        cache_after: list[tuple[array, int, int]] = []
        completed = False
        try:
            for storage, mode in specifications:
                if mode not in {"in", "out", "inout"}:
                    raise ValueError(f"unknown CUDA buffer mode: {mode}")
                size = max(1, len(storage) * storage.itemsize)
                persistent = id(storage) in self._persistent_arrays
                reusable = persistent or getattr(
                    storage, "_mimillm_cuda_array", False
                )
                pointer = (
                    self._cached_pointer(storage, size)
                    if mode == "in" and reusable
                    else self._take_cached_pointer(storage, size)
                    if mode == "in"
                    else self._cached_pointer(storage, size)
                    if mode == "inout"
                    else None
                )
                if pointer is None:
                    pointer = self.allocate(size)
                    temporary.append((pointer, size))
                    if mode in {"in", "inout"}:
                        self.upload(pointer, storage)
                elif (
                    mode in {"in", "inout"}
                    and getattr(storage, "_mimillm_cuda_array", False)
                    and not getattr(storage, "_device_current", True)
                ):
                    self.upload(pointer, storage)
                elif mode == "in" and not reusable:
                    temporary.append((pointer, size))
                pointers.append(pointer)
                if (
                    mode in {"out", "inout"}
                    and (
                        getattr(storage, "_mimillm_cuda_array", False)
                        or persistent
                    )
                    and self._cached_pointer(storage, size) is None
                ):
                    cache_after.append((storage, pointer, size))
            yield tuple(pointers)
            completed = True
        finally:
            cached_pointers: set[int] = set()
            if completed:
                for storage, pointer, size in cache_after:
                    self._cache_pointer(storage, pointer, size)
                    cached_pointers.add(pointer)
            for pointer, size in temporary:
                if pointer not in cached_pointers:
                    self.release(pointer, size)

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
        if getattr(storage, "_mimillm_cuda_array", False):
            storage._ensure_host()
        size = len(storage) * storage.itemsize
        if not size:
            return
        view = (ctypes.c_ubyte * size).from_buffer(storage)
        self._check(
            self.driver.cuMemcpyHtoD_v2(pointer, ctypes.cast(view, ctypes.c_void_p), size),
            "cuMemcpyHtoD",
        )
        if getattr(storage, "_mimillm_cuda_array", False):
            storage._device_current = True

    def download(self, storage: array, pointer: int) -> None:
        size = len(storage) * storage.itemsize
        if not size:
            return
        view = (ctypes.c_ubyte * size).from_buffer(storage)
        self._check(
            self.driver.cuMemcpyDtoH_v2(ctypes.cast(view, ctypes.c_void_p), pointer, size),
            "cuMemcpyDtoH",
        )

    def finish_output(self, storage: array, pointer: int) -> None:
        """Keep CUDA arrays device-resident; materialize ordinary arrays immediately."""
        if getattr(storage, "_mimillm_cuda_array", False):
            storage._host_current = False
            storage._device_current = True
            storage._runtime_reference = weakref.ref(self)
            return
        self.download(storage, pointer)

    def materialize(self, storage: _CudaArray) -> None:
        size = max(1, len(storage) * storage.itemsize)
        pointer = self._cached_pointer(storage, size)
        if pointer is None:
            raise RuntimeError("CUDA tensor has no device allocation to materialize")
        self.download(storage, pointer)
        storage._host_current = True

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

    def synchronize(self) -> None:
        """Wait for queued CUDA work when a host-visible result is not requested."""
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

    def register_optimizer_state(
        self,
        parameters: Sequence[array],
        first_moments: Sequence[array],
        second_moments: Sequence[array],
    ) -> None:
        """Keep model parameters and AdamW moments resident on the GPU."""
        for storage in (*parameters, *first_moments, *second_moments):
            self.runtime.register_persistent(storage)

    def prepare_optimizer_state(
        self,
        parameters: Sequence[array],
        first_moments: Sequence[array],
        second_moments: Sequence[array],
    ) -> tuple[list[array], list[array], list[array]]:
        """Create lazily synchronized host mirrors for CUDA optimizer state."""
        groups: list[list[array]] = []
        for values in (parameters, first_moments, second_moments):
            converted: list[array] = []
            for storage in values:
                if getattr(storage, "_mimillm_cuda_array", False):
                    resident = storage
                else:
                    resident = _CudaArray("f", storage)
                    resident._host_current = True
                    resident._device_current = False
                converted.append(resident)
            groups.append(converted)
        prepared = (groups[0], groups[1], groups[2])
        self.register_optimizer_state(*prepared)
        return prepared

    def empty_cache(self) -> None:
        """Release inactive CUDA workspaces while preserving live tensors."""
        self.runtime.empty_cache()

    def memory_stats(self) -> dict[str, int]:
        """Return CUDA cache statistics for diagnostics and stress tests."""
        return self.runtime.memory_stats()

    def _elementwise_binary(
        self, kernel: str, left: Sequence[float], right: Sequence[float],
    ) -> array:
        if len(left) != len(right):
            raise ValueError(f"{kernel}: buffer lengths do not match")
        left_values, right_values = _float_storage(left), _float_storage(right)
        output = _output_storage(len(left_values))
        with self.runtime.tensor_buffers(
            (left_values, "in"), (right_values, "in"), (output, "out"),
        ) as (a, b, result):
            self.runtime.launch(
                kernel, ((len(output) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1),
                (BLOCK_SIZE, 1, 1),
                [self._pointer(a), self._pointer(b), self._pointer(result), self._count(len(output))],
            )
            self.runtime.finish_output(output, result)
        return output

    def add(self, left: Sequence[float], right: Sequence[float]) -> array:
        return self._elementwise_binary("mimillm_add", left, right)

    def add_row_vector(
        self, values: Sequence[float], bias: Sequence[float],
        rows: int, columns: int,
    ) -> array:
        """Add one feature vector to every row without broadcast metadata."""
        source, bias_values = _float_storage(values), _float_storage(bias)
        if len(source) != rows * columns or len(bias_values) != columns:
            raise ValueError("row-vector addition buffers do not match their shape")
        output = _output_storage(len(source))
        with self.runtime.tensor_buffers(
            (source, "in"), (bias_values, "in"), (output, "out"),
        ) as (source_pointer, bias_pointer, output_pointer):
            self.runtime.launch(
                "mimillm_add_row_vector",
                ((len(source) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1),
                (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(source_pointer), self._pointer(bias_pointer),
                    self._pointer(output_pointer), self._count(len(source)),
                    self._count(columns),
                ],
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def sum_columns(
        self, values: Sequence[float], rows: int, columns: int,
    ) -> array:
        """Reduce a row-major matrix over rows for a bias gradient."""
        source = _float_storage(values)
        if len(source) != rows * columns:
            raise ValueError("column reduction buffer does not match its shape")
        output = _output_storage(columns)
        with self.runtime.tensor_buffers(
            (source, "in"), (output, "out"),
        ) as (source_pointer, output_pointer):
            self.runtime.launch(
                "mimillm_sum_columns", (columns, 1, 1), (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(source_pointer), self._pointer(output_pointer),
                    self._count(rows), self._count(columns),
                ],
                shared_memory=BLOCK_SIZE * 4,
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def multiply(self, left: Sequence[float], right: Sequence[float]) -> array:
        return self._elementwise_binary("mimillm_multiply", left, right)

    def scalar_multiply(self, values: Sequence[float], scalar: float) -> array:
        source = _float_storage(values)
        output = _output_storage(len(source))
        with self.runtime.tensor_buffers(
            (source, "in"), (output, "out"),
        ) as (input_pointer, output_pointer):
            self.runtime.launch(
                "mimillm_scalar_multiply",
                ((len(source) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(input_pointer), ctypes.c_float(scalar), self._pointer(output_pointer), self._count(len(source))],
            )
            self.runtime.finish_output(output, output_pointer)
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
        output = _output_storage(count)
        metadata_size = len(shape) * 8
        with self.runtime.tensor_buffers(
            (source, "in"), (output, "out"),
        ) as (source_pointer, output_pointer), self.runtime.buffers(
            metadata_size, metadata_size, metadata_size, metadata_size,
        ) as metadata_pointers:
            source_stride_pointer, shape_pointer, output_stride_pointer, axes_pointer = metadata_pointers
            for pointer, storage in (
                (source_stride_pointer, source_strides),
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
            self.runtime.finish_output(output, output_pointer)
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
        output = _output_storage(_product(output_shape))
        metadata_size = len(output_shape) * 8
        with self.runtime.tensor_buffers(
            (left_values, "in"), (right_values, "in"), (output, "out"),
        ) as (left_pointer, right_pointer, output_pointer), self.runtime.buffers(
            *([metadata_size] * 4),
        ) as metadata_pointers:
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
            self.runtime.finish_output(output, output_pointer)
        return output

    def broadcast_binary_backward(
        self, left: Sequence[float], right: Sequence[float], grad_output: Sequence[float],
        left_shape: Sequence[int], right_shape: Sequence[int],
        output_shape: Sequence[int], operation: str,
    ) -> tuple[array, array]:
        left_values, right_values, gradient = map(_float_storage, (left, right, grad_output))
        metadata = self._broadcast_metadata(left_shape, right_shape, output_shape)
        grad_left = _output_storage(len(left_values))
        grad_right = _output_storage(len(right_values))
        metadata_size = len(output_shape) * 8
        with self.runtime.tensor_buffers(
            (left_values, "in"), (right_values, "in"), (gradient, "in"),
            (grad_left, "out"), (grad_right, "out"),
        ) as tensor_pointers, self.runtime.buffers(
            *([metadata_size] * 4),
        ) as metadata_pointers:
            left_pointer, right_pointer, gradient_pointer, grad_left_pointer, grad_right_pointer = tensor_pointers
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
            self.runtime.finish_output(grad_left, grad_left_pointer)
            self.runtime.finish_output(grad_right, grad_right_pointer)
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
        output = _output_storage(batches * rows * columns)
        with self.runtime.tensor_buffers(
            (left_values, "in"), (right_values, "in"), (output, "out"),
        ) as pointers:
            left_pointer, right_pointer, output_pointer = pointers
            if self.runtime.cublas is not None:
                self.runtime.cublas.matmul(
                    left_pointer, right_pointer, output_pointer,
                    batches, rows, inner, columns,
                )
            else:
                self.runtime.launch(
                    "mimillm_matmul",
                    ((columns + TILE_SIZE - 1) // TILE_SIZE, (rows + TILE_SIZE - 1) // TILE_SIZE, batches),
                    (TILE_SIZE, TILE_SIZE, 1),
                    [
                        self._pointer(left_pointer), self._pointer(right_pointer), self._pointer(output_pointer),
                        self._count(batches), self._count(rows), self._count(inner), self._count(columns),
                    ],
                )
            self.runtime.finish_output(output, output_pointer)
        return output

    def matmul_backward_left(
        self, grad_output: Sequence[float], right: Sequence[float],
        batches: int, rows: int, inner: int, columns: int,
    ) -> array:
        """Compute ``grad_output @ right.T`` without materializing ``right.T``."""
        if self.runtime.cublas is None:
            right_shape = (
                (inner, columns)
                if batches == 1 else
                (batches, inner, columns)
            )
            axes = (
                (1, 0)
                if batches == 1 else
                (0, 2, 1)
            )
            transposed = self.permute(right, right_shape, axes)
            return self.batched_matmul(
                grad_output, transposed, batches, rows, columns, inner,
            )
        gradient, right_values = _float_storage(grad_output), _float_storage(right)
        if (
            len(gradient) != batches * rows * columns
            or len(right_values) != batches * inner * columns
        ):
            raise ValueError("matmul left-gradient buffers do not match their shape")
        output = _output_storage(batches * rows * inner)
        with self.runtime.tensor_buffers(
            (gradient, "in"), (right_values, "in"), (output, "out"),
        ) as (gradient_pointer, right_pointer, output_pointer):
            self.runtime.cublas.matmul(
                gradient_pointer, right_pointer, output_pointer,
                batches, rows, columns, inner,
                transpose_right=True,
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def matmul_backward_right(
        self, left: Sequence[float], grad_output: Sequence[float],
        batches: int, rows: int, inner: int, columns: int,
    ) -> array:
        """Compute ``left.T @ grad_output`` without materializing ``left.T``."""
        if self.runtime.cublas is None:
            left_shape = (
                (rows, inner)
                if batches == 1 else
                (batches, rows, inner)
            )
            axes = (
                (1, 0)
                if batches == 1 else
                (0, 2, 1)
            )
            transposed = self.permute(left, left_shape, axes)
            return self.batched_matmul(
                transposed, grad_output, batches, inner, rows, columns,
            )
        left_values, gradient = _float_storage(left), _float_storage(grad_output)
        if (
            len(left_values) != batches * rows * inner
            or len(gradient) != batches * rows * columns
        ):
            raise ValueError("matmul right-gradient buffers do not match their shape")
        output = _output_storage(batches * inner * columns)
        with self.runtime.tensor_buffers(
            (left_values, "in"), (gradient, "in"), (output, "out"),
        ) as (left_pointer, gradient_pointer, output_pointer):
            self.runtime.cublas.matmul(
                left_pointer, gradient_pointer, output_pointer,
                batches, inner, rows, columns,
                transpose_left=True,
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def softmax_rows(self, values: Sequence[float], rows: int, columns: int) -> array:
        return self._rows_operation("mimillm_softmax_rows", values, rows, columns)

    def causal_softmax_rows(
        self, values: Sequence[float], rows: int, columns: int,
        sequence_length: int, scale: float,
    ) -> array:
        """Apply attention scaling, causal masking, and softmax in one kernel."""
        source = _float_storage(values)
        if (
            len(source) != rows * columns
            or sequence_length <= 0
            or columns != sequence_length
            or rows % sequence_length
        ):
            raise ValueError("causal softmax buffer does not match its shape")
        output = _output_storage(len(source))
        with self.runtime.tensor_buffers(
            (source, "in"), (output, "out"),
        ) as (source_pointer, output_pointer):
            self.runtime.launch(
                "mimillm_causal_softmax_rows",
                (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(source_pointer), self._pointer(output_pointer),
                    self._count(rows), self._count(columns),
                    self._count(sequence_length), ctypes.c_float(scale),
                ],
                shared_memory=BLOCK_SIZE * 4,
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def _rows_operation(self, kernel: str, values: Sequence[float], rows: int, columns: int) -> array:
        source = _float_storage(values)
        if len(source) != rows * columns:
            raise ValueError(f"{kernel}: buffer size does not match its shape")
        output = _output_storage(len(source))
        with self.runtime.tensor_buffers(
            (source, "in"), (output, "out"),
        ) as (source_pointer, output_pointer):
            self.runtime.launch(
                kernel, (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(source_pointer), self._pointer(output_pointer), self._count(rows), self._count(columns)],
                shared_memory=BLOCK_SIZE * 4,
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def softmax_backward(
        self, output_values: Sequence[float], grad_output: Sequence[float],
        rows: int, columns: int,
    ) -> array:
        values, gradient = _float_storage(output_values), _float_storage(grad_output)
        if len(values) != rows * columns or len(gradient) != len(values):
            raise ValueError("softmax backward buffers do not match their shape")
        output = _output_storage(len(values))
        with self.runtime.tensor_buffers(
            (values, "in"), (gradient, "in"), (output, "out"),
        ) as pointers:
            values_pointer, gradient_pointer, output_pointer = pointers
            self.runtime.launch(
                "mimillm_softmax_backward", (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(values_pointer), self._pointer(gradient_pointer), self._pointer(output_pointer), self._count(rows), self._count(columns)],
                shared_memory=BLOCK_SIZE * 4,
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def causal_softmax_backward(
        self, output_values: Sequence[float], grad_output: Sequence[float],
        rows: int, columns: int, scale: float,
    ) -> array:
        """Backpropagate through the fused scaled causal softmax."""
        values, gradient = _float_storage(output_values), _float_storage(grad_output)
        if len(values) != rows * columns or len(gradient) != len(values):
            raise ValueError("causal softmax backward buffers do not match their shape")
        output = _output_storage(len(values))
        with self.runtime.tensor_buffers(
            (values, "in"), (gradient, "in"), (output, "out"),
        ) as (values_pointer, gradient_pointer, output_pointer):
            self.runtime.launch(
                "mimillm_causal_softmax_backward",
                (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(values_pointer), self._pointer(gradient_pointer),
                    self._pointer(output_pointer), self._count(rows),
                    self._count(columns), ctypes.c_float(scale),
                ],
                shared_memory=BLOCK_SIZE * 4,
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def rms_norm(
        self, values: Sequence[float], weight: Sequence[float],
        rows: int, columns: int, epsilon: float,
    ) -> array:
        """Normalize and scale rows with one CUDA kernel."""
        source, scale = _float_storage(values), _float_storage(weight)
        if len(source) != rows * columns or len(scale) != columns:
            raise ValueError("RMSNorm buffers do not match their shape")
        output = _output_storage(len(source))
        with self.runtime.tensor_buffers(
            (source, "in"), (scale, "in"), (output, "out"),
        ) as (source_pointer, scale_pointer, output_pointer):
            self.runtime.launch(
                "mimillm_rms_norm", (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(source_pointer), self._pointer(scale_pointer),
                    self._pointer(output_pointer), self._count(rows),
                    self._count(columns), ctypes.c_float(epsilon),
                ],
                shared_memory=BLOCK_SIZE * 4,
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def rms_norm_backward(
        self, values: Sequence[float], weight: Sequence[float],
        grad_output: Sequence[float], rows: int, columns: int, epsilon: float,
    ) -> tuple[array, array]:
        """Return input and scale gradients for fused RMSNorm."""
        source, scale, gradient = map(
            _float_storage, (values, weight, grad_output),
        )
        if (
            len(source) != rows * columns
            or len(scale) != columns
            or len(gradient) != len(source)
        ):
            raise ValueError("RMSNorm backward buffers do not match their shape")
        grad_input = _output_storage(len(source))
        grad_weight = _output_storage(columns)
        with self.runtime.tensor_buffers(
            (source, "in"), (scale, "in"), (gradient, "in"),
            (grad_input, "out"), (grad_weight, "out"),
        ) as pointers:
            (
                source_pointer, scale_pointer, gradient_pointer,
                grad_input_pointer, grad_weight_pointer,
            ) = pointers
            self.runtime.zero(grad_weight_pointer, columns * 4)
            self.runtime.launch(
                "mimillm_rms_norm_backward",
                (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(source_pointer), self._pointer(scale_pointer),
                    self._pointer(gradient_pointer), self._pointer(grad_input_pointer),
                    self._pointer(grad_weight_pointer), self._count(rows),
                    self._count(columns), ctypes.c_float(epsilon),
                ],
                shared_memory=BLOCK_SIZE * 8,
            )
            self.runtime.finish_output(grad_input, grad_input_pointer)
            self.runtime.finish_output(grad_weight, grad_weight_pointer)
        return grad_input, grad_weight

    def sum_rows(self, values: Sequence[float], rows: int, columns: int) -> array:
        source = _float_storage(values)
        output = _output_storage(rows)
        with self.runtime.tensor_buffers(
            (source, "in"), (output, "out"),
        ) as (source_pointer, output_pointer):
            self.runtime.launch(
                "mimillm_sum_rows", (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(source_pointer), self._pointer(output_pointer), self._count(rows), self._count(columns)],
                shared_memory=BLOCK_SIZE * 4,
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def sum_rows_backward(self, grad_output: Sequence[float], rows: int, columns: int) -> array:
        gradient = _float_storage(grad_output)
        output = _output_storage(rows * columns)
        with self.runtime.tensor_buffers(
            (gradient, "in"), (output, "out"),
        ) as (gradient_pointer, output_pointer):
            self.runtime.launch(
                "mimillm_sum_rows_backward", ((len(output) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1),
                (BLOCK_SIZE, 1, 1),
                [self._pointer(gradient_pointer), self._pointer(output_pointer), self._count(len(output)), self._count(columns)],
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def relu(self, values: Sequence[float]) -> array:
        return self._unary("mimillm_relu", values)

    def _unary(self, kernel: str, values: Sequence[float]) -> array:
        source = _float_storage(values)
        output = _output_storage(len(source))
        with self.runtime.tensor_buffers(
            (source, "in"), (output, "out"),
        ) as (source_pointer, output_pointer):
            self.runtime.launch(
                kernel, ((len(source) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(source_pointer), self._pointer(output_pointer), self._count(len(source))],
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def relu_backward(self, values: Sequence[float], grad_output: Sequence[float]) -> array:
        source, gradient = _float_storage(values), _float_storage(grad_output)
        output = _output_storage(len(source))
        with self.runtime.tensor_buffers(
            (source, "in"), (gradient, "in"), (output, "out"),
        ) as pointers:
            source_pointer, gradient_pointer, output_pointer = pointers
            self.runtime.launch(
                "mimillm_relu_backward", ((len(source) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(source_pointer), self._pointer(gradient_pointer), self._pointer(output_pointer), self._count(len(source))],
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def embedding_gather(
        self, table: Sequence[float], indices: Sequence[int], vocab: int, width: int,
    ) -> array:
        values, ids = _float_storage(table), _int32_storage(indices)
        output = _output_storage(len(ids) * width)
        with self.runtime.tensor_buffers(
            (values, "in"), (ids, "in"), (output, "out"),
        ) as pointers:
            table_pointer, ids_pointer, output_pointer = pointers
            self.runtime.launch(
                "mimillm_embedding_gather", ((len(output) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(table_pointer), self._pointer(ids_pointer), self._pointer(output_pointer), self._count(width), self._count(len(output))],
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def embedding_scatter_add(
        self, indices: Sequence[int], grad_output: Sequence[float], vocab: int, width: int,
    ) -> array:
        ids, gradient = _int32_storage(indices), _float_storage(grad_output)
        output = _output_storage(vocab * width)
        with self.runtime.tensor_buffers(
            (ids, "in"), (gradient, "in"), (output, "out"),
        ) as pointers:
            ids_pointer, gradient_pointer, output_pointer = pointers
            self.runtime.zero(output_pointer, len(output) * 4)
            self.runtime.launch(
                "mimillm_embedding_scatter", ((len(gradient) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(ids_pointer), self._pointer(gradient_pointer), self._pointer(output_pointer), self._count(width), self._count(len(gradient))],
            )
            self.runtime.finish_output(output, output_pointer)
        return output

    def _cross_entropy(
        self, logits: Sequence[float], targets: Sequence[int], rows: int, classes: int,
        weights: Sequence[float] | None, *, gradient: bool,
    ) -> tuple[float, array | None]:
        values, ids = _float_storage(logits), _int32_storage(targets)
        weight_values = _float_storage(weights) if weights is not None else None
        weight_sum = float(sum(weight_values)) if weight_values is not None else float(rows)
        output_gradient = _output_storage(len(values)) if gradient else None
        tensor_specs: list[tuple[array, str]] = [
            (values, "in"), (ids, "in"),
        ]
        if weight_values is not None:
            tensor_specs.append((weight_values, "in"))
        if output_gradient is not None:
            tensor_specs.append((output_gradient, "out"))
        with self.runtime.tensor_buffers(*tensor_specs) as tensor_pointers, self.runtime.buffers(
            4,
        ) as (loss_pointer,):
            remaining = list(tensor_pointers)
            logits_pointer = remaining.pop(0)
            ids_pointer = remaining.pop(0)
            weights_pointer = remaining.pop(0) if weight_values is not None else 0
            gradient_pointer = remaining.pop(0) if output_gradient is not None else 0
            self.runtime.zero(loss_pointer, 4)
            self.runtime.launch(
                "mimillm_cross_entropy_loss", (rows, 1, 1), (BLOCK_SIZE, 1, 1),
                [
                    self._pointer(logits_pointer), self._pointer(ids_pointer), self._pointer(weights_pointer),
                    ctypes.c_float(weight_sum), self._pointer(loss_pointer), self._count(rows), self._count(classes),
                ], shared_memory=BLOCK_SIZE * 4,
            )
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
                self.runtime.finish_output(output_gradient, gradient_pointer)
            loss_storage = array("f", [0.0])
            self.runtime.download(loss_storage, loss_pointer)
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
        with self.runtime.tensor_buffers(
            (parameter, "inout"), (gradient_values, "in"),
            (first_moment, "inout"), (second_moment, "inout"),
        ) as pointers:
            parameter_pointer, gradient_pointer, first_pointer, second_pointer = pointers
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
            self.runtime.finish_output(parameter, parameter_pointer)
            self.runtime.finish_output(first_moment, first_pointer)
            self.runtime.finish_output(second_moment, second_pointer)

    def sum_squares(self, values: Sequence[float]) -> float:
        return self.global_sum_squares((values,))

    def global_sum_squares(
        self, values: Sequence[Sequence[float]],
    ) -> float:
        """Reduce multiple resident gradients with one host synchronization."""
        sources = [_float_storage(source) for source in values if len(source)]
        if not sources:
            return 0.0
        with self.runtime.buffers(4) as (output_pointer,):
            self.runtime.zero(output_pointer, 4)
            for source in sources:
                with self.runtime.tensor_buffers(
                    (source, "in"),
                ) as (source_pointer,):
                    self.runtime.launch(
                        "mimillm_sum_squares",
                        ((len(source) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1),
                        (BLOCK_SIZE, 1, 1),
                        [
                            self._pointer(source_pointer),
                            self._pointer(output_pointer),
                            self._count(len(source)),
                        ],
                    )
            output = array("f", [0.0])
            self.runtime.finish_output(output, output_pointer)
        return float(output[0])

    def scale_inplace(self, values: array, scalar: float) -> None:
        with self.runtime.tensor_buffers((values, "inout")) as (pointer,):
            self.runtime.launch(
                "mimillm_scalar_multiply", ((len(values) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1), (BLOCK_SIZE, 1, 1),
                [self._pointer(pointer), ctypes.c_float(scalar), self._pointer(pointer), self._count(len(values))],
            )
            self.runtime.finish_output(values, pointer)


def is_available() -> bool:
    """Return whether the NVIDIA driver, NVRTC, and bundled kernels are usable."""
    try:
        _runtime()
        return True
    except (FileNotFoundError, OSError, RuntimeError):
        return False
