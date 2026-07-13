"""Minimal, dependency-free SafeTensors reader and writer for float32 tensors."""

from __future__ import annotations

import json
import os
import struct
import sys
from array import array
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .tensor import Tensor


HEADER_LENGTH = struct.Struct("<Q")
MAX_HEADER_SIZE = 100 * 1024 * 1024


def _shape_size(shape: list[int]) -> int:
    size = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise ValueError("SafeTensors shape dimensions must be non-negative integers")
        size *= dimension
    return size


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate SafeTensors header key: {key}")
        result[key] = value
    return result


def _float32_bytes(tensor: Tensor) -> bytes:
    values = array("f", tensor.data)
    if values.itemsize != 4:
        raise RuntimeError("this Python build does not provide 32-bit array('f')")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def save_safetensors(
    path: str | Path,
    tensors: Mapping[str, Tensor],
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Atomically writes float32 tensors using the documented SafeTensors layout."""
    if not tensors:
        raise ValueError("at least one tensor is required")
    if metadata is not None and not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        raise TypeError("SafeTensors metadata must contain only string keys and values")

    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = dict(metadata)
    buffers: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name or name == "__metadata__":
            raise ValueError(f"invalid SafeTensors tensor name: {name!r}")
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name!r} is not a mimiLLM Tensor")
        raw = _float32_bytes(tensor)
        end = offset + len(raw)
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        buffers.append(raw)
        offset = end

    encoded = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_HEADER_SIZE:
        raise ValueError("SafeTensors header exceeds the 100 MiB safety limit")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(HEADER_LENGTH.pack(len(encoded)))
        stream.write(encoded)
        for raw in buffers:
            stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination


def load_safetensors(path: str | Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Loads and validates an uncompressed float32 SafeTensors file."""
    source = Path(path)
    file_size = source.stat().st_size
    with source.open("rb") as stream:
        raw_length = stream.read(HEADER_LENGTH.size)
        if len(raw_length) != HEADER_LENGTH.size:
            raise ValueError("SafeTensors file is too short")
        (header_size,) = HEADER_LENGTH.unpack(raw_length)
        if header_size > MAX_HEADER_SIZE:
            raise ValueError("SafeTensors header exceeds the 100 MiB safety limit")
        if header_size > file_size - HEADER_LENGTH.size:
            raise ValueError("SafeTensors header extends beyond the file")
        encoded = stream.read(header_size)
        try:
            header = json.loads(
                encoded.decode("utf-8"), object_pairs_hook=_without_duplicate_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid SafeTensors JSON header: {exc}") from exc
        if not isinstance(header, dict):
            raise ValueError("SafeTensors header must be a JSON object")

        raw_metadata = header.pop("__metadata__", {})
        if not isinstance(raw_metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_metadata.items()
        ):
            raise ValueError("SafeTensors metadata must contain only string values")

        data_start = HEADER_LENGTH.size + header_size
        data_size = file_size - data_start
        descriptors: list[tuple[int, int, str, tuple[int, ...]]] = []
        for name, descriptor in header.items():
            if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
                raise ValueError("invalid SafeTensors tensor descriptor")
            if descriptor.get("dtype") != "F32":
                raise ValueError(
                    f"tensor {name!r} uses unsupported dtype {descriptor.get('dtype')!r}; "
                    "mimiLLM currently supports F32"
                )
            shape = descriptor.get("shape")
            offsets = descriptor.get("data_offsets")
            if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"tensor {name!r} has an invalid shape or data_offsets")
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in offsets):
                raise ValueError(f"tensor {name!r} has non-integer data offsets")
            begin, end = offsets
            expected = _shape_size(shape) * 4
            if begin < 0 or end < begin or end - begin != expected or end > data_size:
                raise ValueError(f"tensor {name!r} has invalid data offsets")
            descriptors.append((begin, end, name, tuple(shape)))

        descriptors.sort(key=lambda item: (item[0], item[1], item[2]))
        cursor = 0
        for begin, end, name, _ in descriptors:
            if begin != cursor:
                raise ValueError(f"SafeTensors data contains a gap or overlap before {name!r}")
            cursor = end
        if cursor != data_size:
            raise ValueError("SafeTensors data contains trailing bytes")

        tensors: dict[str, Tensor] = {}
        for begin, end, name, shape in descriptors:
            stream.seek(data_start + begin)
            raw = stream.read(end - begin)
            if len(raw) != end - begin:
                raise ValueError(f"tensor {name!r} ended unexpectedly")
            values = array("f")
            values.frombytes(raw)
            if values.itemsize != 4:
                raise RuntimeError("this Python build does not provide 32-bit array('f')")
            if sys.byteorder != "little":
                values.byteswap()
            tensors[name] = Tensor(values, shape)
    return tensors, dict(raw_metadata)
