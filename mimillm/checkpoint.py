"""Версионированный бинарный checkpoint без небезопасного pickle."""

from __future__ import annotations

import json
import os
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .module import Module
from .optim import Optimizer
from .tensor import Tensor


MAGIC = b"MIMILLM1"
VERSION = 2
SUPPORTED_VERSIONS = {1, 2}
HEADER = struct.Struct("<8sIQ")
MAX_METADATA = 16 * 1024 * 1024
MAX_VALUES = 500_000_000
_OPTIMIZER_BUFFER_FIELDS = (
    "first_moments",
    "second_moments",
    "gradient_accumulators",
)


@dataclass
class CheckpointData:
    """Проверенное содержимое, пригодное для инспекции или загрузки."""

    config: dict[str, Any]
    step: int
    seed: int
    parameters: dict[str, Tensor]
    optimizer_state: dict[str, object] | None


def _shape_product(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension < 0:
            raise ValueError("checkpoint содержит некорректную форму")
        result *= dimension
    return result


def _write_floats(stream: Any, values: array) -> None:
    data = array("f", values)
    if sys.byteorder != "little":
        data.byteswap()
    stream.write(data.tobytes())


def _read_floats(stream: Any, count: int) -> array:
    if count < 0 or count > MAX_VALUES:
        raise ValueError("checkpoint содержит недопустимое число float32")
    raw = stream.read(count * 4)
    if len(raw) != count * 4:
        raise ValueError("checkpoint неожиданно закончился внутри float32-буфера")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _optimizer_payload(
    optimizer_state: dict[str, object],
) -> tuple[dict[str, object], list[array]]:
    """Отделяет большие float32-буферы от JSON metadata."""
    optimizer_meta = {
        key: value
        for key, value in optimizer_state.items()
        if key not in _OPTIMIZER_BUFFER_FIELDS
    }
    descriptors: list[dict[str, object]] = []
    payload: list[array] = []
    for name in _OPTIMIZER_BUFFER_FIELDS:
        values = optimizer_state.get(name, [])
        if not isinstance(values, list):
            raise TypeError(f"optimizer state field {name!r} must be a list")
        if not values:
            continue
        checked: list[array] = []
        for buffer in values:
            if not isinstance(buffer, array) or buffer.typecode != "f":
                raise TypeError(
                    f"optimizer state field {name!r} must contain array('f') buffers"
                )
            checked.append(buffer)
        descriptors.append({
            "name": name,
            "counts": [len(buffer) for buffer in checked],
        })
        payload.extend(checked)
    if descriptors:
        optimizer_meta["buffer_descriptors"] = descriptors
    return optimizer_meta, payload


def save_checkpoint(
    path: str | Path, model: Module, optimizer: Optimizer | None, *,
    config: dict[str, Any], step: int, seed: int,
) -> Path:
    """Атомарно сохраняет схему JSON и последовательные бинарные буферы."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parameters = list(model.named_parameters())
    descriptors = [
        {"name": name, "shape": list(parameter.shape), "count": parameter.numel}
        for name, parameter in parameters
    ]
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    optimizer_meta: dict[str, object] | None = None
    optimizer_buffers: list[array] = []
    if optimizer_state is not None:
        optimizer_meta, optimizer_buffers = _optimizer_payload(optimizer_state)
    metadata = {
        "format": "mimiLLM-checkpoint", "config": config, "step": int(step),
        "seed": int(seed), "parameters": descriptors, "optimizer": optimizer_meta,
    }
    encoded = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_METADATA:
        raise ValueError("metadata checkpoint слишком велики")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, VERSION, len(encoded)))
        stream.write(encoded)
        for _, parameter in parameters:
            _write_floats(stream, parameter.data)
        for values in optimizer_buffers:
            _write_floats(stream, values)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination


def _load_optimizer_v1(
    stream: Any, optimizer_meta: dict[str, object],
) -> dict[str, object]:
    """Читает старую схему: один список размеров для first/second moments."""
    optimizer_state = dict(optimizer_meta)
    counts = optimizer_state.pop("moment_counts", [])
    if counts:
        if not isinstance(counts, list) or not all(isinstance(count, int) for count in counts):
            raise ValueError("некорректные размеры moments")
        optimizer_state["first_moments"] = [
            _read_floats(stream, count) for count in counts
        ]
        optimizer_state["second_moments"] = [
            _read_floats(stream, count) for count in counts
        ]
    return optimizer_state


def _load_optimizer_v2(
    stream: Any, optimizer_meta: dict[str, object],
) -> dict[str, object]:
    """Читает именованные группы float32-буферов optimizer state."""
    optimizer_state = dict(optimizer_meta)
    descriptors = optimizer_state.pop("buffer_descriptors", [])
    if not isinstance(descriptors, list):
        raise ValueError("optimizer buffer_descriptors должны быть списком")
    seen: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("неверный descriptor optimizer buffer")
        name = descriptor.get("name")
        counts = descriptor.get("counts")
        if (
            not isinstance(name, str)
            or name not in _OPTIMIZER_BUFFER_FIELDS
            or name in seen
        ):
            raise ValueError("неверное или повторное имя optimizer buffer")
        if (
            not isinstance(counts, list)
            or not all(
                isinstance(count, int) and not isinstance(count, bool) and count >= 0
                for count in counts
            )
        ):
            raise ValueError(f"некорректные размеры optimizer buffer {name!r}")
        optimizer_state[name] = [
            _read_floats(stream, count) for count in counts
        ]
        seen.add(name)
    return optimizer_state


def load_checkpoint(
    path: str | Path, model: Module | None = None, optimizer: Optimizer | None = None,
) -> CheckpointData:
    """Проверяет заголовок, размеры, конец файла и при необходимости загружает объекты."""
    source_path = Path(path)
    with source_path.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("checkpoint слишком короткий")
        magic, version, metadata_size = HEADER.unpack(header)
        if magic != MAGIC:
            raise ValueError("неверный magic header checkpoint")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"неподдерживаемая версия checkpoint: {version}")
        if metadata_size > MAX_METADATA:
            raise ValueError("metadata checkpoint превышают безопасный предел")
        encoded = stream.read(metadata_size)
        if len(encoded) != metadata_size:
            raise ValueError("checkpoint оборван внутри metadata")
        try:
            metadata = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"повреждены metadata checkpoint: {exc}") from exc
        if not isinstance(metadata, dict) or not isinstance(metadata.get("parameters"), list):
            raise ValueError("metadata checkpoint имеют неверную схему")
        if metadata.get("format") != "mimiLLM-checkpoint":
            raise ValueError("checkpoint содержит неизвестный формат metadata")
        parameters: dict[str, Tensor] = {}
        for descriptor in metadata["parameters"]:
            if not isinstance(descriptor, dict):
                raise ValueError("неверный descriptor параметра")
            name, shape, count = descriptor.get("name"), descriptor.get("shape"), descriptor.get("count")
            if not isinstance(name, str) or not isinstance(shape, list) or not isinstance(count, int):
                raise ValueError("неверные поля descriptor параметра")
            if _shape_product(shape) != count:
                raise ValueError(f"форма и count параметра {name} не совпадают")
            if name in parameters:
                raise ValueError(f"повторное имя параметра: {name}")
            parameters[name] = Tensor(_read_floats(stream, count), tuple(shape))
        optimizer_meta = metadata.get("optimizer")
        optimizer_state: dict[str, object] | None = None
        if optimizer_meta is not None:
            if not isinstance(optimizer_meta, dict):
                raise ValueError("optimizer metadata должны быть объектом")
            optimizer_state = (
                _load_optimizer_v1(stream, optimizer_meta)
                if version == 1
                else _load_optimizer_v2(stream, optimizer_meta)
            )
        if stream.read(1):
            raise ValueError("после ожидаемого конца checkpoint обнаружены лишние данные")
    data = CheckpointData(
        config=dict(metadata.get("config", {})), step=int(metadata.get("step", 0)),
        seed=int(metadata.get("seed", 0)), parameters=parameters,
        optimizer_state=optimizer_state,
    )
    if model is not None:
        model.load_state_dict(parameters)
    if optimizer is not None:
        if optimizer_state is None:
            raise ValueError("checkpoint не содержит состояние оптимизатора")
        if not hasattr(optimizer, "load_state_dict"):
            raise TypeError("оптимизатор не поддерживает load_state_dict")
        optimizer.load_state_dict(optimizer_state)  # type: ignore[attr-defined]
    return data
