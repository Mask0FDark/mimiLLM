"""Непрерывный float32 Tensor и минимальный динамический граф autograd."""

from __future__ import annotations

import math
import random
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from .backend import get_backend


_GRAD_ENABLED = True


def is_grad_enabled() -> bool:
    """Сообщает, строят ли новые операции динамический граф."""
    return _GRAD_ENABLED


@contextmanager
def no_grad() -> Iterator[None]:
    """Временно отключает граф для validation и авторегрессионного inference."""
    global _GRAD_ENABLED
    previous = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


def _product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return tuple(reversed(result))


def _unravel(index: int, shape: Sequence[int]) -> list[int]:
    coordinates: list[int] = []
    for stride, dimension in zip(_contiguous_strides(shape), shape):
        coordinates.append((index // stride) % dimension)
    return coordinates


def _ravel(coordinates: Sequence[int], strides: Sequence[int]) -> int:
    return sum(coordinate * stride for coordinate, stride in zip(coordinates, strides))


def _normalize_shape(shape: Sequence[int] | int | None, size: int) -> tuple[int, ...]:
    if shape is None:
        return (size,)
    if isinstance(shape, int):
        shape = (shape,)
    normalized = tuple(int(value) for value in shape)
    negative = [i for i, value in enumerate(normalized) if value == -1]
    if len(negative) > 1:
        raise ValueError("форма может содержать только один размер -1")
    if negative:
        known = _product([value for value in normalized if value != -1])
        if known == 0 or size % known:
            raise ValueError("невозможно вывести размер -1")
        normalized = tuple(size // known if value == -1 else value for value in normalized)
    if any(value < 0 for value in normalized):
        raise ValueError("размеры Tensor не могут быть отрицательными")
    if _product(normalized) != size:
        raise ValueError(f"форма {normalized} требует {_product(normalized)} элементов, получено {size}")
    return normalized


def _broadcast_shape(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    length = max(len(left), len(right))
    a = (1,) * (length - len(left)) + tuple(left)
    b = (1,) * (length - len(right)) + tuple(right)
    output: list[int] = []
    for left_dim, right_dim in zip(a, b):
        if left_dim != right_dim and left_dim != 1 and right_dim != 1:
            raise ValueError(f"несовместимые формы для broadcasting: {tuple(left)} и {tuple(right)}")
        output.append(max(left_dim, right_dim))
    return tuple(output)


def _broadcast_index(output_coords: Sequence[int], source_shape: Sequence[int]) -> int:
    if not source_shape:
        return 0
    offset = len(output_coords) - len(source_shape)
    coords = [0 if dim == 1 else output_coords[offset + i] for i, dim in enumerate(source_shape)]
    return _ravel(coords, _contiguous_strides(source_shape))


def _broadcast_map(source_shape: Sequence[int], output_shape: Sequence[int]) -> list[int]:
    """Строит карту один раз, ускоряя частые scalar/bias/leading broadcasts."""
    output_size = _product(output_shape)
    if tuple(source_shape) == tuple(output_shape):
        return list(range(output_size))
    if _product(source_shape) == 1:
        return [0] * output_size
    padded = (1,) * (len(output_shape) - len(source_shape)) + tuple(source_shape)
    first_value = next((index for index, dimension in enumerate(padded) if dimension != 1), len(padded))
    if all(dimension == 1 for dimension in padded[:first_value]) and padded[first_value:] == tuple(output_shape[first_value:]):
        source_size = _product(source_shape)
        return [index % source_size for index in range(output_size)]
    result: list[int] = []
    for flat in range(output_size):
        result.append(_broadcast_index(_unravel(flat, output_shape), source_shape))
    return result


class Tensor:
    """Плотный float32-тензор с данными и локальной backward-функцией.

    Все представления непрерывны. Это сознательное ограничение первой версии:
    операции перестановки осей копируют данные, поэтому C++ всегда получает
    простой указатель на линейный буфер без сведений о Python-объектах.
    """

    def __init__(
        self,
        data: Iterable[float] | array,
        shape: Sequence[int] | int | None = None,
        *,
        requires_grad: bool = False,
        parents: tuple["Tensor", ...] = (),
        backward_fn: Callable[[], None] | None = None,
        operation: str = "",
    ) -> None:
        self.data = (
            data
            if isinstance(data, array)
            and data.typecode == "f"
            and getattr(data, "_mimillm_cuda_array", False)
            else array("f", data)
        )
        self.shape = _normalize_shape(shape, len(self.data))
        self.strides = _contiguous_strides(self.shape)
        self.requires_grad = bool(requires_grad and _GRAD_ENABLED)
        self.grad: array | None = None
        self.parents = parents if self.requires_grad else ()
        self._backward_fn = backward_fn
        self.operation = operation

    @property
    def ndim(self) -> int:
        """Число осей."""
        return len(self.shape)

    @property
    def numel(self) -> int:
        """Число float32-элементов в буфере."""
        return len(self.data)

    @classmethod
    def zeros(cls, shape: Sequence[int] | int, *, requires_grad: bool = False) -> "Tensor":
        normalized = (shape,) if isinstance(shape, int) else tuple(shape)
        return cls(array("f", [0.0]) * _product(normalized), normalized, requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape: Sequence[int] | int, *, requires_grad: bool = False) -> "Tensor":
        normalized = (shape,) if isinstance(shape, int) else tuple(shape)
        return cls(array("f", [1.0]) * _product(normalized), normalized, requires_grad=requires_grad)

    @classmethod
    def randn(
        cls, shape: Sequence[int] | int, *, rng: random.Random | None = None,
        scale: float = 1.0, requires_grad: bool = False,
    ) -> "Tensor":
        normalized = (shape,) if isinstance(shape, int) else tuple(shape)
        source = rng or random
        return cls(
            (source.gauss(0.0, scale) for _ in range(_product(normalized))),
            normalized, requires_grad=requires_grad,
        )

    def __repr__(self) -> str:
        preview = list(self.data[:6])
        suffix = "..." if self.numel > 6 else ""
        return f"Tensor(shape={self.shape}, data={preview}{suffix}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("скалярный Tensor не имеет len()")
        return self.shape[0]

    def __getitem__(self, key: int | tuple[int, ...]) -> float | "Tensor":
        keys = (key,) if isinstance(key, int) else key
        if not isinstance(keys, tuple) or not all(isinstance(item, int) for item in keys):
            raise TypeError("поддерживаются только целочисленные индексы")
        if len(keys) > self.ndim:
            raise IndexError("слишком много индексов")
        normalized: list[int] = []
        for axis, item in enumerate(keys):
            dimension = self.shape[axis]
            item = item + dimension if item < 0 else item
            if not 0 <= item < dimension:
                raise IndexError("индекс Tensor вне диапазона")
            normalized.append(item)
        offset = sum(item * self.strides[axis] for axis, item in enumerate(normalized))
        if len(keys) == self.ndim:
            return float(self.data[offset])
        remaining_shape = self.shape[len(keys):]
        count = _product(remaining_shape)
        return Tensor(self.data[offset:offset + count], remaining_shape)

    def item(self) -> float:
        """Возвращает значение одноэлементного Tensor."""
        if self.numel != 1:
            raise ValueError("item() допустим только для одного элемента")
        return float(self.data[0])

    def tolist(self) -> list[float]:
        """Возвращает плоскую Python-копию данных."""
        return list(self.data)

    def clone(self, *, requires_grad: bool | None = None) -> "Tensor":
        """Создаёт независимую копию данных без графа."""
        flag = self.requires_grad if requires_grad is None else requires_grad
        return Tensor(self.tolist(), self.shape, requires_grad=flag)

    def detach(self) -> "Tensor":
        """Создаёт копию, отсоединённую от графа."""
        return Tensor(self.tolist(), self.shape)

    def contiguous(self) -> "Tensor":
        """Возвращает Tensor; текущая реализация всегда непрерывна."""
        return self

    def zero_grad(self) -> None:
        """Удаляет накопленный градиент."""
        self.grad = None

    def _accumulate_grad(
        self, values: Sequence[float], *, take_ownership: bool = False,
    ) -> None:
        if not self.requires_grad:
            return
        if len(values) != self.numel:
            raise ValueError("внутренняя ошибка: неверный размер градиента")
        if self.grad is None:
            if (
                not take_ownership
                and getattr(values, "_mimillm_cuda_array", False)
            ):
                # Autograd consumers may later mutate their accumulated
                # gradient. Keep a distinct CUDA allocation without forcing
                # a device-to-host synchronization merely to break aliasing.
                self.grad = get_backend().scalar_multiply(values, 1.0)
            else:
                self.grad = (
                    values
                    if take_ownership
                    and isinstance(values, array)
                    and values.typecode == "f"
                    else array("f", values)
                )
        else:
            self.grad = get_backend().add(self.grad, values)

    def backward(self, gradient: "Tensor | Sequence[float] | float | None" = None) -> None:
        """Строит топологический порядок и распространяет градиент назад."""
        if not self.requires_grad:
            raise RuntimeError("backward() вызван для Tensor без requires_grad")
        if gradient is None:
            if self.numel != 1:
                raise ValueError("для нескалярного Tensor требуется явный gradient")
            initial: Sequence[float] = [1.0]
        elif isinstance(gradient, Tensor):
            initial = gradient.data
        elif isinstance(gradient, (int, float)):
            initial = [float(gradient)]
        else:
            initial = gradient
        if len(initial) != self.numel:
            raise ValueError("размер начального градиента не совпадает с Tensor")

        order: list[Tensor] = []
        visited: set[int] = set()
        pending: list[tuple[Tensor, bool]] = [(self, False)]
        while pending:
            node, expanded = pending.pop()
            if expanded:
                order.append(node)
                continue
            identity = id(node)
            if identity in visited:
                continue
            visited.add(identity)
            pending.append((node, True))
            pending.extend((parent, False) for parent in reversed(node.parents))
        self._accumulate_grad(initial)
        for node in reversed(order):
            if node._backward_fn is not None and node.grad is not None:
                node._backward_fn()
            if node._backward_fn is not None:
                node._backward_fn = None
                node.parents = ()

    @staticmethod
    def _coerce(value: Any) -> "Tensor":
        if isinstance(value, Tensor):
            return value
        if isinstance(value, (int, float)):
            return Tensor([float(value)], ())
        raise TypeError(f"операция Tensor не поддерживает {type(value).__name__}")

    def _binary(
        self, other: Any, forward: Callable[[float, float], float],
        grad_left: Callable[[float, float], float], grad_right: Callable[[float, float], float],
        operation: str,
    ) -> "Tensor":
        right = self._coerce(other)
        output_shape = _broadcast_shape(self.shape, right.shape)
        output_size = _product(output_shape)
        selected_backend = get_backend()
        same_shape = self.shape == right.shape == output_shape
        right_scalar = right.numel == 1 and self.shape == output_shape
        native_broadcast = bool(
            getattr(selected_backend, "supports_native_broadcast", False)
        )
        fast_row_vector = bool(
            operation == "add"
            and self.shape == output_shape
            and self.ndim >= 2
            and right.ndim == 1
            and self.shape[-1] == right.shape[0]
            and callable(getattr(selected_backend, "add_row_vector", None))
            and callable(getattr(selected_backend, "sum_columns", None))
        )
        row_vector_rows = self.numel // right.numel if fast_row_vector else 0
        left_map: list[int] | None = None
        right_map: list[int] | None = None
        if same_shape and operation == "add":
            values = selected_backend.add(self.data, right.data)
        elif same_shape and operation == "mul":
            values = selected_backend.multiply(self.data, right.data)
        elif right_scalar and operation == "mul":
            values = selected_backend.scalar_multiply(self.data, right.data[0])
        elif right_scalar and operation == "div":
            values = selected_backend.scalar_multiply(self.data, 1.0 / right.data[0])
        elif right_scalar:
            scalar = right.data[0]
            values = array("f", (forward(value, scalar) for value in self.data))
        elif fast_row_vector:
            values = selected_backend.add_row_vector(
                self.data, right.data, row_vector_rows, right.numel,
            )
        elif native_broadcast:
            values = selected_backend.broadcast_binary(
                self.data, right.data, self.shape, right.shape,
                output_shape, operation,
            )
        else:
            left_map = _broadcast_map(self.shape, output_shape)
            right_map = _broadcast_map(right.shape, output_shape)
            values = array("f", (
                forward(self.data[left_map[flat]], right.data[right_map[flat]])
                for flat in range(output_size)
            ))
        requires_grad = self.requires_grad or right.requires_grad
        output = Tensor(values, output_shape, requires_grad=requires_grad, parents=(self, right), operation=operation)

        def backward_fn() -> None:
            assert output.grad is not None
            if same_shape:
                if self.requires_grad:
                    if operation in {"add", "sub"}:
                        self._accumulate_grad(output.grad)
                    elif operation == "mul":
                        self._accumulate_grad(
                            selected_backend.multiply(output.grad, right.data),
                            take_ownership=True,
                        )
                    else:
                        self._accumulate_grad(array("f", (
                            upstream * grad_left(left, right_value)
                            for upstream, left, right_value in zip(
                                output.grad, self.data, right.data
                            )
                        )))
                if right.requires_grad:
                    if operation == "add":
                        right._accumulate_grad(output.grad)
                    elif operation == "sub":
                        right._accumulate_grad(
                            selected_backend.scalar_multiply(output.grad, -1.0),
                            take_ownership=True,
                        )
                    elif operation == "mul":
                        right._accumulate_grad(
                            selected_backend.multiply(output.grad, self.data),
                            take_ownership=True,
                        )
                    else:
                        right._accumulate_grad(array("f", (
                            upstream * grad_right(left, right_value)
                            for upstream, left, right_value in zip(
                                output.grad, self.data, right.data
                            )
                        )))
                return
            if right_scalar:
                scalar = right.data[0]
                if self.requires_grad:
                    if operation in {"add", "sub"}:
                        grad = array("f", output.grad)
                    elif operation == "mul":
                        grad = selected_backend.scalar_multiply(output.grad, scalar)
                    elif operation == "div":
                        grad = selected_backend.scalar_multiply(output.grad, 1.0 / scalar)
                    else:
                        grad = array("f", (
                            upstream * grad_left(left, scalar)
                            for upstream, left in zip(output.grad, self.data)
                        ))
                    self._accumulate_grad(grad)
                if right.requires_grad:
                    right._accumulate_grad([sum(
                        upstream * grad_right(left, scalar)
                        for upstream, left in zip(output.grad, self.data)
                    )])
                return
            if fast_row_vector:
                if self.requires_grad:
                    self._accumulate_grad(output.grad)
                if right.requires_grad:
                    right._accumulate_grad(
                        selected_backend.sum_columns(
                            output.grad, row_vector_rows, right.numel,
                        ),
                        take_ownership=True,
                    )
                return
            if native_broadcast:
                native_grad_left, native_grad_right = selected_backend.broadcast_binary_backward(
                    self.data, right.data, output.grad, self.shape, right.shape,
                    output_shape, operation,
                )
                if self.requires_grad:
                    self._accumulate_grad(native_grad_left, take_ownership=True)
                if right.requires_grad:
                    right._accumulate_grad(native_grad_right, take_ownership=True)
                return
            assert left_map is not None and right_map is not None
            if self.requires_grad:
                grad = array("f", [0.0]) * self.numel
                for i, upstream in enumerate(output.grad):
                    grad[left_map[i]] += upstream * grad_left(self.data[left_map[i]], right.data[right_map[i]])
                self._accumulate_grad(grad)
            if right.requires_grad:
                grad = array("f", [0.0]) * right.numel
                for i, upstream in enumerate(output.grad):
                    grad[right_map[i]] += upstream * grad_right(self.data[left_map[i]], right.data[right_map[i]])
                right._accumulate_grad(grad)

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def __add__(self, other: Any) -> "Tensor":
        return self._binary(other, lambda a, b: a + b, lambda _a, _b: 1.0, lambda _a, _b: 1.0, "add")

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __sub__(self, other: Any) -> "Tensor":
        return self._binary(other, lambda a, b: a - b, lambda _a, _b: 1.0, lambda _a, _b: -1.0, "sub")

    def __rsub__(self, other: Any) -> "Tensor":
        return self._coerce(other) - self

    def __mul__(self, other: Any) -> "Tensor":
        return self._binary(other, lambda a, b: a * b, lambda _a, b: b, lambda a, _b: a, "mul")

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __truediv__(self, other: Any) -> "Tensor":
        return self._binary(
            other, lambda a, b: a / b, lambda _a, b: 1.0 / b,
            lambda a, b: -a / (b * b), "div",
        )

    def __rtruediv__(self, other: Any) -> "Tensor":
        return self._coerce(other) / self

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def _unary(
        self, forward: Callable[[float], float], derivative: Callable[[float, float], float], operation: str,
    ) -> "Tensor":
        output = Tensor(
            (forward(value) for value in self.data), self.shape,
            requires_grad=self.requires_grad, parents=(self,), operation=operation,
        )

        def backward_fn() -> None:
            assert output.grad is not None
            self._accumulate_grad(array("f", (
                output.grad[i] * derivative(self.data[i], output.data[i]) for i in range(self.numel)
            )))

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def exp(self) -> "Tensor":
        """Поэлементная экспонента."""
        return self._unary(math.exp, lambda _x, y: y, "exp")

    def log(self) -> "Tensor":
        """Поэлементный натуральный логарифм."""
        return self._unary(math.log, lambda x, _y: 1.0 / x, "log")

    def sqrt(self) -> "Tensor":
        """Поэлементный квадратный корень."""
        return self._unary(math.sqrt, lambda _x, y: 0.5 / y, "sqrt")

    def relu(self) -> "Tensor":
        """Поэлементная функция max(0, x)."""
        selected_backend = get_backend()
        output = Tensor(
            selected_backend.relu(self.data), self.shape,
            requires_grad=self.requires_grad, parents=(self,), operation="relu",
        )

        def backward_fn() -> None:
            assert output.grad is not None
            self._accumulate_grad(
                selected_backend.relu_backward(self.data, output.grad),
                take_ownership=True,
            )

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def reshape(self, *shape: int | Sequence[int]) -> "Tensor":
        """Меняет форму без изменения порядка элементов."""
        requested: Sequence[int]
        if len(shape) == 1 and not isinstance(shape[0], int):
            requested = shape[0]
        else:
            requested = shape  # type: ignore[assignment]
        normalized = _normalize_shape(requested, self.numel)
        output = Tensor(
            self.data, normalized, requires_grad=self.requires_grad,
            parents=(self,), operation="reshape",
        )

        def backward_fn() -> None:
            assert output.grad is not None
            self._accumulate_grad(output.grad)

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def permute(self, axes: Sequence[int]) -> "Tensor":
        """Переставляет оси и создаёт непрерывную копию."""
        normalized = tuple(axis + self.ndim if axis < 0 else axis for axis in axes)
        if sorted(normalized) != list(range(self.ndim)):
            raise ValueError("axes должны быть перестановкой всех осей")
        output_shape = tuple(self.shape[axis] for axis in normalized)
        inverse = tuple(normalized.index(axis) for axis in range(self.ndim))
        selected_backend = get_backend()
        values = selected_backend.permute(self.data, self.shape, normalized)
        output = Tensor(
            values, output_shape, requires_grad=self.requires_grad,
            parents=(self,), operation="permute",
        )

        def backward_fn() -> None:
            assert output.grad is not None
            self._accumulate_grad(
                selected_backend.permute(output.grad, output_shape, inverse),
                take_ownership=True,
            )

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def transpose(self, dim0: int = -2, dim1: int = -1) -> "Tensor":
        """Меняет местами две оси и создаёт непрерывную копию."""
        dim0 = dim0 + self.ndim if dim0 < 0 else dim0
        dim1 = dim1 + self.ndim if dim1 < 0 else dim1
        if not 0 <= dim0 < self.ndim or not 0 <= dim1 < self.ndim:
            raise ValueError("ось transpose вне диапазона")
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return self.permute(axes)

    @property
    def T(self) -> "Tensor":
        """Транспонирует последние две оси."""
        if self.ndim < 2:
            raise ValueError("T требует минимум две оси")
        return self.transpose(-2, -1)

    def sum(self, axis: int | None = None, *, keepdims: bool = False) -> "Tensor":
        """Суммирует все элементы либо одну указанную ось."""
        fast_last_axis = False
        rows = columns = 0
        selected_backend = get_backend()
        if axis is None:
            output_shape = (1,) * self.ndim if keepdims else ()
            output = Tensor([sum(self.data)], output_shape, requires_grad=self.requires_grad, parents=(self,), operation="sum")
            groups = [0] * self.numel
        else:
            axis = axis + self.ndim if axis < 0 else axis
            if not 0 <= axis < self.ndim:
                raise ValueError("ось sum вне диапазона")
            output_shape_list = list(self.shape)
            if keepdims:
                output_shape_list[axis] = 1
            else:
                output_shape_list.pop(axis)
            output_shape = tuple(output_shape_list)
            if axis == self.ndim - 1:
                fast_last_axis = True
                columns = self.shape[-1]
                rows = self.numel // columns
                values = selected_backend.sum_rows(self.data, rows, columns)
                groups = []
            else:
                values = array("f", [0.0]) * _product(output_shape)
                groups = []
                output_strides = _contiguous_strides(output_shape)
                for flat, value in enumerate(self.data):
                    coords = _unravel(flat, self.shape)
                    if keepdims:
                        coords[axis] = 0
                    else:
                        coords.pop(axis)
                    group = _ravel(coords, output_strides)
                    groups.append(group)
                    values[group] += value
            output = Tensor(values, output_shape, requires_grad=self.requires_grad, parents=(self,), operation="sum")

        def backward_fn() -> None:
            assert output.grad is not None
            if fast_last_axis:
                self._accumulate_grad(
                    selected_backend.sum_rows_backward(output.grad, rows, columns),
                    take_ownership=True,
                )
            else:
                self._accumulate_grad(array("f", (output.grad[group] for group in groups)))

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def mean(self, axis: int | None = None, *, keepdims: bool = False) -> "Tensor":
        """Вычисляет среднее всех элементов либо одной оси."""
        divisor = self.numel if axis is None else self.shape[axis % self.ndim]
        if divisor == 0:
            raise ValueError("mean пустого Tensor не определён")
        return self.sum(axis, keepdims=keepdims) / float(divisor)

    def matmul(self, right: "Tensor") -> "Tensor":
        """Умножает 2D или одинаково батчированные матрицы по последним осям."""
        if not isinstance(right, Tensor):
            raise TypeError("matmul ожидает Tensor")
        if self.ndim < 2 or right.ndim < 2:
            raise ValueError("matmul требует Tensor минимум с двумя осями")
        if self.shape[-1] != right.shape[-2]:
            raise ValueError(f"matmul: внутренние размеры не совпадают: {self.shape} и {right.shape}")
        left_leading, right_leading = self.shape[:-2], right.shape[:-2]
        if left_leading != right_leading:
            raise ValueError("matmul первой версии требует одинаковые batch-размеры")
        batches = _product(left_leading) if left_leading else 1
        rows, inner, columns = self.shape[-2], self.shape[-1], right.shape[-1]
        if batches == 1:
            values = get_backend().matmul(self.data, right.data, rows, inner, columns)
        else:
            values = get_backend().batched_matmul(
                self.data, right.data, batches, rows, inner, columns
            )
        output_shape = (*left_leading, rows, columns)
        requires_grad = self.requires_grad or right.requires_grad
        output = Tensor(values, output_shape, requires_grad=requires_grad, parents=(self, right), operation="matmul")

        def backward_fn() -> None:
            assert output.grad is not None
            selected_backend = get_backend()
            if self.requires_grad:
                native = getattr(
                    selected_backend, "matmul_backward_left", None,
                )
                if callable(native):
                    grad_left = native(
                        output.grad, right.data,
                        batches, rows, inner, columns,
                    )
                else:
                    right_axes = (
                        *range(right.ndim - 2), right.ndim - 1, right.ndim - 2,
                    )
                    right_transposed = selected_backend.permute(
                        right.data, right.shape, right_axes
                    )
                    if batches == 1:
                        grad_left = selected_backend.matmul(
                            output.grad, right_transposed, rows, columns, inner
                        )
                    else:
                        grad_left = selected_backend.batched_matmul(
                            output.grad, right_transposed,
                            batches, rows, columns, inner,
                        )
                self._accumulate_grad(grad_left, take_ownership=True)
            if right.requires_grad:
                native = getattr(
                    selected_backend, "matmul_backward_right", None,
                )
                if callable(native):
                    grad_right = native(
                        self.data, output.grad,
                        batches, rows, inner, columns,
                    )
                else:
                    left_axes = (
                        *range(self.ndim - 2), self.ndim - 1, self.ndim - 2,
                    )
                    left_transposed = selected_backend.permute(
                        self.data, self.shape, left_axes
                    )
                    if batches == 1:
                        grad_right = selected_backend.matmul(
                            left_transposed, output.grad, inner, rows, columns
                        )
                    else:
                        grad_right = selected_backend.batched_matmul(
                            left_transposed, output.grad,
                            batches, inner, rows, columns,
                        )
                right._accumulate_grad(grad_right, take_ownership=True)

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def __matmul__(self, right: "Tensor") -> "Tensor":
        return self.matmul(right)

    def batched_matmul(self, right: "Tensor") -> "Tensor":
        """Явный псевдоним matmul для батчированных матриц."""
        return self.matmul(right)

    def softmax(self, axis: int = -1) -> "Tensor":
        """Вычисляет стабильный softmax по последней оси."""
        axis = axis + self.ndim if axis < 0 else axis
        if axis != self.ndim - 1:
            raise ValueError("softmax первой версии поддерживает только последнюю ось")
        columns = self.shape[-1]
        rows = self.numel // columns
        selected_backend = get_backend()
        values = selected_backend.softmax_rows(self.data, rows, columns)
        output = Tensor(values, self.shape, requires_grad=self.requires_grad, parents=(self,), operation="softmax")

        def backward_fn() -> None:
            assert output.grad is not None
            self._accumulate_grad(
                selected_backend.softmax_backward(
                    output.data, output.grad, rows, columns
                ),
                take_ownership=True,
            )

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def causal_softmax(
        self, *, scale: float = 1.0, sequence_length: int | None = None,
    ) -> "Tensor":
        """Combine attention scaling, causal masking, and row softmax."""
        if self.ndim < 2 or self.shape[-2] != self.shape[-1]:
            raise ValueError("causal_softmax expects square matrices on the last two axes")
        columns = self.shape[-1]
        sequence = columns if sequence_length is None else int(sequence_length)
        rows = self.numel // columns
        if sequence != columns or rows % sequence:
            raise ValueError("sequence_length does not match the causal attention shape")
        selected_backend = get_backend()
        forward = getattr(selected_backend, "causal_softmax_rows", None)
        backward = getattr(selected_backend, "causal_softmax_backward", None)
        if not callable(forward) or not callable(backward):
            mask = Tensor(
                (
                    0.0 if column <= row % sequence else -1.0e9
                    for row in range(rows)
                    for column in range(columns)
                ),
                self.shape,
            )
            return (self * float(scale) + mask).softmax(axis=-1)
        values = forward(
            self.data, rows, columns, sequence, float(scale),
        )
        output = Tensor(
            values, self.shape, requires_grad=self.requires_grad,
            parents=(self,), operation="causal_softmax",
        )

        def backward_fn() -> None:
            assert output.grad is not None
            self._accumulate_grad(
                backward(
                    output.data, output.grad, rows, columns, float(scale),
                ),
                take_ownership=True,
            )

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def rms_norm(self, weight: "Tensor", *, epsilon: float = 1e-5) -> "Tensor":
        """Normalize the last axis and apply a learned per-feature scale."""
        if not isinstance(weight, Tensor):
            raise TypeError("rms_norm expects a Tensor weight")
        if (
            self.ndim < 1
            or weight.ndim != 1
            or self.shape[-1] != weight.shape[0]
            or epsilon <= 0.0
        ):
            raise ValueError("RMSNorm shape or epsilon is invalid")
        selected_backend = get_backend()
        forward = getattr(selected_backend, "rms_norm", None)
        backward = getattr(selected_backend, "rms_norm_backward", None)
        if not callable(forward) or not callable(backward):
            inverse_rms = (
                (self * self).mean(axis=-1, keepdims=True) + float(epsilon)
            ).sqrt()
            return (self / inverse_rms) * weight
        columns = self.shape[-1]
        rows = self.numel // columns
        values = forward(
            self.data, weight.data, rows, columns, float(epsilon),
        )
        requires_grad = self.requires_grad or weight.requires_grad
        output = Tensor(
            values, self.shape, requires_grad=requires_grad,
            parents=(self, weight), operation="rms_norm",
        )

        def backward_fn() -> None:
            assert output.grad is not None
            grad_input, grad_weight = backward(
                self.data, weight.data, output.grad,
                rows, columns, float(epsilon),
            )
            if self.requires_grad:
                self._accumulate_grad(grad_input, take_ownership=True)
            if weight.requires_grad:
                weight._accumulate_grad(grad_weight, take_ownership=True)

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def embedding(self, indices: Sequence[int]) -> "Tensor":
        """Выбирает строки таблицы формы (vocab, embedding)."""
        if self.ndim != 2:
            raise ValueError("embedding ожидает таблицу с двумя осями")
        rows, width = self.shape
        checked = [int(index) for index in indices]
        if any(index < 0 or index >= rows for index in checked):
            raise IndexError("индекс embedding вне словаря")
        selected_backend = get_backend()
        values = selected_backend.embedding_gather(self.data, checked, rows, width)
        output = Tensor(
            values, (len(checked), width), requires_grad=self.requires_grad,
            parents=(self,), operation="embedding",
        )

        def backward_fn() -> None:
            assert output.grad is not None
            grad = selected_backend.embedding_scatter_add(
                checked, output.grad, rows, width
            )
            self._accumulate_grad(grad, take_ownership=True)

        output._backward_fn = backward_fn if output.requires_grad else None
        return output

    def cross_entropy(
        self, targets: Sequence[int], *, weights: Sequence[float] | None = None,
    ) -> "Tensor":
        """Средняя cross-entropy для logits формы (..., classes).

        ``weights`` задаёт вес каждой строки logits. Нулевой вес исключает
        позицию из loss и backward; это используется для padding и prompt
        части обучающих пар вопрос–ответ.
        """
        if self.ndim < 2:
            raise ValueError("cross_entropy ожидает logits минимум с двумя осями")
        classes = self.shape[-1]
        rows = self.numel // classes
        checked = [int(target) for target in targets]
        if len(checked) != rows:
            raise ValueError(f"ожидалось {rows} target, получено {len(checked)}")
        if any(target < 0 or target >= classes for target in checked):
            raise IndexError("target вне диапазона классов")
        checked_weights: list[float] | None = None
        weight_sum = float(rows)
        if weights is not None:
            checked_weights = [float(weight) for weight in weights]
            if len(checked_weights) != rows:
                raise ValueError(f"ожидалось {rows} weights, получено {len(checked_weights)}")
            if any(not math.isfinite(weight) or weight < 0.0 for weight in checked_weights):
                raise ValueError("weights должны быть конечными и неотрицательными")
            weight_sum = sum(checked_weights)
            if weight_sum <= 0.0:
                raise ValueError("хотя бы один weight должен быть положительным")
        selected_backend = get_backend()
        native_weighted = checked_weights is not None and hasattr(
            selected_backend, "weighted_cross_entropy"
        )
        base_gradient: array | None = None
        if native_weighted:
            loss, native_gradient = selected_backend.weighted_cross_entropy(
                self.data, checked, checked_weights, rows, classes,
                compute_gradient=self.requires_grad,
            )
            base_gradient = native_gradient if self.requires_grad else None
        elif checked_weights is None:
            loss = selected_backend.cross_entropy(self.data, checked, rows, classes)
        else:
            loss_sum = 0.0
            for row, (target, weight) in enumerate(zip(checked, checked_weights)):
                if weight == 0.0:
                    continue
                offset = row * classes
                maximum = max(self.data[offset:offset + classes])
                exponential_sum = sum(
                    math.exp(float(value) - maximum)
                    for value in self.data[offset:offset + classes]
                )
                loss_sum += weight * (
                    maximum + math.log(exponential_sum) - self.data[offset + target]
                )
            loss = loss_sum / weight_sum
        if base_gradient is None and self.requires_grad:
            base_gradient = selected_backend.cross_entropy_backward(
                self.data, checked, rows, classes
            )
        if base_gradient is not None and checked_weights is not None and not native_weighted:
            for row, weight in enumerate(checked_weights):
                scale = weight * rows / weight_sum
                offset = row * classes
                for column in range(classes):
                    base_gradient[offset + column] *= scale
        output = Tensor([loss], (), requires_grad=self.requires_grad, parents=(self,), operation="cross_entropy")

        def backward_fn() -> None:
            assert output.grad is not None
            assert base_gradient is not None
            scale = output.grad[0]
            grad = (
                base_gradient
                if scale == 1.0
                else selected_backend.scalar_multiply(base_gradient, scale)
            )
            self._accumulate_grad(grad, take_ownership=True)

        output._backward_fn = backward_fn if output.requires_grad else None
        return output
