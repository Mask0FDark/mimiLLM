"""Медленные эталонные float32-ядра на стандартной библиотеке Python."""

from __future__ import annotations

import math
from array import array
from collections.abc import Sequence


def add(left: Sequence[float], right: Sequence[float]) -> array:
    """Складывает одинаковые по длине непрерывные буферы."""
    if len(left) != len(right):
        raise ValueError("add: длины буферов не совпадают")
    return array("f", (left[i] + right[i] for i in range(len(left))))


def multiply(left: Sequence[float], right: Sequence[float]) -> array:
    """Перемножает одинаковые по длине непрерывные буферы."""
    if len(left) != len(right):
        raise ValueError("multiply: длины буферов не совпадают")
    return array("f", (left[i] * right[i] for i in range(len(left))))


def scalar_multiply(values: Sequence[float], scalar: float) -> array:
    """Умножает буфер на число."""
    return array("f", (value * scalar for value in values))


def matmul(
    left: Sequence[float], right: Sequence[float], rows: int, inner: int, columns: int
) -> array:
    """Выполняет матричное умножение плотно упакованных матриц."""
    if rows < 0 or inner <= 0 or columns < 0:
        raise ValueError("matmul: размеры должны быть неотрицательными, inner > 0")
    if len(left) != rows * inner or len(right) != inner * columns:
        raise ValueError("matmul: размер буфера не соответствует форме")
    output = array("f", [0.0]) * (rows * columns)
    for row in range(rows):
        left_offset = row * inner
        out_offset = row * columns
        for k in range(inner):
            value = left[left_offset + k]
            right_offset = k * columns
            for column in range(columns):
                output[out_offset + column] += value * right[right_offset + column]
    return output


def batched_matmul(
    left: Sequence[float], right: Sequence[float], batches: int,
    rows: int, inner: int, columns: int,
) -> array:
    """Выполняет независимое matmul для каждой пары матриц батча."""
    if batches <= 0:
        raise ValueError("batched_matmul: batches должен быть положительным")
    if len(left) != batches * rows * inner or len(right) != batches * inner * columns:
        raise ValueError("batched_matmul: размер буфера не соответствует форме")
    output = array("f")
    left_size, right_size = rows * inner, inner * columns
    for batch in range(batches):
        output.extend(matmul(
            left[batch * left_size:(batch + 1) * left_size],
            right[batch * right_size:(batch + 1) * right_size],
            rows, inner, columns,
        ))
    return output


def softmax_rows(values: Sequence[float], rows: int, columns: int) -> array:
    """Стабильно вычисляет softmax для каждой строки."""
    if rows < 0 or columns <= 0 or len(values) != rows * columns:
        raise ValueError("softmax: размер буфера не соответствует форме")
    output = array("f", [0.0]) * len(values)
    for row in range(rows):
        offset = row * columns
        maximum = max(values[offset:offset + columns])
        denominator = 0.0
        for column in range(columns):
            current = math.exp(values[offset + column] - maximum)
            output[offset + column] = current
            denominator += current
        for column in range(columns):
            output[offset + column] /= denominator
    return output


def relu(values: Sequence[float]) -> array:
    """Применяет ReLU к буферу."""
    return array("f", (value if value > 0.0 else 0.0 for value in values))


def relu_backward(values: Sequence[float], grad_output: Sequence[float]) -> array:
    """Распространяет градиент через ReLU."""
    if len(values) != len(grad_output):
        raise ValueError("relu_backward: длины буферов не совпадают")
    return array("f", (
        grad_output[i] if values[i] > 0.0 else 0.0 for i in range(len(values))
    ))

