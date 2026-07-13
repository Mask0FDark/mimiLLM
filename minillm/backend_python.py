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


def embedding_gather(
    table: Sequence[float], indices: Sequence[int], vocab: int, width: int,
) -> array:
    """Собирает строки таблицы embedding."""
    if len(table) != vocab * width or vocab <= 0 or width <= 0:
        raise ValueError("embedding_gather: неверная форма таблицы")
    output = array("f")
    for index in indices:
        if index < 0 or index >= vocab:
            raise IndexError("embedding index out of range")
        output.extend(table[index * width:(index + 1) * width])
    return output


def embedding_scatter_add(
    indices: Sequence[int], grad_output: Sequence[float], vocab: int, width: int,
) -> array:
    """Суммирует градиенты повторяющихся embedding-индексов."""
    if len(grad_output) != len(indices) * width:
        raise ValueError("embedding_scatter_add: неверная форма градиента")
    output = array("f", [0.0]) * (vocab * width)
    for row, index in enumerate(indices):
        if index < 0 or index >= vocab:
            raise IndexError("embedding index out of range")
        for column in range(width):
            output[index * width + column] += grad_output[row * width + column]
    return output


def cross_entropy(
    logits: Sequence[float], targets: Sequence[int], rows: int, classes: int,
) -> float:
    """Вычисляет среднюю стабильную cross-entropy."""
    probabilities = softmax_rows(logits, rows, classes)
    if len(targets) != rows:
        raise ValueError("cross_entropy: неверное число targets")
    return -sum(
        math.log(max(float(probabilities[row * classes + target]), 1e-30))
        for row, target in enumerate(targets)
    ) / rows


def cross_entropy_backward(
    logits: Sequence[float], targets: Sequence[int], rows: int, classes: int,
) -> array:
    """Возвращает d(mean cross-entropy)/d(logits)."""
    output = softmax_rows(logits, rows, classes)
    for row, target in enumerate(targets):
        output[row * classes + target] -= 1.0
    scale = 1.0 / rows
    for index in range(len(output)):
        output[index] *= scale
    return output
