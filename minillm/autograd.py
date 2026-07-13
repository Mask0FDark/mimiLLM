"""Вспомогательные средства проверки самописного autograd."""

from __future__ import annotations

from collections.abc import Callable

from .tensor import Tensor


def numerical_gradient(
    function: Callable[[], Tensor], tensor: Tensor, *, epsilon: float = 1e-3
) -> list[float]:
    """Вычисляет центральную конечную разность для каждого элемента Tensor."""
    if epsilon <= 0.0:
        raise ValueError("epsilon должен быть положительным")
    result: list[float] = []
    for index in range(tensor.numel):
        original = tensor.data[index]
        tensor.data[index] = original + epsilon
        positive = function().item()
        tensor.data[index] = original - epsilon
        negative = function().item()
        tensor.data[index] = original
        result.append((positive - negative) / (2.0 * epsilon))
    return result


def gradcheck(
    function: Callable[[], Tensor], tensors: list[Tensor], *,
    epsilon: float = 1e-3, tolerance: float = 3e-3,
) -> tuple[bool, float]:
    """Сравнивает autograd с конечными разностями и возвращает максимум ошибки."""
    for tensor in tensors:
        tensor.zero_grad()
    value = function()
    if value.numel != 1:
        raise ValueError("gradcheck требует скалярный результат")
    value.backward()
    analytical = [list(tensor.grad or []) for tensor in tensors]
    maximum_error = 0.0
    for tensor_index, tensor in enumerate(tensors):
        numeric = numerical_gradient(function, tensor, epsilon=epsilon)
        if len(analytical[tensor_index]) != tensor.numel:
            raise RuntimeError("autograd не создал градиент проверяемого Tensor")
        for actual, expected in zip(analytical[tensor_index], numeric):
            maximum_error = max(maximum_error, abs(actual - expected))
    return maximum_error <= tolerance, maximum_error

