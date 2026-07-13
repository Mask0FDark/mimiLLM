"""Самостоятельные CPU-оптимизаторы."""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterable

from .parameter import Parameter


class Optimizer:
    """Общие операции над фиксированным списком параметров."""

    def __init__(self, parameters: Iterable[Parameter], learning_rate: float) -> None:
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("оптимизатору нужен хотя бы один Parameter")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate должен быть положительным")
        self.learning_rate = float(learning_rate)

    def zero_grad(self) -> None:
        """Удаляет накопленные градиенты."""
        for parameter in self.parameters:
            parameter.zero_grad()

    def clip_grad_norm(self, max_norm: float) -> float:
        """Ограничивает общую L2-норму и возвращает исходное значение."""
        if max_norm <= 0.0:
            raise ValueError("max_norm должен быть положительным")
        squared = 0.0
        for parameter in self.parameters:
            if parameter.grad is not None:
                squared += sum(float(value) * value for value in parameter.grad)
        norm = math.sqrt(squared)
        if norm > max_norm:
            scale = max_norm / (norm + 1e-12)
            for parameter in self.parameters:
                if parameter.grad is not None:
                    for index in range(parameter.numel):
                        parameter.grad[index] *= scale
        return norm

    def step(self) -> None:
        """Обновляет параметры."""
        raise NotImplementedError

    def state_dict(self) -> dict[str, object]:
        """Возвращает сериализуемое состояние."""
        return {"learning_rate": self.learning_rate}


class SGD(Optimizer):
    """Стохастический градиентный спуск без momentum."""

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            for index, gradient in enumerate(parameter.grad):
                parameter.data[index] -= self.learning_rate * gradient

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Восстанавливает learning rate."""
        self.learning_rate = float(state["learning_rate"])

