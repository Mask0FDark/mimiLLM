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
        from .backend import get_backend

        selected_backend = get_backend()
        gradients = [
            parameter.grad
            for parameter in self.parameters
            if parameter.grad is not None
        ]
        if hasattr(selected_backend, "global_sum_squares"):
            squared = selected_backend.global_sum_squares(gradients)
        else:
            native_reduction = hasattr(selected_backend, "sum_squares")
            squared = 0.0
            for gradient in gradients:
                squared += (
                    selected_backend.sum_squares(gradient)
                    if native_reduction
                    else sum(float(value) * value for value in gradient)
                )
        norm = math.sqrt(squared)
        if norm > max_norm:
            scale = max_norm / (norm + 1e-12)
            for parameter in self.parameters:
                if parameter.grad is not None:
                    if hasattr(selected_backend, "scale_inplace"):
                        selected_backend.scale_inplace(parameter.grad, scale)
                    else:
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


class AdamW(Optimizer):
    """Adam с bias correction и отделённым L2 decay параметров."""

    def __init__(
        self, parameters: Iterable[Parameter], learning_rate: float = 3e-4, *,
        beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(parameters, learning_rate)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 и beta2 должны быть в диапазоне [0, 1)")
        if epsilon <= 0.0 or weight_decay < 0.0:
            raise ValueError("epsilon > 0, weight_decay >= 0")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self.first_moments = [array("f", [0.0]) * parameter.numel for parameter in self.parameters]
        self.second_moments = [array("f", [0.0]) * parameter.numel for parameter in self.parameters]
        self._register_native_state()

    def _register_native_state(self) -> None:
        from .backend import get_backend

        selected_backend = get_backend()
        if hasattr(selected_backend, "prepare_optimizer_state"):
            parameters, first, second = selected_backend.prepare_optimizer_state(
                [parameter.data for parameter in self.parameters],
                self.first_moments,
                self.second_moments,
            )
            for parameter, storage in zip(self.parameters, parameters):
                parameter.data = storage
            self.first_moments = first
            self.second_moments = second
        elif hasattr(selected_backend, "register_optimizer_state"):
            selected_backend.register_optimizer_state(
                [parameter.data for parameter in self.parameters],
                self.first_moments,
                self.second_moments,
            )

    def step(self) -> None:
        self.step_count += 1
        from .backend import get_backend

        selected_backend = get_backend()
        if hasattr(selected_backend, "adamw_update"):
            for parameter, first, second in zip(
                self.parameters, self.first_moments, self.second_moments
            ):
                if parameter.grad is not None:
                    selected_backend.adamw_update(
                        parameter.data, parameter.grad, first, second,
                        learning_rate=self.learning_rate, beta1=self.beta1,
                        beta2=self.beta2, epsilon=self.epsilon,
                        weight_decay=self.weight_decay, step=self.step_count,
                    )
            return
        correction1 = 1.0 - self.beta1 ** self.step_count
        correction2 = 1.0 - self.beta2 ** self.step_count
        for parameter, first, second in zip(
            self.parameters, self.first_moments, self.second_moments
        ):
            if parameter.grad is None:
                continue
            for index, gradient in enumerate(parameter.grad):
                first[index] = self.beta1 * first[index] + (1.0 - self.beta1) * gradient
                second[index] = self.beta2 * second[index] + (1.0 - self.beta2) * gradient * gradient
                first_hat = first[index] / correction1
                second_hat = second[index] / correction2
                update = first_hat / (math.sqrt(second_hat) + self.epsilon)
                parameter.data[index] -= self.learning_rate * (
                    update + self.weight_decay * parameter.data[index]
                )

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "AdamW", "learning_rate": self.learning_rate,
            "beta1": self.beta1, "beta2": self.beta2, "epsilon": self.epsilon,
            "weight_decay": self.weight_decay, "step_count": self.step_count,
            "first_moments": [array("f", values) for values in self.first_moments],
            "second_moments": [array("f", values) for values in self.second_moments],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Строго восстанавливает гиперпараметры и буферы моментов."""
        if state.get("type") not in (None, "AdamW"):
            raise ValueError("checkpoint содержит состояние другого оптимизатора")
        first = state.get("first_moments")
        second = state.get("second_moments")
        if not isinstance(first, list) or not isinstance(second, list):
            raise ValueError("состояние AdamW не содержит moments")
        expected = [parameter.numel for parameter in self.parameters]
        if [len(values) for values in first] != expected or [len(values) for values in second] != expected:
            raise ValueError("размеры moments AdamW не совпадают с параметрами")
        self.learning_rate = float(state["learning_rate"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.epsilon = float(state["epsilon"])
        self.weight_decay = float(state["weight_decay"])
        self.step_count = int(state["step_count"])
        self.first_moments = [array("f", values) for values in first]
        self.second_moments = [array("f", values) for values in second]
        self._register_native_state()
