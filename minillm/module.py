"""Минимальная иерархия модулей и параметров."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .parameter import Parameter
from .tensor import Tensor


class Module:
    """Базовый класс слоя с рекурсивной регистрацией атрибутов."""

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in {"_parameters", "_modules", "training"}:
            if isinstance(value, Parameter):
                self._parameters[name] = value
                self._modules.pop(name, None)
            elif isinstance(value, Module):
                self._modules[name] = value
                self._parameters.pop(name, None)
            elif hasattr(self, "_parameters"):
                self._parameters.pop(name, None)
                self._modules.pop(name, None)
        object.__setattr__(self, name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Вычисляет выход; конкретный модуль обязан переопределить метод."""
        raise NotImplementedError(f"{type(self).__name__}.forward не реализован")

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Обходит параметры в стабильном порядке с полными именами."""
        for name, parameter in self._parameters.items():
            yield f"{prefix}{name}", parameter
        for name, module in self._modules.items():
            child_prefix = f"{prefix}{name}."
            yield from module.named_parameters(child_prefix)

    def parameters(self) -> list[Parameter]:
        """Возвращает список всех обучаемых параметров."""
        return [parameter for _, parameter in self.named_parameters()]

    def zero_grad(self) -> None:
        """Удаляет градиенты всех параметров."""
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        """Рекурсивно переключает учебный режим."""
        self.training = bool(mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """Переключает модуль в режим оценки."""
        return self.train(False)

    def state_dict(self) -> dict[str, Tensor]:
        """Создаёт отсоединённые копии параметров по именам."""
        return {name: parameter.detach() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Строго загружает параметры с проверкой имён и форм."""
        current = dict(self.named_parameters())
        missing = sorted(set(current) - set(state))
        unexpected = sorted(set(state) - set(current))
        if missing or unexpected:
            raise ValueError(f"несовпадение state_dict: missing={missing}, unexpected={unexpected}")
        for name, parameter in current.items():
            source = state[name]
            if parameter.shape != source.shape:
                raise ValueError(f"параметр {name}: ожидалась форма {parameter.shape}, получена {source.shape}")
            parameter.data[:] = source.data

