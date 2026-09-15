"""Самостоятельные CPU-оптимизаторы."""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterable, Sequence

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
            if hasattr(selected_backend, "scale_tensors_inplace"):
                selected_backend.scale_tensors_inplace(gradients, scale)
            else:
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
    """AdamW с накоплением градиентов и выборочным weight decay."""

    def __init__(
        self, parameters: Iterable[Parameter], learning_rate: float = 3e-4, *,
        beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int | None = None,
        decay_mask: Sequence[bool] | None = None,
    ) -> None:
        parameter_list = list(parameters)
        inferred_steps = {
            int(value)
            for parameter in parameter_list
            if (value := getattr(
                parameter, "_mimillm_gradient_accumulation_steps", None
            )) is not None
        }
        if gradient_accumulation_steps is None:
            if len(inferred_steps) > 1:
                raise ValueError(
                    "parameters содержат несовместимые gradient accumulation hints"
                )
            gradient_accumulation_steps = (
                next(iter(inferred_steps)) if inferred_steps else 1
            )
        inferred_decay_mask = [
            getattr(parameter, "_mimillm_weight_decay_enabled", None)
            for parameter in parameter_list
        ]
        if decay_mask is None and any(
            value is not None for value in inferred_decay_mask
        ):
            if any(value is None for value in inferred_decay_mask):
                raise ValueError(
                    "parameters содержат неполный набор weight decay hints"
                )
            decay_mask = [bool(value) for value in inferred_decay_mask]
        super().__init__(parameter_list, learning_rate)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 и beta2 должны быть в диапазоне [0, 1)")
        if epsilon <= 0.0 or weight_decay < 0.0:
            raise ValueError("epsilon > 0, weight_decay >= 0")
        if (
            not isinstance(gradient_accumulation_steps, int)
            or isinstance(gradient_accumulation_steps, bool)
            or gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps должен быть положительным целым числом")
        if decay_mask is None:
            checked_decay_mask = [True] * len(self.parameters)
        else:
            checked_decay_mask = list(decay_mask)
            if len(checked_decay_mask) != len(self.parameters):
                raise ValueError("decay_mask должен содержать по одному значению на Parameter")
            if any(type(value) is not bool for value in checked_decay_mask):
                raise TypeError("decay_mask должен содержать только bool")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.decay_mask = checked_decay_mask
        self.step_count = 0
        self.accumulation_count = 0
        self.first_moments = [array("f", [0.0]) * parameter.numel for parameter in self.parameters]
        self.second_moments = [array("f", [0.0]) * parameter.numel for parameter in self.parameters]
        self.gradient_accumulators: list[array | None] = [None] * len(self.parameters)
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

    def _register_accumulation_state(self) -> None:
        """Возвращает сохранённые накопленные градиенты в активный backend."""
        if self.accumulation_count <= 0:
            self.gradient_accumulators = [None] * len(self.parameters)
            return
        from .backend import get_backend

        selected_backend = get_backend()
        restored: list[array | None] = []
        for values in self.gradient_accumulators:
            if values is None:
                restored.append(None)
            elif hasattr(selected_backend, "scalar_multiply"):
                restored.append(selected_backend.scalar_multiply(values, 1.0))
            else:
                restored.append(array("f", values))
        self.gradient_accumulators = restored

    def _current_active(
        self,
    ) -> list[tuple[int, Parameter, Sequence[float], array, array]]:
        return [
            (index, parameter, parameter.grad, first, second)
            for index, (parameter, first, second) in enumerate(zip(
                self.parameters, self.first_moments, self.second_moments
            ))
            if parameter.grad is not None
        ]

    def _accumulate_current_gradients(self) -> bool:
        """Копит microbatch-градиенты и сообщает, готов ли optimizer update."""
        if self.gradient_accumulation_steps == 1:
            return bool(self._current_active())

        active = self._current_active()
        if not active:
            return False
        from .backend import get_backend

        selected_backend = get_backend()
        for index, _, gradient, _, _ in active:
            previous = self.gradient_accumulators[index]
            if previous is None:
                if hasattr(selected_backend, "scalar_multiply"):
                    accumulated = selected_backend.scalar_multiply(gradient, 1.0)
                else:
                    accumulated = array("f", gradient)
            elif hasattr(selected_backend, "add"):
                accumulated = selected_backend.add(previous, gradient)
            else:
                accumulated = array(
                    "f", (left + right for left, right in zip(previous, gradient))
                )
            self.gradient_accumulators[index] = accumulated
        self.accumulation_count += 1
        return self.accumulation_count >= self.gradient_accumulation_steps

    def _activate_accumulated_gradients(self) -> None:
        """Подменяет leaf-gradients средним градиентом накопленного окна."""
        if self.gradient_accumulation_steps == 1:
            return
        if self.accumulation_count != self.gradient_accumulation_steps:
            raise RuntimeError("накоплен неполный gradient accumulation window")
        scale = 1.0 / self.accumulation_count
        from .backend import get_backend

        selected_backend = get_backend()
        active_accumulators = [
            values for values in self.gradient_accumulators if values is not None
        ]
        if hasattr(selected_backend, "scale_tensors_inplace"):
            selected_backend.scale_tensors_inplace(active_accumulators, scale)
        elif hasattr(selected_backend, "scale_inplace"):
            for values in active_accumulators:
                selected_backend.scale_inplace(values, scale)
        else:
            for values in active_accumulators:
                for index in range(len(values)):
                    values[index] *= scale
        for parameter, values in zip(self.parameters, self.gradient_accumulators):
            parameter.grad = values
        self.gradient_accumulators = [None] * len(self.parameters)
        self.accumulation_count = 0

    def _active_update_groups(
        self,
    ) -> list[tuple[bool, list[tuple[Parameter, Sequence[float], array, array]]]]:
        groups: list[tuple[bool, list[tuple[Parameter, Sequence[float], array, array]]]] = []
        for decay_enabled in (True, False):
            values = [
                (parameter, parameter.grad, first, second)
                for parameter, first, second, use_decay in zip(
                    self.parameters,
                    self.first_moments,
                    self.second_moments,
                    self.decay_mask,
                )
                if parameter.grad is not None and use_decay is decay_enabled
            ]
            if values:
                groups.append((decay_enabled, values))
        return groups

    def _apply_update(self) -> None:
        """Применяет один AdamW update к уже подготовленным градиентам."""
        groups = self._active_update_groups()
        if not groups:
            return
        self.step_count += 1
        from .backend import get_backend

        selected_backend = get_backend()
        if hasattr(selected_backend, "adamw_update_many"):
            for decay_enabled, active in groups:
                selected_backend.adamw_update_many(
                    [item[0].data for item in active],
                    [item[1] for item in active],
                    [item[2] for item in active],
                    [item[3] for item in active],
                    learning_rate=self.learning_rate,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon,
                    weight_decay=self.weight_decay if decay_enabled else 0.0,
                    step=self.step_count,
                )
            return
        if hasattr(selected_backend, "adamw_update"):
            for decay_enabled, active in groups:
                current_decay = self.weight_decay if decay_enabled else 0.0
                for parameter, gradient, first, second in active:
                    selected_backend.adamw_update(
                        parameter.data, gradient, first, second,
                        learning_rate=self.learning_rate, beta1=self.beta1,
                        beta2=self.beta2, epsilon=self.epsilon,
                        weight_decay=current_decay, step=self.step_count,
                    )
            return
        correction1 = 1.0 - self.beta1 ** self.step_count
        correction2 = 1.0 - self.beta2 ** self.step_count
        for decay_enabled, active in groups:
            current_decay = self.weight_decay if decay_enabled else 0.0
            for parameter, gradient, first, second in active:
                for index, gradient_value in enumerate(gradient):
                    first[index] = (
                        self.beta1 * first[index]
                        + (1.0 - self.beta1) * gradient_value
                    )
                    second[index] = (
                        self.beta2 * second[index]
                        + (1.0 - self.beta2) * gradient_value * gradient_value
                    )
                    first_hat = first[index] / correction1
                    second_hat = second[index] / correction2
                    update = first_hat / (math.sqrt(second_hat) + self.epsilon)
                    parameter.data[index] -= self.learning_rate * (
                        update + current_decay * parameter.data[index]
                    )

    def step(self) -> None:
        if not self._accumulate_current_gradients():
            return
        self._activate_accumulated_gradients()
        self._apply_update()

    def step_clipped(self, max_norm: float) -> float:
        """Копит microbatch-градиенты и clip/update выполняет только на границе окна."""
        if max_norm <= 0.0:
            raise ValueError("max_norm must be positive")
        if not self._accumulate_current_gradients():
            # 0.0 means this microstep only filled the accumulation buffer.
            # Keeping a float preserves the historical optimizer/trainer API.
            return 0.0
        self._activate_accumulated_gradients()
        from .backend import get_backend

        selected_backend = get_backend()
        active = self._current_active()
        active_decay_values = {
            self.decay_mask[index] for index, *_ in active
        }
        fused = getattr(
            selected_backend, "adamw_update_many_clipped", None,
        )
        if callable(fused) and active and len(active_decay_values) == 1:
            self.step_count += 1
            use_decay = next(iter(active_decay_values))
            return float(fused(
                [item[1].data for item in active],
                [item[2] for item in active],
                [item[3] for item in active],
                [item[4] for item in active],
                max_norm=max_norm,
                learning_rate=self.learning_rate,
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon,
                weight_decay=self.weight_decay if use_decay else 0.0,
                step=self.step_count,
            ))
        norm = self.clip_grad_norm(max_norm)
        self._apply_update()
        return norm

    def state_dict(self) -> dict[str, object]:
        pending: list[array] = []
        if self.accumulation_count > 0:
            pending = [
                (
                    array("f", values)
                    if values is not None
                    else array("f", [0.0]) * parameter.numel
                )
                for parameter, values in zip(
                    self.parameters, self.gradient_accumulators
                )
            ]
        accumulator_mask = [
            values is not None for values in self.gradient_accumulators
        ]
        return {
            "type": "AdamW", "learning_rate": self.learning_rate,
            "beta1": self.beta1, "beta2": self.beta2, "epsilon": self.epsilon,
            "weight_decay": self.weight_decay, "step_count": self.step_count,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "accumulation_count": self.accumulation_count,
            "decay_mask": list(self.decay_mask),
            "gradient_accumulator_mask": accumulator_mask,
            "first_moments": [array("f", values) for values in self.first_moments],
            "second_moments": [array("f", values) for values in self.second_moments],
            "gradient_accumulators": pending,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Строго восстанавливает гиперпараметры, moments и незавершённое окно."""
        if state.get("type") not in (None, "AdamW"):
            raise ValueError("checkpoint содержит состояние другого оптимизатора")
        first = state.get("first_moments")
        second = state.get("second_moments")
        if not isinstance(first, list) or not isinstance(second, list):
            raise ValueError("состояние AdamW не содержит moments")
        expected = [parameter.numel for parameter in self.parameters]
        if [len(values) for values in first] != expected or [len(values) for values in second] != expected:
            raise ValueError("размеры moments AdamW не совпадают с параметрами")

        accumulation_steps = int(
            state.get("gradient_accumulation_steps", self.gradient_accumulation_steps)
        )
        accumulation_count = int(state.get("accumulation_count", 0))
        if accumulation_steps <= 0:
            raise ValueError("checkpoint содержит неверный gradient_accumulation_steps")
        if not 0 <= accumulation_count < accumulation_steps:
            raise ValueError("checkpoint содержит неверный accumulation_count")

        stored_decay_mask = state.get("decay_mask")
        if stored_decay_mask is None:
            decay_mask = list(self.decay_mask)
        elif (
            isinstance(stored_decay_mask, list)
            and len(stored_decay_mask) == len(self.parameters)
            and all(type(value) is bool for value in stored_decay_mask)
        ):
            decay_mask = list(stored_decay_mask)
        else:
            raise ValueError("checkpoint содержит неверный decay_mask")

        pending = state.get("gradient_accumulators", [])
        accumulator_mask = state.get("gradient_accumulator_mask")
        if not isinstance(pending, list):
            raise ValueError("checkpoint содержит неверные gradient_accumulators")
        if accumulator_mask is None:
            checked_accumulator_mask = [True] * len(self.parameters)
        elif (
            isinstance(accumulator_mask, list)
            and len(accumulator_mask) == len(self.parameters)
            and all(type(value) is bool for value in accumulator_mask)
        ):
            checked_accumulator_mask = list(accumulator_mask)
        else:
            raise ValueError("checkpoint содержит неверный gradient_accumulator_mask")
        if accumulation_count == 0:
            if pending:
                raise ValueError(
                    "checkpoint содержит gradient_accumulators при accumulation_count=0"
                )
            restored_pending: list[array | None] = [None] * len(self.parameters)
        else:
            if [len(values) for values in pending] != expected:
                raise ValueError(
                    "размеры gradient_accumulators не совпадают с параметрами"
                )
            restored_pending = [
                array("f", values) if active else None
                for values, active in zip(pending, checked_accumulator_mask)
            ]

        self.learning_rate = float(state["learning_rate"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.epsilon = float(state["epsilon"])
        self.weight_decay = float(state["weight_decay"])
        self.step_count = int(state["step_count"])
        self.gradient_accumulation_steps = accumulation_steps
        self.accumulation_count = accumulation_count
        self.decay_mask = decay_mask
        self.first_moments = [array("f", values) for values in first]
        self.second_moments = [array("f", values) for values in second]
        self.gradient_accumulators = restored_pending
        self._register_native_state()
        self._register_accumulation_state()
