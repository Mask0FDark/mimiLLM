"""Experimental CUDA Graph training path for fixed Transformer shapes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from .backend import get_backend
from .optim import AdamW
from .transformer import DecoderTransformer
from .utils import flatten


@dataclass(frozen=True)
class StaticTrainingStepResult:
    """Host-visible metrics produced by one captured training step."""

    loss: float
    gradient_norm: float | None
    seconds: float
    tokens: int
    graph_seconds: float
    optimizer_seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0.0 else float("inf")


class StaticCudaTrainer:
    """Replay a fixed-shape mimiLLM forward/backward through a CUDA Graph.

    Model math and autograd topology are prepared once. Later calls only upload
    token IDs and targets, launch the captured graph, and run the ordinary
    AdamW optimizer. Keeping AdamW outside the graph preserves its dynamic step
    counter, learning-rate schedules, clipping, and checkpoint format.
    """

    def __init__(
        self,
        model: DecoderTransformer,
        optimizer: AdamW,
        sample_inputs: list[list[int]],
        sample_targets: list[list[int]],
        sample_loss_weights: list[list[float]] | None = None,
    ) -> None:
        backend = get_backend()
        required = (
            "create_static_trace", "capture_static", "static_loss",
            "synchronize",
        )
        if getattr(backend, "name", None) != "cuda" or any(
            not hasattr(backend, name) for name in required
        ):
            raise RuntimeError(
                "StaticCudaTrainer requires the mimiLLM CUDA backend"
            )
        if hasattr(backend, "set_tf32"):
            backend.set_tf32(model.config.cuda_tf32)
        if optimizer.parameters != model.parameters():
            raise ValueError("optimizer parameters must belong to the captured model")
        self.backend = backend
        self.model = model
        self.optimizer = optimizer
        self.batch_size = len(sample_inputs)
        self.sequence_length = len(sample_inputs[0]) if sample_inputs else 0
        self._validate_batch(sample_inputs, sample_targets)
        self._uses_loss_weights = sample_loss_weights is not None
        self._captured_weight_sum = (
            sum(flatten(sample_loss_weights))
            if sample_loss_weights is not None else None
        )
        if (
            self._captured_weight_sum is not None
            and self._captured_weight_sum <= 0.0
        ):
            raise ValueError("static CUDA loss weights must have a positive sum")
        self.trace = backend.create_static_trace()
        self.graph: Any | None = None
        self._gradient_storages: list[Any] = []
        self._closed = False
        self._prepare(sample_inputs, sample_targets, sample_loss_weights)

    def _validate_batch(
        self, inputs: list[list[int]], targets: list[list[int]],
    ) -> None:
        if len(inputs) != self.batch_size or len(targets) != self.batch_size:
            raise ValueError(
                "static CUDA batch size cannot change after capture"
            )
        if self.sequence_length <= 0:
            raise ValueError("static CUDA sequences cannot be empty")
        if any(len(row) != self.sequence_length for row in inputs):
            raise ValueError(
                "static CUDA input sequence length cannot change after capture"
            )
        if any(len(row) != self.sequence_length for row in targets):
            raise ValueError(
                "static CUDA target sequence length cannot change after capture"
            )

    def _forward_backward(
        self, inputs: list[list[int]], targets: list[list[int]],
        loss_weights: list[list[float]] | None,
    ) -> None:
        logits = self.model(inputs)
        loss = logits.reshape(
            -1, self.model.config.vocab_size,
        ).cross_entropy(
            flatten(targets),
            weights=flatten(loss_weights) if loss_weights is not None else None,
        )
        loss.backward()

    def _prepare(
        self, sample_inputs: list[list[int]], sample_targets: list[list[int]],
        sample_loss_weights: list[list[float]] | None,
    ) -> None:
        self.optimizer.zero_grad()
        with self.trace.recording():
            self._forward_backward(
                sample_inputs, sample_targets, sample_loss_weights,
            )
        self.backend.synchronize()
        self.optimizer.zero_grad()

        def capture_callback() -> None:
            self._forward_backward(
                sample_inputs, sample_targets, sample_loss_weights,
            )

        with self.trace.reusing():
            self.graph = self.backend.capture_static(capture_callback)
        gradients = [parameter.grad for parameter in self.optimizer.parameters]
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("static CUDA capture did not produce every gradient")
        self._gradient_storages = gradients
        self.optimizer.zero_grad()

    def step(
        self,
        inputs: list[list[int]],
        targets: list[list[int]],
        *,
        loss_weights: list[list[float]] | None = None,
        clip_grad_norm: float | None = None,
    ) -> StaticTrainingStepResult:
        if self._closed or self.graph is None:
            raise RuntimeError("StaticCudaTrainer is closed")
        self._validate_batch(inputs, targets)
        if (loss_weights is not None) != self._uses_loss_weights:
            raise ValueError(
                "static CUDA loss-weight mode cannot change after capture"
            )
        started = time.perf_counter()
        self.trace.update("token_ids", flatten(inputs))
        self.trace.update("targets", flatten(targets))
        if loss_weights is not None:
            flat_weights = [float(value) for value in flatten(loss_weights)]
            weight_sum = sum(flat_weights)
            if weight_sum <= 0.0:
                raise ValueError("static CUDA loss weights must have a positive sum")
            assert self._captured_weight_sum is not None
            scale = self._captured_weight_sum / weight_sum
            self.trace.update(
                "loss_weights",
                [value * scale for value in flat_weights],
            )
        self.graph.launch()
        loss = self.backend.static_loss(self.trace)
        graph_finished = time.perf_counter()
        for parameter, gradient in zip(
            self.optimizer.parameters, self._gradient_storages,
        ):
            parameter.grad = gradient
        gradient_norm = (
            self.optimizer.clip_grad_norm(clip_grad_norm)
            if clip_grad_norm is not None
            else None
        )
        self.optimizer.step()
        self.backend.synchronize()
        optimizer_finished = time.perf_counter()
        seconds = optimizer_finished - started
        return StaticTrainingStepResult(
            loss=loss,
            gradient_norm=gradient_norm,
            seconds=seconds,
            tokens=self.batch_size * self.sequence_length,
            graph_seconds=graph_finished - started,
            optimizer_seconds=optimizer_finished - graph_finished,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        self.optimizer.zero_grad()
        self._gradient_storages.clear()
        self.trace.close()
        self._closed = True

    def __enter__(self) -> "StaticCudaTrainer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def compile_static_cuda_training(
    model: DecoderTransformer,
    optimizer: AdamW,
    sample_inputs: list[list[int]],
    sample_targets: list[list[int]],
    sample_loss_weights: list[list[float]] | None = None,
) -> StaticCudaTrainer:
    """Compile a fixed batch/context Transformer train step into a CUDA Graph."""
    return StaticCudaTrainer(
        model, optimizer, sample_inputs, sample_targets, sample_loss_weights,
    )
