"""Experimental CPU+GPU data-parallel training for one local machine."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from .backend import backend_scope, get_backend
from .backend_cpp import CppBackend
from .transformer import DecoderTransformer
from .utils import flatten


@dataclass(frozen=True)
class HybridWorkerResult:
    """Loss and timings produced by one device shard."""

    loss: float
    weight: float
    forward_seconds: float
    backward_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.forward_seconds + self.backward_seconds


@dataclass(frozen=True)
class HybridBatchResult:
    """Combined result of one concurrent CPU+GPU forward/backward pass."""

    loss: float
    gpu: HybridWorkerResult
    cpu: HybridWorkerResult
    gpu_batch_size: int
    cpu_batch_size: int
    wall_seconds: float
    merge_seconds: float


class HybridDataParallel:
    """Splits a batch between CUDA and C++, then averages model gradients."""

    def __init__(
        self,
        model: DecoderTransformer,
        *,
        cpu_batch_size: int = 1,
        cpu_threads: int | None = 4,
        gpu_backend: Any | None = None,
        cpu_backend: Any | None = None,
    ) -> None:
        if cpu_batch_size <= 0:
            raise ValueError("cpu_batch_size must be positive")
        if cpu_threads is not None and cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive or None")
        self.model = model
        self.cpu_batch_size = cpu_batch_size
        self.gpu_backend = gpu_backend or get_backend()
        if getattr(self.gpu_backend, "name", None) != "cuda" and gpu_backend is None:
            raise RuntimeError("hybrid training requires the CUDA backend")
        self.cpu_backend = cpu_backend or CppBackend()
        if cpu_threads is not None and hasattr(self.cpu_backend, "set_num_threads"):
            self.cpu_backend.set_num_threads(cpu_threads)
        self.cpu_threads = getattr(self.cpu_backend, "num_threads", cpu_threads)
        self.cpu_model = DecoderTransformer(model.config)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mimillm-hybrid")
        self.sync_replica()

    def sync_replica(self) -> None:
        """Copies updated host-side parameters to the CPU model replica."""
        for source, destination in zip(
            self.model.parameters(), self.cpu_model.parameters(), strict=True,
        ):
            destination.data[:] = source.data
        self.cpu_model.zero_grad()

    @staticmethod
    def _worker(
        model: DecoderTransformer,
        selected_backend: Any,
        inputs: list[list[int]],
        targets: list[list[int]],
        loss_weights: list[list[float]],
    ) -> HybridWorkerResult:
        weight = float(sum(sum(row) for row in loss_weights))
        if weight <= 0.0:
            raise ValueError("a hybrid shard has no supervised tokens")
        model.zero_grad()
        with backend_scope(selected_backend):
            started = time.perf_counter()
            logits = model(inputs)
            forward_finished = time.perf_counter()
            loss = logits.reshape(-1, model.config.vocab_size).cross_entropy(
                flatten(targets), weights=flatten(loss_weights),  # type: ignore[arg-type]
            )
            loss_value = loss.item()
            loss.backward()
            backward_finished = time.perf_counter()
        return HybridWorkerResult(
            loss=loss_value,
            weight=weight,
            forward_seconds=forward_finished - started,
            backward_seconds=backward_finished - forward_finished,
        )

    @staticmethod
    def _resolve(futures: list[Future[HybridWorkerResult]]) -> list[HybridWorkerResult]:
        try:
            return [future.result() for future in futures]
        except BaseException:
            wait(futures)
            raise

    def forward_backward(
        self,
        inputs: list[list[int]],
        targets: list[list[int]],
        loss_weights: list[list[float]],
    ) -> HybridBatchResult:
        """Runs both shards concurrently and writes combined gradients to model."""
        batch_size = len(inputs)
        if batch_size < 2:
            raise ValueError("hybrid training requires batch_size of at least 2")
        if len(targets) != batch_size or len(loss_weights) != batch_size:
            raise ValueError("hybrid inputs, targets, and weights must have equal batch size")
        cpu_size = min(self.cpu_batch_size, batch_size - 1)
        gpu_size = batch_size - cpu_size
        started = time.perf_counter()
        gpu_future = self.executor.submit(
            self._worker,
            self.model,
            self.gpu_backend,
            inputs[:gpu_size],
            targets[:gpu_size],
            loss_weights[:gpu_size],
        )
        cpu_future = self.executor.submit(
            self._worker,
            self.cpu_model,
            self.cpu_backend,
            inputs[gpu_size:],
            targets[gpu_size:],
            loss_weights[gpu_size:],
        )
        gpu_result, cpu_result = self._resolve([gpu_future, cpu_future])
        workers_finished = time.perf_counter()
        total_weight = gpu_result.weight + cpu_result.weight
        gpu_scale = gpu_result.weight / total_weight
        cpu_scale = cpu_result.weight / total_weight
        with backend_scope(self.cpu_backend):
            for gpu_parameter, cpu_parameter in zip(
                self.model.parameters(), self.cpu_model.parameters(), strict=True,
            ):
                if gpu_parameter.grad is None or cpu_parameter.grad is None:
                    raise RuntimeError("a hybrid model parameter is missing its gradient")
                gpu_gradient = self.cpu_backend.scalar_multiply(
                    gpu_parameter.grad, gpu_scale,
                )
                cpu_gradient = self.cpu_backend.scalar_multiply(
                    cpu_parameter.grad, cpu_scale,
                )
                gpu_parameter.grad = self.cpu_backend.add(gpu_gradient, cpu_gradient)
        merged = time.perf_counter()
        return HybridBatchResult(
            loss=(gpu_result.loss * gpu_result.weight + cpu_result.loss * cpu_result.weight)
            / total_weight,
            gpu=gpu_result,
            cpu=cpu_result,
            gpu_batch_size=gpu_size,
            cpu_batch_size=cpu_size,
            wall_seconds=workers_finished - started,
            merge_seconds=merged - workers_finished,
        )

    def close(self) -> None:
        """Stops persistent worker threads after outstanding calls finish."""
        self.executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "HybridDataParallel":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
