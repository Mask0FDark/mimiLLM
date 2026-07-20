"""High-level training workflow for projects that use mimiLLM as a library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .api import load_model, save_model
from .backend import get_backend, reset_backend
from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import TokenDataset, load_qa_text, load_text_documents
from .optim import AdamW
from .tensor import no_grad
from .tokenizer import (
    BpeTokenizer,
    ByteTokenizer,
    analyze_tokenizer,
    create_tokenizer,
    format_qa_text,
    save_tokenizer,
    save_tokenizer_report,
    train_bpe_tokenizer,
)
from .transformer import DecoderTransformer, TransformerConfig
from .utils import flatten, learning_rate_at


@dataclass(frozen=True)
class TrainingResult:
    """Files and in-memory model produced by a training run."""

    model: DecoderTransformer
    weights_dir: Path
    checkpoint_path: Path
    step: int
    interrupted: bool = False


def _format_duration(seconds: float | None) -> str:
    """Formats an elapsed time or ETA without external progress-bar packages."""
    if seconds is None:
        return "--:--"
    total = max(0, int(seconds + 0.5))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _progress_bar(step: int, total: int, width: int = 20) -> str:
    """Builds a compact ASCII progress bar that works in Windows terminals."""
    ratio = min(1.0, max(0.0, step / total))
    filled = min(width, int(ratio * width))
    if filled >= width:
        body = "=" * width
    else:
        body = "=" * filled + ">" + "." * (width - filled - 1)
    return f"[{body}]"


class _TrainingProgress:
    """Renders separate epoch blocks with a live batch row, similar to YOLO CLI."""

    def __init__(
        self, total_steps: int, start_step: int, batches_per_epoch: int,
    ) -> None:
        self.total_steps = total_steps
        self.start_step = start_step
        self.batches_per_epoch = batches_per_epoch
        self.total_epochs = math.ceil(total_steps / batches_per_epoch)
        self.started = time.perf_counter()
        self.interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._line_open = False
        self.current_epoch = 0
        self.current_epoch_batches = 0
        self.epoch_started = self.started

    def _location(self, step: int) -> tuple[int, int, int]:
        epoch = (step - 1) // self.batches_per_epoch + 1
        batch = (step - 1) % self.batches_per_epoch + 1
        first_step = (epoch - 1) * self.batches_per_epoch + 1
        batches = min(self.batches_per_epoch, self.total_steps - first_step + 1)
        return epoch, batch, batches

    def starts_epoch(self, step: int) -> bool:
        epoch, _, _ = self._location(step)
        return epoch != self.current_epoch

    def ends_epoch(self, step: int) -> bool:
        _, batch, batches = self._location(step)
        return batch == batches

    def begin_epoch(self, next_step: int) -> None:
        self.close_line()
        epoch, _, batches = self._location(next_step)
        self.current_epoch = epoch
        self.current_epoch_batches = batches
        self.epoch_started = time.perf_counter()
        first_step = (epoch - 1) * self.batches_per_epoch + 1
        last_step = first_step + batches - 1
        print(
            f"\nEpoch {epoch}/{self.total_epochs} | "
            f"batches={batches} | global steps={first_step}-{last_step}",
            flush=True,
        )
        print(
            f"{'Batch':>9}  {'Step':>11}  {'Progress':<18}  {'Src':>22}  "
            f"{'Phase':>9}  {'Tokens':>6}  {'Loss':>9}  {'Avg loss':>9}  {'Grad':>8}  "
            f"{'Val loss':>9}  {'LR':>9}  {'tok/s':>7}  "
            f"{'Time':>7}  {'Epoch':>8}  {'ETA':>8}",
            flush=True,
        )

    def stage(
        self,
        step: int,
        *,
        phase: str,
        source: str = "-",
        tokens: int = 0,
        train_loss: float | None = None,
        validation_loss: float | None = None,
        learning_rate: float = 0.0,
        batch_seconds: float | None = None,
    ) -> None:
        """Shows the current stage before a potentially long operation starts."""
        epoch, batch, batches = self._location(step)
        if epoch != self.current_epoch:
            self.begin_epoch(step)
        self._render(
            batch=batch,
            step=step,
            completed_step=step - 1,
            batches=batches,
            source=source,
            phase=phase,
            tokens=tokens,
            train_loss=train_loss,
            average_loss=None,
            gradient_norm=None,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            tokens_per_second=0.0,
            batch_seconds=batch_seconds,
        )

    def update(
        self,
        step: int,
        *,
        source: str,
        phase: str = "done",
        tokens: int,
        train_loss: float | None,
        average_loss: float | None,
        gradient_norm: float | None,
        validation_loss: float | None,
        learning_rate: float,
        tokens_per_second: float,
        batch_seconds: float | None,
        validating: bool = False,
        permanent: bool = False,
    ) -> None:
        epoch, batch, batches = self._location(step)
        if epoch != self.current_epoch:
            self.begin_epoch(step)
        self._render(
            batch=batch,
            step=step,
            batches=batches,
            source=source,
            phase=phase,
            tokens=tokens,
            train_loss=train_loss,
            average_loss=average_loss,
            gradient_norm=gradient_norm,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            tokens_per_second=tokens_per_second,
            batch_seconds=batch_seconds,
            validating=validating,
            permanent=permanent,
        )

    def _render(
        self,
        *,
        batch: int,
        step: int,
        completed_step: int | None = None,
        batches: int,
        source: str,
        phase: str,
        tokens: int,
        train_loss: float | None,
        average_loss: float | None,
        gradient_norm: float | None,
        validation_loss: float | None,
        learning_rate: float,
        tokens_per_second: float,
        batch_seconds: float | None,
        validating: bool = False,
        permanent: bool = False,
    ) -> None:
        elapsed = time.perf_counter() - self.started
        finished_step = step if completed_step is None else completed_step
        completed = finished_step - self.start_step
        remaining_steps = self.total_steps - finished_step
        eta = elapsed / completed * remaining_steps if completed > 0 else None
        epoch_elapsed = time.perf_counter() - self.epoch_started
        batch_text = f"{batch}/{batches}"
        step_text = f"{step}/{self.total_steps}"
        train_text = "-" if train_loss is None else f"{train_loss:.5f}"
        average_text = "-" if average_loss is None else f"{average_loss:.5f}"
        gradient_text = "-" if gradient_norm is None else f"{gradient_norm:.3f}"
        if validating:
            validation_text = "running"
        else:
            validation_text = "-" if validation_loss is None else f"{validation_loss:.5f}"
        if batch_seconds is None:
            batch_time_text = "-"
        elif batch_seconds < 60.0:
            batch_time_text = f"{batch_seconds:.2f}s"
        else:
            batch_time_text = _format_duration(batch_seconds)
        line = (
            f"{batch_text:>9}  {step_text:>11}  {_progress_bar(batch, batches, 16):<18}  "
            f"{source:>22}  {phase:>9}  {tokens:>6}  {train_text:>9}  {average_text:>9}  "
            f"{gradient_text:>8}  {validation_text:>9}  {learning_rate:>9.3g}  "
            f"{tokens_per_second:>7.1f}  {batch_time_text:>7}  "
            f"{_format_duration(epoch_elapsed):>8}  {_format_duration(eta):>8}"
        )
        if self.interactive:
            print(f"\r{line}", end="\n" if permanent else "", flush=True)
            self._line_open = not permanent
        else:
            print(line, flush=True)

    def finish_epoch(
        self,
        *,
        batches: int,
        tokens: int,
        average_loss: float,
        validation_loss: float | None,
        source_counts: dict[str, int],
        timings: dict[str, float],
        saved: bool,
    ) -> None:
        self.close_line()
        validation_text = "-" if validation_loss is None else f"{validation_loss:.5f}"
        source_text = " | ".join(
            f"{name}={count}" for name, count in sorted(source_counts.items())
        )
        saved_text = " | checkpoint=saved" if saved else ""
        timing_text = " | ".join(
            f"{name}={seconds / batches:.2f}s"
            for name, seconds in timings.items()
        )
        print(
            f"Epoch {self.current_epoch}/{self.total_epochs} complete | "
            f"batches={batches} | tokens={tokens:,} | avg_loss={average_loss:.5f} | "
            f"val_loss={validation_text} | {source_text} | "
            f"{timing_text} | time={_format_duration(time.perf_counter() - self.epoch_started)}"
            f"{saved_text}",
            flush=True,
        )

    def close_line(self) -> None:
        if self.interactive and self._line_open:
            print(flush=True)
            self._line_open = False


def _resolve(base_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_best_validation(path: Path) -> tuple[float, int]:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        loss = float(values["loss"])
        step = int(values["step"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid best-validation metadata: {path}") from exc
    if not math.isfinite(loss) or loss < 0.0 or step < 0:
        raise ValueError(f"invalid best-validation values: {path}")
    configured_weights = values.get("weights")
    if configured_weights is None:
        model_path = path.parent / "model.safetensors"
    elif isinstance(configured_weights, str) and configured_weights:
        weights_dir = Path(configured_weights)
        if not weights_dir.is_absolute():
            weights_dir = (path.parent / weights_dir).resolve()
        model_path = weights_dir / "model.safetensors"
    else:
        raise ValueError(f"invalid best-validation weights path: {path}")
    expected_hash = values.get("model_sha256")
    if expected_hash is not None:
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not model_path.is_file()
            or _file_sha256(model_path) != expected_hash
        ):
            raise ValueError(
                f"best-validation metadata does not match its model weights: {path}"
            )
    return loss, step


def _save_best_validation(
    path: Path, loss: float, step: int, weights_dir: Path,
) -> None:
    model_path = weights_dir / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"cannot record best validation without model weights: {model_path}"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        stored_weights = weights_dir.relative_to(path.parent).as_posix()
    except ValueError:
        stored_weights = str(weights_dir)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            {
                "loss": loss,
                "step": step,
                "weights": stored_weights,
                "model_sha256": _file_sha256(model_path),
            },
            stream,
            indent=2,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _save_validation_snapshot(
    destination: Path,
    model: DecoderTransformer,
    *,
    loss: float,
    step: int,
) -> Path:
    snapshot = destination / "validation" / f"step_{step:08d}"
    save_model(snapshot, model)
    metadata = snapshot / "validation.json"
    temporary = metadata.with_suffix(metadata.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump({"loss": loss, "step": step}, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(metadata)
    return snapshot


def _tokenizer_corpus(config: TransformerConfig, base_dir: Path) -> list[str]:
    corpus: list[str] = []
    if config.text_ratio > 0.0:
        text_train = _resolve(base_dir, config.text_train_path)
        corpus.extend(text for _, text in load_text_documents(text_train))
    if config.text_ratio < 1.0:
        question_train = _resolve(base_dir, config.question_train_path)
        corpus.extend(
            format_qa_text(question, answer)
            for question, answer in load_qa_text(question_train)
        )
    if not corpus:
        raise ValueError("tokenizer training corpus is empty")
    return corpus


def train_tokenizer_from_config(
    config_path: str | Path = "config.json",
    *,
    output_path: str | Path | None = None,
    vocab_size: int | None = None,
    min_frequency: int = 2,
    pretokenizer: str = BpeTokenizer.DEFAULT_PRETOKENIZER,
    ensure_unicode_characters: bool = True,
    report_path: str | Path | None = None,
) -> BpeTokenizer:
    """Trains BPE from train sources and writes a measured quality report."""
    path = Path(config_path).resolve()
    config = TransformerConfig.from_json(path)
    base_dir = path.parent
    target_vocab_size = vocab_size
    if target_vocab_size is None:
        target_vocab_size = (
            config.vocab_size if config.tokenizer.strip().lower() == "bpe" else 4096
        )
    corpus = _tokenizer_corpus(config, base_dir)
    tokenizer = train_bpe_tokenizer(
        corpus,
        vocab_size=target_vocab_size,
        min_frequency=min_frequency,
        pretokenizer=pretokenizer,
        ensure_unicode_characters=ensure_unicode_characters,
    )
    destination = Path(output_path) if output_path is not None else base_dir / "tokenizer.json"
    if not destination.is_absolute():
        destination = base_dir / destination
    save_tokenizer(tokenizer, destination)
    report_destination = (
        Path(report_path) if report_path is not None
        else destination.with_name("tokenizer_report.json")
    )
    if not report_destination.is_absolute():
        report_destination = base_dir / report_destination
    save_tokenizer_report(
        analyze_tokenizer(tokenizer, corpus), report_destination,
    )
    return tokenizer


def _datasets(
    config: TransformerConfig, base_dir: Path, tokenizer: ByteTokenizer,
) -> tuple[TokenDataset, TokenDataset]:
    question_train = (
        _resolve(base_dir, config.question_train_path) if config.text_ratio < 1.0 else None
    )
    question_validation = (
        _resolve(base_dir, config.question_validation_path) if config.text_ratio < 1.0 else None
    )
    text_train = _resolve(base_dir, config.text_train_path) if config.text_ratio > 0.0 else None
    text_validation = (
        _resolve(base_dir, config.text_validation_path) if config.text_ratio > 0.0 else None
    )
    dataset_options = {
        "tokenizer": tokenizer,
        "text_ratio": config.text_ratio,
        "qa_prompt_weight": config.qa_prompt_weight,
        "qa_answer_prefix_weight": config.qa_answer_prefix_weight,
        "qa_answer_prefix_tokens": config.qa_answer_prefix_tokens,
        "qa_source_weights": config.qa_source_weights,
    }
    return (
        TokenDataset(question_train, text_paths=text_train, **dataset_options),
        TokenDataset(question_validation, text_paths=text_validation, **dataset_options),
    )


def _batches_per_epoch(dataset: TokenDataset, config: TransformerConfig) -> int:
    """Uses an explicit value or estimates batches needed to cover each source."""
    if config.batches_per_epoch is not None:
        return config.batches_per_epoch

    estimates: list[int] = []
    for source, probability in dataset.source_weights():
        if source == "text":
            windows = sum(
                max(1, math.ceil((len(sequence) - 1) / config.context_length))
                for sequence in dataset.text_sequences
            )
            batches = math.ceil(windows / config.batch_size)
        else:
            batches = math.ceil(
                len(dataset.qa_sequences_by_source[source]) / config.batch_size
            )
        estimates.append(math.ceil(batches / probability))
    return max(1, *estimates)


def validation_loss(
    model: DecoderTransformer, dataset: TokenDataset, config: TransformerConfig,
    *, progress_callback: Callable[[str, int, int], None] | None = None,
) -> float:
    """Computes loss over every supervised validation token."""
    total = 0.0
    with no_grad():
        for source, weight in dataset.source_weights():
            source_loss = 0.0
            source_tokens = 0.0
            batches = dataset.validation_batch_count(
                config.batch_size, config.context_length, source=source,
            )
            for batch_index, (inputs, targets, loss_weights) in enumerate(
                dataset.validation_batches(
                    config.batch_size, config.context_length, source=source,
                ),
                1,
            ):
                supervised_tokens = sum(sum(row) for row in loss_weights)
                logits = model(inputs)
                batch_loss = logits.reshape(-1, config.vocab_size).cross_entropy(
                    flatten(targets), weights=flatten(loss_weights)  # type: ignore[arg-type]
                ).item()
                source_loss += batch_loss * supervised_tokens
                source_tokens += supervised_tokens
                if progress_callback is not None:
                    progress_callback(source, batch_index, batches)
            if source_tokens <= 0.0:
                raise ValueError(f"validation source {source!r} has no supervised tokens")
            total += weight * source_loss / source_tokens
    return total


def train_model(
    config: TransformerConfig,
    *,
    base_dir: str | Path = ".",
    output_dir: str | Path = "weights",
    resume: str | Path | None = None,
    init_from: str | Path | None = None,
) -> TrainingResult:
    """Trains from configured project data and exports standard reusable weights."""
    if resume is not None and init_from is not None:
        raise ValueError("resume and init_from are mutually exclusive")
    project_dir = Path(base_dir).resolve()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = project_dir / destination
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "training_checkpoint.bin"
    latest_weights_dir = destination / "last"
    best_validation_path = destination / "best_validation.json"

    tokenizer_path = _resolve(project_dir, config.tokenizer_path)
    tokenizer = create_tokenizer(
        config.tokenizer,
        path=tokenizer_path if config.tokenizer.strip().lower() == "bpe" else None,
    )
    if tokenizer.VOCAB_SIZE != config.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary has {tokenizer.VOCAB_SIZE} tokens, "
            f"but config vocab_size is {config.vocab_size}"
        )
    if isinstance(tokenizer, BpeTokenizer):
        save_tokenizer(tokenizer, destination / "tokenizer.json")

    print("\nmimiLLM training", flush=True)
    print(f"Preparing datasets from {project_dir} ...", end=" ", flush=True)
    train_data, validation_data = _datasets(config, project_dir, tokenizer)
    print("done", flush=True)
    print("Building model and optimizer ...", end=" ", flush=True)
    model = DecoderTransformer(config, tokenizer_model=tokenizer)
    initialized_from: Path | None = None
    if init_from is not None:
        initialized_from = Path(init_from)
        if not initialized_from.is_absolute():
            initialized_from = project_dir / initialized_from
        initialized_from = initialized_from.resolve()
        source_model = load_model(initialized_from, eval_mode=False)
        tokenizer_matches = type(source_model.tokenizer) is type(tokenizer)
        if tokenizer_matches and isinstance(tokenizer, BpeTokenizer):
            tokenizer_matches = source_model.tokenizer.to_dict() == tokenizer.to_dict()
        if not tokenizer_matches:
            raise ValueError(
                "init_from uses a different tokenizer; staged training must reuse "
                "the exact same tokenizer vocabulary and merge order"
            )
        try:
            model.load_state_dict(source_model.state_dict())
        except ValueError as exc:
            raise ValueError(
                "init_from model architecture is incompatible with the new stage"
            ) from exc
    optimizer = AdamW(
        model.parameters(), config.learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    print("done", flush=True)
    if initialized_from is not None:
        print(
            f"Initialized weights from {initialized_from}; optimizer and LR schedule reset",
            flush=True,
        )
    rng = random.Random(config.seed)
    start_step = 0
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.is_absolute():
            resume_path = project_dir / resume_path
        loaded = load_checkpoint(resume_path, model, optimizer)
        start_step = loaded.step
        if start_step > config.steps:
            raise ValueError(
                f"checkpoint step {start_step} exceeds configured steps {config.steps}"
            )
        rng = random.Random(loaded.seed + start_step)
        print(f"Resume: step {start_step} from {resume_path}", flush=True)

    best_validation_loss = math.inf
    best_validation_step = -1
    if resume is not None and best_validation_path.is_file():
        best_validation_loss, best_validation_step = _load_best_validation(
            best_validation_path
        )
        print(
            f"Best validation: {best_validation_loss:.5f} at step {best_validation_step}",
            flush=True,
        )

    backend = get_backend()
    backend_name = getattr(backend, "name", "python")
    if backend_name == "cuda":
        backend_details = (
            f"device={getattr(backend, 'device_name', 'NVIDIA GPU')} | "
            f"vram={getattr(backend, 'device_memory', 0) / (1024 ** 3):.1f}GB"
        )
    else:
        backend_details = f"threads={getattr(backend, 'num_threads', 1)}"
    batches_per_epoch = _batches_per_epoch(train_data, config)
    total_epochs = math.ceil(config.steps / batches_per_epoch)
    print(
        f"Model: {model.parameter_count():,} parameters | "
        f"layers={config.n_layers} | heads={config.n_heads} | "
        f"d_model={config.d_model} | context={config.context_length}",
        flush=True,
    )
    print(
        f"Data: {len(train_data.examples)} questions | "
        f"{len(train_data.text_documents)} texts | "
        f"{train_data.qa_tokens + train_data.text_tokens:,} train tokens | "
        f"text_ratio={config.text_ratio:.2f}",
        flush=True,
    )
    print(
        "Source mix: " + " | ".join(
            f"{source}={weight:.1%}"
            for source, weight in train_data.source_weights()
        ),
        flush=True,
    )
    print(
        f"Run: {start_step}->{config.steps} steps | epochs={total_epochs} | "
        f"batches/epoch={batches_per_epoch} | batch_size={config.batch_size} | "
        f"backend={backend_name} | {backend_details} | output={destination}",
        flush=True,
    )
    print(
        f"Optimizer: AdamW | betas=({config.adam_beta1:g}, {config.adam_beta2:g}) | "
        f"weight_decay={config.weight_decay:g} | clip={config.gradient_clip_norm:g} | "
        f"schedule={config.learning_rate_schedule} | warmup={config.warmup_steps} | "
        f"min_lr={config.learning_rate * config.min_learning_rate_ratio:g}",
        flush=True,
    )
    progress = _TrainingProgress(config.steps, start_step, batches_per_epoch)

    last_step = start_step
    last_checkpoint_step = -1
    latest_validation_loss: float | None = None
    epoch_loss_sum = 0.0
    epoch_batches = 0
    epoch_tokens = 0
    epoch_sources: dict[str, int] = {}
    epoch_timings: dict[str, float] = {}
    interrupted = False
    stopped_early = False
    validations_without_improvement = 0
    try:
        for step in range(start_step + 1, config.steps + 1):
            if progress.starts_epoch(step):
                progress.begin_epoch(step)
                epoch_loss_sum = 0.0
                epoch_batches = 0
                epoch_tokens = 0
                epoch_sources = {}
                epoch_timings = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
            started = time.perf_counter()
            optimizer.learning_rate = learning_rate_at(
                step, config.steps, config.learning_rate, config.warmup_steps,
                schedule=config.learning_rate_schedule,
                min_ratio=config.min_learning_rate_ratio,
            )
            progress.stage(
                step,
                phase="data",
                validation_loss=latest_validation_loss,
                learning_rate=optimizer.learning_rate,
                batch_seconds=0.0,
            )
            inputs, targets, loss_weights = train_data.sample_batch_with_loss_weights(
                config.batch_size, config.context_length, rng
            )
            data_finished = time.perf_counter()
            tokens = sum(len(row) for row in inputs)
            progress.stage(
                step,
                phase="forward",
                source=train_data.last_source,
                tokens=tokens,
                validation_loss=latest_validation_loss,
                learning_rate=optimizer.learning_rate,
                batch_seconds=data_finished - started,
            )
            logits = model(inputs)
            forward_finished = time.perf_counter()
            progress.stage(
                step,
                phase="loss",
                source=train_data.last_source,
                tokens=tokens,
                validation_loss=latest_validation_loss,
                learning_rate=optimizer.learning_rate,
                batch_seconds=forward_finished - started,
            )
            loss = logits.reshape(-1, config.vocab_size).cross_entropy(
                flatten(targets), weights=flatten(loss_weights)  # type: ignore[arg-type]
            )
            loss_value = loss.item()
            loss_finished = time.perf_counter()
            progress.stage(
                step,
                phase="backward",
                source=train_data.last_source,
                tokens=tokens,
                train_loss=loss_value,
                validation_loss=latest_validation_loss,
                learning_rate=optimizer.learning_rate,
                batch_seconds=loss_finished - started,
            )
            loss.backward()
            backward_finished = time.perf_counter()
            progress.stage(
                step,
                phase="optimizer",
                source=train_data.last_source,
                tokens=tokens,
                train_loss=loss_value,
                validation_loss=latest_validation_loss,
                learning_rate=optimizer.learning_rate,
                batch_seconds=backward_finished - started,
            )
            gradient_norm = optimizer.clip_grad_norm(config.gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_finished = time.perf_counter()
            last_step = step
            elapsed = optimizer_finished - started
            tokens_per_second = tokens / elapsed
            epoch_loss_sum += loss_value
            epoch_batches += 1
            epoch_tokens += tokens
            epoch_timings["forward"] += forward_finished - data_finished
            epoch_timings["backward"] += backward_finished - loss_finished
            epoch_timings["optimizer"] += optimizer_finished - backward_finished
            epoch_sources[train_data.last_source] = (
                epoch_sources.get(train_data.last_source, 0) + 1
            )
            average_loss = epoch_loss_sum / epoch_batches
            ends_epoch = progress.ends_epoch(step)
            should_validate = (
                step % config.validation_interval == 0
                or step == config.steps
            )
            should_checkpoint = (
                step % config.checkpoint_interval == 0
                or step == config.steps
            )
            if should_validate:
                progress.update(
                    step,
                    source=train_data.last_source,
                    phase="validate",
                    tokens=tokens,
                    train_loss=loss_value,
                    average_loss=average_loss,
                    gradient_norm=gradient_norm,
                    validation_loss=None,
                    learning_rate=optimizer.learning_rate,
                    tokens_per_second=tokens_per_second,
                    batch_seconds=elapsed,
                    validating=True,
                )
                def report_validation(source: str, batch: int, batches: int) -> None:
                    progress.stage(
                        step,
                        phase=f"val {batch}/{batches}",
                        source=source,
                        tokens=tokens,
                        train_loss=loss_value,
                        validation_loss=latest_validation_loss,
                        learning_rate=optimizer.learning_rate,
                        batch_seconds=time.perf_counter() - started,
                    )

                latest_validation_loss = validation_loss(
                    model,
                    validation_data,
                    config,
                    progress_callback=report_validation,
                )
                if config.save_validation_checkpoints:
                    _save_validation_snapshot(
                        destination,
                        model,
                        loss=latest_validation_loss,
                        step=step,
                    )
                previous_best = best_validation_loss
                if latest_validation_loss < best_validation_loss:
                    best_validation_loss = latest_validation_loss
                    best_validation_step = step
                    best_weights_dir = destination / "best"
                    save_model(best_weights_dir, model)
                    _save_best_validation(
                        best_validation_path,
                        best_validation_loss,
                        best_validation_step,
                        best_weights_dir,
                    )
                    # Keep the output root directly loadable for existing users.
                    save_model(destination, model)
                if latest_validation_loss < (
                    previous_best - config.early_stopping_min_delta
                ):
                    validations_without_improvement = 0
                else:
                    validations_without_improvement += 1
                if (
                    config.early_stopping_patience is not None
                    and validations_without_improvement >= config.early_stopping_patience
                ):
                    stopped_early = True
            progress.update(
                step,
                source=train_data.last_source,
                tokens=tokens,
                train_loss=loss_value,
                average_loss=average_loss,
                gradient_norm=gradient_norm,
                validation_loss=latest_validation_loss,
                learning_rate=optimizer.learning_rate,
                tokens_per_second=tokens_per_second,
                batch_seconds=elapsed,
                permanent=True,
            )
            if should_checkpoint:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    config=config.to_dict(),
                    step=step,
                    seed=config.seed,
                )
                last_checkpoint_step = step
                save_model(latest_weights_dir, model)
            if stopped_early:
                if last_checkpoint_step != step:
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        config=config.to_dict(),
                        step=step,
                        seed=config.seed,
                    )
                    last_checkpoint_step = step
                    save_model(latest_weights_dir, model)
                progress.close_line()
                print(
                    f"Early stopping at step {step}: validation did not improve by "
                    f"{config.early_stopping_min_delta:g} for "
                    f"{config.early_stopping_patience} validations",
                    flush=True,
                )
                break
            if ends_epoch:
                progress.finish_epoch(
                    batches=epoch_batches,
                    tokens=epoch_tokens,
                    average_loss=average_loss,
                    validation_loss=latest_validation_loss,
                    source_counts=epoch_sources,
                    timings=epoch_timings,
                    saved=should_checkpoint,
                )
    except KeyboardInterrupt:
        interrupted = True
        progress.close_line()
        print("Stopping training and saving the current state ...", flush=True)
        checkpoint_path = destination / "training_checkpoint_interrupted.bin"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config=config.to_dict(),
            step=last_step,
            seed=config.seed,
        )
        last_checkpoint_step = last_step
        save_model(latest_weights_dir, model)
        if best_validation_step < 0:
            save_model(destination, model)
        print(f"Training interrupted at step {last_step}. Saved: {checkpoint_path}", flush=True)

    if not interrupted:
        progress.close_line()
        if last_checkpoint_step != last_step:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config=config.to_dict(),
                step=last_step,
                seed=config.seed,
            )
        save_model(latest_weights_dir, model)
        if best_validation_step < 0:
            save_model(destination, model)
        status = "Training stopped early" if stopped_early else "Training complete"
        print(f"{status}: {last_step}/{config.steps} steps", flush=True)
        if best_validation_step >= 0:
            print(
                f"Best validation: {best_validation_loss:.5f} at step "
                f"{best_validation_step} | weights={destination}",
                flush=True,
            )
        print(f"Latest weights: {latest_weights_dir}", flush=True)
    return TrainingResult(model, destination, checkpoint_path, last_step, interrupted)


def train_from_config(
    config_path: str | Path = "config.json",
    *,
    output_dir: str | Path = "weights",
    resume: str | Path | None = None,
    init_from: str | Path | None = None,
    steps: int | None = None,
    batches_per_epoch: int | None = None,
    backend: str | None = None,
) -> TrainingResult:
    """Loads JSON config; relative data paths are based on the config's directory."""
    path = Path(config_path).resolve()
    config = TransformerConfig.from_json(path)
    if steps is not None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        config = replace(config, steps=steps)
    if batches_per_epoch is not None:
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        config = replace(config, batches_per_epoch=batches_per_epoch)
    if backend is not None:
        selected = backend.lower()
        if selected not in {"auto", "cuda", "cpp", "python"}:
            raise ValueError("backend must be auto, cuda, cpp, or python")
        os.environ["MIMILLM_BACKEND"] = selected
        reset_backend()
    return train_model(
        config,
        base_dir=path.parent,
        output_dir=output_dir,
        resume=resume,
        init_from=init_from,
    )


def main(
    default_config: str | Path = "config.json",
    default_output_dir: str | Path = "weights",
) -> None:
    parser = argparse.ArgumentParser(description="Train a mimiLLM model")
    parser.add_argument(
        "--config", type=Path, default=Path(default_config),
        help=f"path to the training configuration (default: {default_config})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(default_output_dir),
        help=f"directory for model weights and checkpoints (default: {default_output_dir})",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="resume from a training checkpoint",
    )
    parser.add_argument(
        "--init-from", type=Path,
        help="initialize a new training stage from reusable model weights",
    )
    parser.add_argument(
        "--steps", type=int,
        help="override the total number of optimizer steps from the configuration",
    )
    parser.add_argument(
        "--batches-per-epoch", type=int,
        help="number of optimizer batches grouped into one displayed epoch",
    )
    parser.add_argument(
        "--backend", choices=("auto", "cuda", "cpp", "python"),
        help="compute backend; defaults to MIMILLM_BACKEND or auto",
    )
    args = parser.parse_args()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute()
        else (Path.cwd() / args.output_dir).resolve()
    )
    resume = args.resume
    if resume is not None and not resume.is_absolute():
        resume = (Path.cwd() / resume).resolve()
    init_from = args.init_from
    if init_from is not None and not init_from.is_absolute():
        init_from = (Path.cwd() / init_from).resolve()
    result = train_from_config(
        args.config,
        output_dir=output_dir,
        resume=resume,
        init_from=init_from,
        steps=args.steps,
        batches_per_epoch=args.batches_per_epoch,
        backend=args.backend,
    )
    print(f"Weights: {result.weights_dir}")
    print(f"Training checkpoint: {result.checkpoint_path}")
