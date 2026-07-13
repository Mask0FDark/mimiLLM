"""High-level training workflow for projects that use mimiLLM as a library."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .api import save_model
from .backend import get_backend
from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import TokenDataset
from .optim import AdamW
from .tensor import no_grad
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
            f"{'Batch':>9}  {'Step':>11}  {'Progress':<18}  {'Src':>5}  "
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
            f"{source:>5}  {phase:>9}  {tokens:>6}  {train_text:>9}  {average_text:>9}  "
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


def _datasets(
    config: TransformerConfig, base_dir: Path,
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
    return (
        TokenDataset(question_train, text_paths=text_train, text_ratio=config.text_ratio),
        TokenDataset(
            question_validation,
            text_paths=text_validation,
            text_ratio=config.text_ratio,
        ),
    )


def _batches_per_epoch(dataset: TokenDataset, config: TransformerConfig) -> int:
    """Uses an explicit value or estimates batches needed to cover each source."""
    if config.batches_per_epoch is not None:
        return config.batches_per_epoch

    estimates: list[int] = []
    has_questions = bool(dataset.sequences)
    has_texts = bool(dataset.text_sequences)
    if has_questions:
        question_batches = math.ceil(len(dataset.sequences) / config.batch_size)
        probability = 1.0 - config.text_ratio if has_texts else 1.0
        if probability > 0.0:
            estimates.append(math.ceil(question_batches / probability))
    if has_texts:
        text_windows = sum(
            max(1, math.ceil((len(sequence) - 1) / config.context_length))
            for sequence in dataset.text_sequences
        )
        text_batches = math.ceil(text_windows / config.batch_size)
        probability = config.text_ratio if has_questions else 1.0
        if probability > 0.0:
            estimates.append(math.ceil(text_batches / probability))
    return max(1, *estimates)


def validation_loss(
    model: DecoderTransformer, dataset: TokenDataset, config: TransformerConfig,
) -> float:
    """Computes deterministic, source-weighted validation loss."""
    total = 0.0
    with no_grad():
        for source, weight in dataset.source_weights():
            inputs, targets, loss_weights = dataset.deterministic_batch_with_loss_weights(
                config.batch_size, config.context_length, source=source
            )
            logits = model(inputs)
            loss = logits.reshape(-1, config.vocab_size).cross_entropy(
                flatten(targets), weights=flatten(loss_weights)  # type: ignore[arg-type]
            ).item()
            total += weight * loss
    return total


def train_model(
    config: TransformerConfig,
    *,
    base_dir: str | Path = ".",
    output_dir: str | Path = "weights",
    resume: str | Path | None = None,
) -> TrainingResult:
    """Trains from configured project data and exports standard reusable weights."""
    project_dir = Path(base_dir).resolve()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = project_dir / destination
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "training_checkpoint.bin"

    print("\nmimiLLM training", flush=True)
    print(f"Preparing datasets from {project_dir} ...", end=" ", flush=True)
    train_data, validation_data = _datasets(config, project_dir)
    print("done", flush=True)
    print("Building model and optimizer ...", end=" ", flush=True)
    model = DecoderTransformer(config)
    optimizer = AdamW(
        model.parameters(), config.learning_rate, weight_decay=config.weight_decay
    )
    print("done", flush=True)
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

    backend = get_backend()
    backend_name = getattr(backend, "name", "python")
    threads = getattr(backend, "num_threads", 1)
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
        f"Run: {start_step}->{config.steps} steps | epochs={total_epochs} | "
        f"batches/epoch={batches_per_epoch} | batch_size={config.batch_size} | "
        f"backend={backend_name} | threads={threads} | output={destination}",
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
                step, config.steps, config.learning_rate, config.warmup_steps
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
            gradient_norm = optimizer.clip_grad_norm(1.0)
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
                or ends_epoch
                or step == config.steps
            )
            should_checkpoint = (
                step % config.checkpoint_interval == 0
                or ends_epoch
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
                latest_validation_loss = validation_loss(model, validation_data, config)
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
                save_model(destination, model)
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
        save_model(destination, model)
        print(f"Training complete: {last_step}/{config.steps} steps", flush=True)
    return TrainingResult(model, destination, checkpoint_path, last_step, interrupted)


def train_from_config(
    config_path: str | Path = "config.json",
    *,
    output_dir: str | Path = "weights",
    resume: str | Path | None = None,
    steps: int | None = None,
    batches_per_epoch: int | None = None,
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
    return train_model(
        config,
        base_dir=path.parent,
        output_dir=output_dir,
        resume=resume,
    )


def main(
    default_config: str | Path = "config.json",
    default_output_dir: str | Path = "weights",
) -> None:
    parser = argparse.ArgumentParser(description="Train a mimiLLM model")
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--output-dir", type=Path, default=Path(default_output_dir))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--batches-per-epoch", type=int,
        help="number of optimizer batches grouped into one displayed epoch",
    )
    args = parser.parse_args()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute()
        else (Path.cwd() / args.output_dir).resolve()
    )
    resume = args.resume
    if resume is not None and not resume.is_absolute():
        resume = (Path.cwd() / resume).resolve()
    result = train_from_config(
        args.config,
        output_dir=output_dir,
        resume=resume,
        steps=args.steps,
        batches_per_epoch=args.batches_per_epoch,
    )
    print(f"Weights: {result.weights_dir}")
    print(f"Training checkpoint: {result.checkpoint_path}")
