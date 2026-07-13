"""High-level training workflow for projects that use mimiLLM as a library."""

from __future__ import annotations

import argparse
import random
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


def validation_loss(
    model: DecoderTransformer, dataset: TokenDataset, config: TransformerConfig,
) -> float:
    """Computes deterministic, source-weighted validation loss."""
    total = 0.0
    with no_grad():
        for source, weight in dataset.source_weights():
            inputs, targets = dataset.deterministic_batch(
                config.batch_size, config.context_length, source=source
            )
            logits = model(inputs)
            loss = logits.reshape(-1, config.vocab_size).cross_entropy(flatten(targets)).item()
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

    train_data, validation_data = _datasets(config, project_dir)
    model = DecoderTransformer(config)
    optimizer = AdamW(
        model.parameters(), config.learning_rate, weight_decay=config.weight_decay
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
        print(f"Resume from step {start_step}: {resume_path}")

    backend = get_backend()
    backend_name = getattr(backend, "name", "python")
    threads = getattr(backend, "num_threads", 1)
    print(
        f"mimiLLM: parameters={model.parameter_count()}, "
        f"backend={backend_name}, threads={threads}"
    )
    print(
        f"data: questions={len(train_data.examples)}, "
        f"texts={len(train_data.text_documents)}, text_ratio={config.text_ratio:.2f}"
    )

    last_step = start_step
    last_checkpoint_step = -1
    interrupted = False
    try:
        for step in range(start_step + 1, config.steps + 1):
            started = time.perf_counter()
            inputs, targets = train_data.sample_batch(
                config.batch_size, config.context_length, rng
            )
            optimizer.learning_rate = learning_rate_at(
                step, config.steps, config.learning_rate, config.warmup_steps
            )
            logits = model(inputs)
            loss = logits.reshape(-1, config.vocab_size).cross_entropy(flatten(targets))
            loss.backward()
            optimizer.clip_grad_norm(1.0)
            optimizer.step()
            optimizer.zero_grad()
            last_step = step
            elapsed = time.perf_counter() - started
            tokens = sum(len(row) for row in inputs)
            message = (
                f"step={step} train_loss={loss.item():.5f} "
                f"tok/s={tokens / elapsed:.1f} time={elapsed:.3f}s "
                f"lr={optimizer.learning_rate:.6g} source={train_data.last_source}"
            )
            if step % config.validation_interval == 0 or step == config.steps:
                message += (
                    f" validation_loss={validation_loss(model, validation_data, config):.5f}"
                )
            print(message, flush=True)
            if step % config.checkpoint_interval == 0 or step == config.steps:
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
    except KeyboardInterrupt:
        interrupted = True
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
        print(f"\nTraining interrupted. Saved: {checkpoint_path}")

    if not interrupted:
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
    return TrainingResult(model, destination, checkpoint_path, last_step, interrupted)


def train_from_config(
    config_path: str | Path = "config.json",
    *,
    output_dir: str | Path = "weights",
    resume: str | Path | None = None,
    steps: int | None = None,
) -> TrainingResult:
    """Loads JSON config; relative data paths are based on the config's directory."""
    path = Path(config_path).resolve()
    config = TransformerConfig.from_json(path)
    if steps is not None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        config = replace(config, steps=steps)
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
    )
    print(f"Weights: {result.weights_dir}")
    print(f"Training checkpoint: {result.checkpoint_path}")
