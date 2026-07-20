"""Small, reproducible diagnostics for the training path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .api import load_model
from .backend import reset_backend
from .dataset import TokenDataset
from .generation import generate_response
from .tokenizer import format_qa_text, save_tokenizer, train_bpe_tokenizer
from .training import train_model, validation_loss
from .transformer import DecoderTransformer, TransformerConfig


DEFAULT_QUESTION = "Сколько дней в неделе?"
DEFAULT_ANSWER = "В неделе семь дней."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(directory: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=directory,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=directory,
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _write_json(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(values, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _response(weights: Path, question: str, backend: str) -> str:
    os.environ["MIMILLM_BACKEND"] = backend
    reset_backend()
    model = load_model(weights)
    return generate_response(
        model, question, max_new_tokens=32, temperature=0.0, top_k=1,
    )


def _response_in_fresh_process(weights: Path, question: str, backend: str) -> str:
    completed = subprocess.run(
        [
            sys.executable, "-m", "mimillm.diagnostics", "--load-only",
            str(weights), "--question", question, "--backend", backend,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return str(json.loads(completed.stdout)["response"])


def run_one_pair_sft_acceptance(
    *,
    backend: str = "python",
    steps: int = 100,
    output_dir: str | Path | None = None,
    fresh_process: bool = True,
) -> dict[str, Any]:
    """Overfits one QA pair and returns a machine-readable acceptance report."""
    if backend not in {"python", "cpp", "cuda", "auto"}:
        raise ValueError("backend must be python, cpp, cuda, or auto")
    if steps <= 0:
        raise ValueError("steps must be positive")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="mimillm-one-pair-")
        root = Path(temporary.name)
    else:
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)

    try:
        question = DEFAULT_QUESTION
        answer = DEFAULT_ANSWER
        qa_text = f"Вопрос: {question}\nОтвет: {answer}\n"
        for split in ("train", "validation"):
            path = root / "data" / "question" / split / "one_pair.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(qa_text, encoding="utf-8")

        formatted = format_qa_text(question, answer)
        tokenizer = train_bpe_tokenizer(
            [formatted], vocab_size=300, min_frequency=1,
        )
        tokenizer_path = save_tokenizer(tokenizer, root / "tokenizer.json")
        config = TransformerConfig(
            vocab_size=tokenizer.VOCAB_SIZE,
            tokenizer="bpe",
            tokenizer_path=str(tokenizer_path),
            context_length=64,
            d_model=8,
            n_layers=1,
            n_heads=1,
            d_mlp=16,
            batch_size=1,
            batches_per_epoch=steps,
            steps=steps,
            learning_rate=0.02,
            weight_decay=0.0,
            gradient_clip_norm=10.0,
            learning_rate_schedule="constant",
            min_learning_rate_ratio=1.0,
            warmup_steps=0,
            validation_interval=max(1, steps // 5),
            checkpoint_interval=steps,
            seed=42,
            text_ratio=0.0,
            qa_prompt_weight=0.0,
            question_train_path="data/question/train",
            question_validation_path="data/question/validation",
        )
        dataset = TokenDataset(
            root / "data" / "question" / "train",
            tokenizer=tokenizer,
            text_ratio=0.0,
            qa_prompt_weight=0.0,
        )
        inputs, targets, weights = dataset.deterministic_batch_with_loss_weights(
            1, config.context_length, source="qa",
        )
        sequence = dataset.sequences[0]
        answer_start = dataset.qa_answer_starts[id(sequence)]
        first_answer_target_index = answer_start - 1
        eos_target_index = len(sequence) - 2
        padding_indices = [
            index for index, token in enumerate(targets[0])
            if token == tokenizer.PAD
        ]
        mask_ok = (
            all(weight == 0.0 for weight in weights[0][:first_answer_target_index])
            and weights[0][first_answer_target_index] > 0.0
            and targets[0][eos_target_index] == tokenizer.EOS
            and weights[0][eos_target_index] > 0.0
            and all(weights[0][index] == 0.0 for index in padding_indices)
        )

        os.environ["MIMILLM_BACKEND"] = backend
        reset_backend()
        initial_model = DecoderTransformer(config, tokenizer_model=tokenizer)
        initial_loss = validation_loss(initial_model, dataset, config)
        prompt_ids = tokenizer.encode_prompt(question)
        initial_logits = list(initial_model([prompt_ids]).data)
        before = generate_response(
            initial_model, question, max_new_tokens=32,
            temperature=0.0, top_k=1,
        )
        result = train_model(config, base_dir=root, output_dir=root / "weights")
        trained = load_model(result.weights_dir)
        trained_logits = list(trained([prompt_ids]).data)
        in_memory_logits = list(result.model([prompt_ids]).data)
        max_logit_change = max(
            abs(after_value - before_value)
            for before_value, after_value in zip(initial_logits, trained_logits)
        )
        max_save_load_logit_difference = max(
            abs(after_value - loaded_value)
            for after_value, loaded_value in zip(in_memory_logits, trained_logits)
        )
        max_parameter_change = max(
            abs(after_value - before_value)
            for initial_parameter, trained_parameter in zip(
                initial_model.parameters(), trained.parameters(),
            )
            for before_value, after_value in zip(
                initial_parameter.data, trained_parameter.data,
            )
        )
        final_loss = validation_loss(trained, dataset, config)
        after = generate_response(
            trained, question, max_new_tokens=32, temperature=0.0, top_k=1,
        )
        reloaded = (
            _response_in_fresh_process(result.weights_dir, question, backend)
            if fresh_process else _response(result.weights_dir, question, backend)
        )
        expected_in_after = answer in after
        expected_in_reloaded = answer in reloaded
        passed = (
            mask_ok
            and final_loss < initial_loss
            and max_logit_change > 0.0
            and max_parameter_change > 0.0
            and max_save_load_logit_difference <= 1e-7
            and expected_in_after
            and expected_in_reloaded
        )
        report: dict[str, Any] = {
            "format": "mimiLLM-one-pair-sft-acceptance-v1",
            "passed": passed,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "backend": backend,
                "git": _git_state(Path(__file__).resolve().parents[1]),
            },
            "seed": config.seed,
            "steps": result.step,
            "config": config.to_dict(),
            "architecture_sha256": hashlib.sha256(json.dumps(
                {
                    key: config.to_dict()[key]
                    for key in (
                        "vocab_size", "tokenizer", "tie_word_embeddings",
                        "context_length", "d_model", "n_layers", "n_heads",
                        "d_mlp",
                    )
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
            "question": question,
            "expected_answer": answer,
            "formatted_text": formatted,
            "tokenizer_sha256": _sha256(tokenizer_path),
            "model_sha256": _sha256(result.weights_dir / "model.safetensors"),
            "token_ids": sequence,
            "prompt_ids": prompt_ids,
            "input_ids": inputs[0],
            "target_ids": targets[0],
            "decoded_target_tokens": [
                {
                    "id": token,
                    "text": tokenizer.decode([token]),
                    "bytes_hex": (
                        tokenizer.token_bytes(token).hex()
                        if tokenizer.token_bytes(token) is not None else None
                    ),
                }
                for token in targets[0]
            ],
            "loss_weights": weights[0],
            "first_answer_target_index": first_answer_target_index,
            "eos_id": tokenizer.EOS,
            "eos_target_index": eos_target_index,
            "mask_passed": mask_ok,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "max_logit_change": max_logit_change,
            "max_parameter_change": max_parameter_change,
            "max_save_load_logit_difference": max_save_load_logit_difference,
            "response_before": before,
            "response_after": after,
            "response_reloaded_in_fresh_process": reloaded,
            "weights": str(result.weights_dir),
        }
        _write_json(root / "one_pair_sft_report.json", report)
        return report
    finally:
        if temporary is not None:
            temporary.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify one-pair SFT masking, overfit, and save/load",
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "cpp", "python"), default="python")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("mimillm_diagnostics/one_pair_sft"),
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION, help=argparse.SUPPRESS)
    parser.add_argument("--load-only", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.load_only is not None:
        print(json.dumps({
            "response": _response(args.load_only, args.question, args.backend),
        }, ensure_ascii=False))
        return
    report = run_one_pair_sft_acceptance(
        backend=args.backend, steps=args.steps, output_dir=args.output_dir,
    )
    print(json.dumps({
        "passed": report["passed"],
        "initial_loss": report["initial_loss"],
        "final_loss": report["final_loss"],
        "response": report["response_after"],
        "reloaded_response": report["response_reloaded_in_fresh_process"],
        "report": str(args.output_dir.resolve() / "one_pair_sft_report.json"),
    }, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
