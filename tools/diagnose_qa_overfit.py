#!/usr/bin/env python3
"""Check whether mimiLLM can exactly memorize one QA example.

This is a diagnostic, not a useful training recipe. A decoder-only model that
cannot overfit one short sequence has a bug in its data, loss, autograd, or
optimizer path. Run it before spending hours on a real corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mimillm import generate_response
from mimillm.optim import AdamW
from mimillm.tokenizer import load_tokenizer
from mimillm.transformer import DecoderTransformer, TransformerConfig


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--tokenizer", type=Path, required=True)
    result.add_argument("--steps", type=int, default=500)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--report-every", type=int, default=50)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.steps <= 0 or args.report_every <= 0 or args.learning_rate <= 0.0:
        raise ValueError("steps, report-every, and learning-rate must be positive")

    original = TransformerConfig.from_json(args.config)
    tokenizer = load_tokenizer(args.tokenizer)
    config_values = original.to_dict()
    config_values.update(
        steps=args.steps,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        weight_decay=0.0,
    )
    config = TransformerConfig.from_dict(config_values)
    model = DecoderTransformer(config, tokenizer_model=tokenizer)
    optimizer = AdamW(
        model.parameters(), learning_rate=args.learning_rate, weight_decay=0.0,
    )

    question = "Как называется столица Франции?"
    answer = "Столица Франции — Париж."
    sequence = tokenizer.encode_qa(question, answer)
    if len(sequence) - 1 > config.context_length:
        raise ValueError(
            f"diagnostic sequence needs {len(sequence) - 1} context tokens, "
            f"but the model has {config.context_length}"
        )
    answer_start = len(tokenizer.encode_prompt(question))
    inputs = [sequence[:-1]]
    targets = sequence[1:]
    weights = [
        0.0 if target_position < answer_start else 1.0
        for target_position in range(1, len(sequence))
    ]

    initial_loss = None
    final_loss = None
    for step in range(1, args.steps + 1):
        logits = model(inputs).reshape(-1, config.vocab_size)
        loss = logits.cross_entropy(targets, weights=weights)
        value = loss.item()
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        loss.backward()
        gradient_norm = optimizer.clip_grad_norm(1.0)
        optimizer.step()
        optimizer.zero_grad()
        if step == 1 or step % args.report_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {"step": step, "loss": value, "gradient_norm": gradient_norm}
                ),
                flush=True,
            )

    generated = generate_response(
        model, question, max_new_tokens=64, temperature=0.0, top_k=1,
    )
    print(
        json.dumps(
            {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "question": question,
                "expected": answer,
                "generated": generated,
                "exact_match": generated == answer,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
