#!/usr/bin/env python3
"""Интерактивный терминальный чат с обученной моделью m0fdii."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from minillm.checkpoint import load_checkpoint
from minillm.generation import answer_question
from minillm.tokenizer import ByteTokenizer
from minillm.transformer import DecoderTransformer, TransformerConfig


def main() -> None:
    # Случайные byte-токены могут дать U+FFFD, которого нет в старых Windows codepage.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Чат с m0fdii")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stored = load_checkpoint(args.checkpoint)
    config = TransformerConfig.from_dict(stored.config)
    model = DecoderTransformer(config).eval()
    load_checkpoint(args.checkpoint, model)
    tokenizer = ByteTokenizer()
    rng = random.Random(args.seed)
    history: list[tuple[str, str]] = []
    print("m0fdii готова. Для выхода напишите /exit.")
    while True:
        try:
            question = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break
        if not question:
            continue
        if question == "/exit":
            print("До свидания!")
            break
        if question == "/help":
            print("Команды: /exit, /help, /reset, /settings")
            continue
        if question == "/reset":
            history.clear()
            print("История диалога очищена; веса модели не изменены.")
            continue
        if question == "/settings":
            print(
                f"temperature={args.temperature}, top_k={args.top_k}, "
                f"max_new_tokens={args.max_new_tokens}, context={config.context_length}"
            )
            continue
        answer = answer_question(
            model, tokenizer, question, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, rng=rng,
        )
        history.append((question, answer))
        print(f"m0fdii: {answer or '[пустая генерация]'}")


if __name__ == "__main__":
    main()
