#!/usr/bin/env python3
"""Интерактивный терминальный чат с обученной моделью mimiLLM."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from mimillm.api import load_model
from mimillm.generation import answer_question
from mimillm.tokenizer import ByteTokenizer


def main() -> None:
    # Случайные byte-токены могут дать U+FFFD, которого нет в старых Windows codepage.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Чат с mimiLLM")
    parser.add_argument(
        "--model", type=Path, required=True,
        help="папка с config.json и model.safetensors (или старый .bin checkpoint)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = load_model(args.model)
    config = model.config
    tokenizer = ByteTokenizer()
    rng = random.Random(args.seed)
    history: list[tuple[str, str]] = []
    print("mimiLLM готова. Для выхода напишите /exit.")
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
        print(f"mimiLLM: {answer or '[пустая генерация]'}")


if __name__ == "__main__":
    main()
