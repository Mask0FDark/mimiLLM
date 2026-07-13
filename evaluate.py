#!/usr/bin/env python3
"""Оценивает сохранённую модель на отдельных данных без обновления весов."""

from __future__ import annotations

import argparse
from pathlib import Path

from mimillm.api import load_model
from mimillm.dataset import TokenDataset
from mimillm.tensor import no_grad
from mimillm.utils import flatten


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation mimiLLM")
    parser.add_argument(
        "--model", type=Path, required=True,
        help="папка модели, .safetensors или старый .bin checkpoint",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--text-data", type=Path, action="append", metavar="PATH",
        help="UTF-8 файл или каталог validation-текстов; можно повторять",
    )
    parser.add_argument("--text-ratio", type=float, help="вес text loss от 0 до 1")
    args = parser.parse_args()
    model = load_model(args.model)
    config = model.config
    text_ratio = config.text_ratio if args.text_ratio is None else args.text_ratio
    if not 0.0 <= text_ratio <= 1.0:
        parser.error("--text-ratio должен быть от 0 до 1")
    dataset = TokenDataset(args.data, text_paths=args.text_data, text_ratio=text_ratio)
    total_loss = 0.0
    details: list[str] = []
    with no_grad():
        for source, weight in dataset.source_weights():
            inputs, targets = dataset.deterministic_batch(
                config.batch_size, config.context_length, source=source
            )
            loss = model(inputs).reshape(-1, config.vocab_size).cross_entropy(flatten(targets)).item()
            total_loss += weight * loss
            details.append(f"{source}_loss={loss:.6f}")
    print(
        f"validation_loss={total_loss:.6f} {' '.join(details)} "
        f"qa_examples={len(dataset.examples)} text_documents={len(dataset.text_documents)} "
        f"tokens={len(dataset.tokens)}"
    )


if __name__ == "__main__":
    main()
