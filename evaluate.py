#!/usr/bin/env python3
"""Оценивает checkpoint на отдельном файле без обновления весов."""

from __future__ import annotations

import argparse
from pathlib import Path

from minillm.checkpoint import load_checkpoint
from minillm.dataset import TokenDataset
from minillm.tensor import no_grad
from minillm.transformer import DecoderTransformer, TransformerConfig
from minillm.utils import flatten


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation m0fdii")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--text-data", type=Path, action="append", metavar="PATH",
        help="UTF-8 файл или каталог validation-текстов; можно повторять",
    )
    parser.add_argument("--text-ratio", type=float, help="вес text loss от 0 до 1")
    args = parser.parse_args()
    stored = load_checkpoint(args.checkpoint)
    config = TransformerConfig.from_dict(stored.config)
    model = DecoderTransformer(config).eval()
    load_checkpoint(args.checkpoint, model)
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
