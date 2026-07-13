#!/usr/bin/env python3
"""Оценивает checkpoint на отдельном файле без обновления весов."""

from __future__ import annotations

import argparse
from pathlib import Path

from minillm.checkpoint import load_checkpoint
from minillm.dataset import TokenDataset
from minillm.transformer import DecoderTransformer, TransformerConfig
from minillm.utils import flatten


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation m0fdii")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    stored = load_checkpoint(args.checkpoint)
    config = TransformerConfig.from_dict(stored.config)
    model = DecoderTransformer(config).eval()
    load_checkpoint(args.checkpoint, model)
    dataset = TokenDataset(args.data)
    inputs, targets = dataset.deterministic_batch(config.batch_size, config.context_length)
    loss = model(inputs).reshape(-1, config.vocab_size).cross_entropy(flatten(targets))
    print(f"validation_loss={loss.item():.6f} examples={len(dataset.examples)} tokens={len(dataset.tokens)}")


if __name__ == "__main__":
    main()

