"""End-to-end smoke test потоковой подготовки `.mmtok` корпуса."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mimillm.dataset import TokenDataset
from mimillm.token_shard import MappedTokenShard, tokenizer_fingerprint
from mimillm.tokenizer import ByteTokenizer
from tools.tokenize_corpus import build_shards


class TokenizeCorpusTests(unittest.TestCase):
    def test_build_shards_and_reload_through_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "data" / "text" / "train"
            validation = root / "data" / "text" / "validation"
            train.mkdir(parents=True)
            validation.mkdir(parents=True)
            (train / "a.txt").write_text(
                "Первый документ. Второй кусок текста. Третий фрагмент.",
                encoding="utf-8",
            )
            (validation / "v.txt").write_text(
                "Проверочный документ.", encoding="utf-8"
            )
            config = {
                "vocab_size": 260,
                "tokenizer": "byte",
                "tie_word_embeddings": True,
                "context_length": 16,
                "d_model": 8,
                "n_layers": 1,
                "n_heads": 2,
                "d_mlp": 16,
                "batch_size": 1,
                "steps": 2,
                "validation_interval": 1,
                "checkpoint_interval": 1,
                "text_ratio": 1.0,
                "text_train_path": "data/text/train",
                "text_validation_path": "data/text/validation",
                "question_train_path": "unused/train",
                "question_validation_path": "unused/validation"
            }
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(config, ensure_ascii=False), encoding="utf-8"
            )
            output = root / "tokens" / "train"
            manifest = build_shards(
                config_path,
                split="train",
                output_dir=output,
                shard_tokens=20,
                chunk_characters=12,
                force=False,
            )
            self.assertGreater(int(manifest["tokens"]), 20)
            self.assertGreater(len(manifest["shards"]), 1)
            self.assertEqual(
                manifest["tokenizer_sha256"],
                tokenizer_fingerprint(ByteTokenizer()),
            )
            self.assertTrue((output / "manifest.json").is_file())
            for descriptor in manifest["shards"]:
                path = output / str(descriptor["file"])
                with MappedTokenShard(
                    path,
                    expected_vocab_size=260,
                    expected_tokenizer_sha256=str(manifest["tokenizer_sha256"]),
                ) as shard:
                    self.assertGreaterEqual(len(shard), 2)

            dataset = TokenDataset(
                tokenizer=ByteTokenizer(), text_paths=output, text_ratio=1.0
            )
            try:
                self.assertEqual(dataset.text_storage, "shard")
                self.assertEqual(dataset.text_tokens, manifest["tokens"])
            finally:
                dataset.close()

    def test_non_empty_output_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            validation = root / "validation"
            train.mkdir()
            validation.mkdir()
            (train / "a.txt").write_text("abc", encoding="utf-8")
            (validation / "v.txt").write_text("xyz", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({
                    "vocab_size": 260,
                    "tokenizer": "byte",
                    "tie_word_embeddings": True,
                    "context_length": 8,
                    "d_model": 8,
                    "n_layers": 1,
                    "n_heads": 2,
                    "d_mlp": 16,
                    "batch_size": 1,
                    "steps": 2,
                    "validation_interval": 1,
                    "checkpoint_interval": 1,
                    "text_ratio": 1.0,
                    "text_train_path": "train",
                    "text_validation_path": "validation",
                    "question_train_path": "unused/train",
                    "question_validation_path": "unused/validation"
                }),
                encoding="utf-8",
            )
            output = root / "tokens"
            output.mkdir()
            (output / "keep.txt").write_text("не удалять", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_shards(
                    config_path,
                    split="train",
                    output_dir=output,
                    shard_tokens=10,
                    chunk_characters=10,
                    force=False,
                )


if __name__ == "__main__":
    unittest.main()
