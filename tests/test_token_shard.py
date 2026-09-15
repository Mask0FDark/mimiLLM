"""Проверки компактного mmap-формата token shard."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from mimillm.dataset import TokenDataset
from mimillm.token_shard import (
    HEADER,
    MAGIC,
    MappedTokenShard,
    discover_token_shards,
    load_token_shards,
    tokenizer_fingerprint,
    write_token_shard,
)
from mimillm.tokenizer import ByteTokenizer


class TokenShardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ByteTokenizer()
        self.fingerprint = tokenizer_fingerprint(self.tokenizer)

    def test_uint16_roundtrip_and_slicing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.mmtok"
            expected = [257, 10, 20, 30, 258]
            write_token_shard(
                path,
                expected,
                vocab_size=260,
                tokenizer_sha256=self.fingerprint,
            )
            with MappedTokenShard(
                path,
                expected_vocab_size=260,
                expected_tokenizer_sha256=self.fingerprint,
            ) as shard:
                self.assertEqual(shard.item_size, 2)
                self.assertEqual(len(shard), len(expected))
                self.assertEqual(shard.tokenizer_sha256, self.fingerprint)
                self.assertEqual(shard[0], expected[0])
                self.assertEqual(shard[-1], expected[-1])
                self.assertEqual(shard[1:4], expected[1:4])
                self.assertEqual(shard[::-1], list(reversed(expected)))

    def test_uint32_is_selected_for_large_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.mmtok"
            values = [1, 65536, 69999]
            fake_fingerprint = "ab" * 32
            write_token_shard(
                path,
                values,
                vocab_size=70000,
                tokenizer_sha256=fake_fingerprint,
            )
            with MappedTokenShard(
                path,
                expected_vocab_size=70000,
                expected_tokenizer_sha256=fake_fingerprint,
            ) as shard:
                self.assertEqual(shard.item_size, 4)
                self.assertEqual(shard[:], values)

    def test_writer_accepts_generator_without_prebuilding_a_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.mmtok"
            write_token_shard(
                path,
                (index % 100 for index in range(10000)),
                vocab_size=260,
                tokenizer_sha256=self.fingerprint,
                chunk_tokens=127,
            )
            with MappedTokenShard(
                path,
                expected_vocab_size=260,
                expected_tokenizer_sha256=self.fingerprint,
            ) as shard:
                self.assertEqual(len(shard), 10000)
                self.assertEqual(shard[9999], 99)

    def test_vocabulary_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.mmtok"
            write_token_shard(
                path,
                [1, 2, 3],
                vocab_size=260,
                tokenizer_sha256=self.fingerprint,
            )
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                MappedTokenShard(path, expected_vocab_size=4096)

    def test_tokenizer_fingerprint_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.mmtok"
            write_token_shard(
                path,
                [1, 2, 3],
                vocab_size=260,
                tokenizer_sha256=self.fingerprint,
            )
            with self.assertRaisesRegex(ValueError, "другим токенизатором"):
                MappedTokenShard(
                    path,
                    expected_vocab_size=260,
                    expected_tokenizer_sha256="00" * 32,
                )

    def test_truncated_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.mmtok"
            path.write_bytes(
                HEADER.pack(
                    MAGIC,
                    1,
                    2,
                    260,
                    10,
                    bytes.fromhex(self.fingerprint),
                ) + b"\0\0"
            )
            with self.assertRaisesRegex(ValueError, "размер token shard"):
                MappedTokenShard(path)

    def test_discovery_is_recursive_stable_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            first = root / "a.mmtok"
            second = nested / "b.mmtok"
            write_token_shard(
                first,
                [1, 2],
                vocab_size=260,
                tokenizer_sha256=self.fingerprint,
            )
            write_token_shard(
                second,
                [3, 4],
                vocab_size=260,
                tokenizer_sha256=self.fingerprint,
            )
            self.assertEqual(discover_token_shards(root), [first, second])
            shards = load_token_shards(
                root,
                expected_vocab_size=260,
                expected_tokenizer_sha256=self.fingerprint,
            )
            try:
                self.assertEqual([list(shard) for shard in shards], [[1, 2], [3, 4]])
            finally:
                for shard in shards:
                    shard.close()

    def test_token_dataset_samples_directly_from_mmap_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "train.mmtok"
            tokens = [257, *range(1, 65), 258]
            write_token_shard(
                path,
                tokens,
                vocab_size=260,
                tokenizer_sha256=self.fingerprint,
            )
            dataset = TokenDataset(
                tokenizer=self.tokenizer,
                text_paths=root,
                text_ratio=1.0,
            )
            try:
                self.assertEqual(dataset.text_storage, "shard")
                self.assertEqual(dataset.text_documents, [])
                self.assertEqual(dataset.text_tokens, len(tokens))
                self.assertFalse(hasattr(dataset, "tokens"))
                self.assertIsInstance(dataset.text_sequences[0], MappedTokenShard)
                self.assertEqual(
                    dataset.text_sequences[0].tokenizer_sha256,
                    self.fingerprint,
                )
                inputs, targets, weights = dataset.sample_batch_with_loss_weights(
                    1, 16, random.Random(7)
                )
                self.assertEqual(inputs[0][1:], targets[0][:-1])
                self.assertTrue(all(weight == 1.0 for weight in weights[0]))
                supervised = sum(
                    sum(sum(row) for row in batch_weights)
                    for _, _, batch_weights in dataset.validation_batches(
                        1, 16, source="text"
                    )
                )
                self.assertEqual(supervised, len(tokens) - 1)
            finally:
                dataset.close()

    def test_raw_and_mmap_files_cannot_be_mixed_implicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw.txt").write_text("обычный текст", encoding="utf-8")
            write_token_shard(
                root / "tokens.mmtok",
                [257, 1, 258],
                vocab_size=260,
                tokenizer_sha256=self.fingerprint,
            )
            with self.assertRaisesRegex(ValueError, "не должен смешивать"):
                TokenDataset(
                    tokenizer=self.tokenizer,
                    text_paths=root,
                    text_ratio=1.0,
                )


if __name__ == "__main__":
    unittest.main()
