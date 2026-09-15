"""Проверки компактного mmap-формата token shard."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mimillm.token_shard import (
    HEADER,
    MAGIC,
    MappedTokenShard,
    discover_token_shards,
    load_token_shards,
    write_token_shard,
)


class TokenShardTests(unittest.TestCase):
    def test_uint16_roundtrip_and_slicing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.mmtok"
            expected = [257, 10, 20, 30, 258]
            write_token_shard(path, expected, vocab_size=8192)
            with MappedTokenShard(path, expected_vocab_size=8192) as shard:
                self.assertEqual(shard.item_size, 2)
                self.assertEqual(len(shard), len(expected))
                self.assertEqual(shard[0], expected[0])
                self.assertEqual(shard[-1], expected[-1])
                self.assertEqual(shard[1:4], expected[1:4])
                self.assertEqual(shard[::-1], list(reversed(expected)))

    def test_uint32_is_selected_for_large_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.mmtok"
            values = [1, 65536, 69999]
            write_token_shard(path, values, vocab_size=70000)
            with MappedTokenShard(path, expected_vocab_size=70000) as shard:
                self.assertEqual(shard.item_size, 4)
                self.assertEqual(shard[:], values)

    def test_writer_accepts_generator_without_prebuilding_a_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.mmtok"
            write_token_shard(
                path,
                (index % 100 for index in range(10000)),
                vocab_size=260,
                chunk_tokens=127,
            )
            with MappedTokenShard(path, expected_vocab_size=260) as shard:
                self.assertEqual(len(shard), 10000)
                self.assertEqual(shard[9999], 99)

    def test_vocabulary_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.mmtok"
            write_token_shard(path, [1, 2, 3], vocab_size=8192)
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                MappedTokenShard(path, expected_vocab_size=4096)

    def test_truncated_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.mmtok"
            path.write_bytes(HEADER.pack(MAGIC, 1, 2, 8192, 10) + b"\0\0")
            with self.assertRaisesRegex(ValueError, "размер token shard"):
                MappedTokenShard(path)

    def test_discovery_is_recursive_stable_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            first = root / "a.mmtok"
            second = nested / "b.mmtok"
            write_token_shard(first, [1, 2], vocab_size=260)
            write_token_shard(second, [3, 4], vocab_size=260)
            self.assertEqual(discover_token_shards(root), [first, second])
            shards = load_token_shards(root, expected_vocab_size=260)
            try:
                self.assertEqual([list(shard) for shard in shards], [[1, 2], [3, 4]])
            finally:
                for shard in shards:
                    shard.close()


if __name__ == "__main__":
    unittest.main()
