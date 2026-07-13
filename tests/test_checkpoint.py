"""Roundtrip и защита собственного checkpoint-формата."""

import os
import json
import struct
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MIMILLM_BACKEND", "python")

from mimillm.checkpoint import load_checkpoint, save_checkpoint
from mimillm.optim import AdamW
from mimillm.transformer import DecoderTransformer, TransformerConfig


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TransformerConfig(
            context_length=4, d_model=4, n_layers=1, n_heads=1, d_mlp=8,
            batch_size=1, steps=2, validation_interval=1, checkpoint_interval=1,
        )

    def test_roundtrip_preserves_logits_and_optimizer(self) -> None:
        model = DecoderTransformer(self.config)
        optimizer = AdamW(model.parameters(), 0.01)
        loss = model([[257, 1, 2, 3]]).reshape(-1, 260).cross_entropy([1, 2, 3, 258])
        loss.backward()
        optimizer.step()
        expected = model([[257, 1, 2, 3]]).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            save_checkpoint(path, model, optimizer, config=self.config.to_dict(), step=1, seed=42)
            restored = DecoderTransformer(self.config)
            restored_optimizer = AdamW(restored.parameters(), 0.5)
            data = load_checkpoint(path, restored, restored_optimizer)
            actual = restored([[257, 1, 2, 3]]).detach()
        self.assertEqual(data.step, 1)
        self.assertEqual(optimizer.step_count, restored_optimizer.step_count)
        self.assertEqual(expected.data, actual.data)

    def test_corruption_is_detected(self) -> None:
        model = DecoderTransformer(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            save_checkpoint(path, model, None, config=self.config.to_dict(), step=0, seed=1)
            content = path.read_bytes()
            path.write_bytes(content[:-7])
            with self.assertRaisesRegex(ValueError, "закончился"):
                load_checkpoint(path)

    def test_bad_magic_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            path.write_bytes(b"NOTMODEL" + b"\0" * 20)
            with self.assertRaisesRegex(ValueError, "magic"):
                load_checkpoint(path)

    def test_unknown_metadata_format_is_detected(self) -> None:
        model = DecoderTransformer(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            save_checkpoint(path, model, None, config=self.config.to_dict(), step=0, seed=1)
            raw = path.read_bytes()
            header = struct.Struct("<8sIQ")
            magic, version, size = header.unpack(raw[:header.size])
            metadata = json.loads(raw[header.size:header.size + size])
            metadata["format"] = "unknown"
            encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
            path.write_bytes(
                header.pack(magic, version, len(encoded)) + encoded + raw[header.size + size:]
            )
            with self.assertRaisesRegex(ValueError, "формат metadata"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
