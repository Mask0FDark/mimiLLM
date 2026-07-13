"""Tests for standard model weights and the SafeTensors subset used by mimiLLM."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from mimillm import Tensor, load_safetensors, save_safetensors


class SafeTensorsTests(unittest.TestCase):
    def test_float32_round_trip_and_documented_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_safetensors(
                path,
                {"weight": Tensor([1.25, -2.5, 3.0, 4.0], (2, 2))},
                metadata={"format": "mimiLLM"},
            )
            raw = path.read_bytes()
            (header_size,) = struct.unpack("<Q", raw[:8])
            header = json.loads(raw[8:8 + header_size].decode("utf-8"))
            self.assertEqual(header["weight"]["dtype"], "F32")
            self.assertEqual(header["weight"]["shape"], [2, 2])
            self.assertEqual(header["weight"]["data_offsets"], [0, 16])
            restored, metadata = load_safetensors(path)
            self.assertEqual(restored["weight"].shape, (2, 2))
            self.assertEqual(restored["weight"].data.tolist(), [1.25, -2.5, 3.0, 4.0])
            self.assertEqual(metadata, {"format": "mimiLLM"})

    def test_trailing_data_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            save_safetensors(path, {"x": Tensor([1.0], (1,))})
            path.write_bytes(path.read_bytes() + b"extra")
            with self.assertRaisesRegex(ValueError, "trailing"):
                load_safetensors(path)


if __name__ == "__main__":
    unittest.main()
