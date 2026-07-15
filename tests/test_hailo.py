"""Tests for optional HailoRT discovery without requiring Hailo hardware."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from mimillm.hailo import (
    hailo_is_available,
    inspect_hailo_hef,
    inspect_hailo_runtime,
)


class FakeDevice:
    @staticmethod
    def scan() -> list[str]:
        return ["0000:01:00.0"]


class NamedInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeHef:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_network_group_names(self) -> list[str]:
        return ["group"]

    def get_networks_names(self) -> list[str]:
        return ["network"]

    def get_input_vstream_infos(self) -> list[NamedInfo]:
        return [NamedInfo("tokens")]

    def get_output_vstream_infos(self) -> list[NamedInfo]:
        return [NamedInfo("logits")]


class HailoTests(unittest.TestCase):
    def fake_platform(self) -> types.ModuleType:
        module = types.ModuleType("hailo_platform")
        module.__version__ = "4.20.0"
        module.Device = FakeDevice
        module.HEF = FakeHef
        return module

    def test_missing_runtime_is_reported_without_import_failure(self) -> None:
        with patch.dict(sys.modules, {"hailo_platform": None}):
            info = inspect_hailo_runtime()
            self.assertFalse(info.available)
            self.assertEqual(info.device_ids, ())
            self.assertIn("unavailable", info.error or "")

    def test_runtime_and_device_are_discovered(self) -> None:
        with patch.dict(sys.modules, {"hailo_platform": self.fake_platform()}):
            info = inspect_hailo_runtime()
            self.assertTrue(info.available)
            self.assertEqual(info.runtime_version, "4.20.0")
            self.assertEqual(info.device_ids, ("0000:01:00.0",))
            self.assertTrue(hailo_is_available())
            self.assertEqual(info.to_dict()["device_ids"], ["0000:01:00.0"])

    def test_hef_metadata_is_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.hef"
            path.write_bytes(b"test")
            with patch.dict(sys.modules, {"hailo_platform": self.fake_platform()}):
                info = inspect_hailo_hef(path)
            self.assertEqual(info.network_groups, ("group",))
            self.assertEqual(info.networks, ("network",))
            self.assertEqual(info.inputs, ("tokens",))
            self.assertEqual(info.outputs, ("logits",))

    def test_missing_hef_is_rejected_before_loading_runtime(self) -> None:
        with self.assertRaises(FileNotFoundError):
            inspect_hailo_hef("missing.hef")


if __name__ == "__main__":
    unittest.main()
