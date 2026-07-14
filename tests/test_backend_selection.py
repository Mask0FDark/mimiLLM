"""Проверки выбора backend и безопасного fallback без библиотеки."""

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mimillm import backend
from mimillm.backend_cpp import default_library_path


class BackendSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        backend.reset_backend()

    def tearDown(self) -> None:
        backend.reset_backend()

    def test_python_can_be_selected_explicitly(self) -> None:
        with patch.dict(os.environ, {"MIMILLM_BACKEND": "python"}):
            self.assertEqual(backend.get_backend().name, "python")

    def test_auto_falls_back_when_library_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-library"
            with patch.dict(os.environ, {"MIMILLM_BACKEND": "auto", "MIMILLM_DISABLE_CUDA": "1"}):
                with patch("mimillm.backend_cpp.default_library_path", return_value=missing):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        selected = backend.get_backend()
        self.assertEqual(selected.name, "python")
        self.assertTrue(any("using Python" in str(item.message) for item in caught))

    def test_explicit_cpp_reports_missing_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-library"
            with patch.dict(os.environ, {"MIMILLM_BACKEND": "cpp"}):
                with patch("mimillm.backend_cpp.default_library_path", return_value=missing):
                    with self.assertRaisesRegex(RuntimeError, "unavailable"):
                        backend.get_backend()

    def test_auto_prefers_cuda(self) -> None:
        expected = SimpleNamespace(name="cuda")
        with patch.dict(os.environ, {"MIMILLM_BACKEND": "auto"}):
            with patch("mimillm.backend_cuda.CudaBackend", return_value=expected):
                self.assertIs(backend.get_backend(), expected)

    def test_explicit_cuda_reports_initialization_error(self) -> None:
        with patch.dict(os.environ, {"MIMILLM_BACKEND": "cuda"}):
            with patch(
                "mimillm.backend_cuda.CudaBackend",
                side_effect=FileNotFoundError("missing NVRTC"),
            ):
                with self.assertRaisesRegex(RuntimeError, "CUDA backend is unavailable"):
                    backend.get_backend()

    def test_explicit_library_path_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "custom-backend.dll"
            with patch.dict(os.environ, {"MIMILLM_BACKEND_LIBRARY": str(expected)}):
                self.assertEqual(default_library_path(), expected.resolve())


if __name__ == "__main__":
    unittest.main()
