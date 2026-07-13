"""Проверки выбора backend и безопасного fallback без библиотеки."""

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from minillm import backend


class BackendSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        backend.reset_backend()

    def tearDown(self) -> None:
        backend.reset_backend()

    def test_python_can_be_selected_explicitly(self) -> None:
        with patch.dict(os.environ, {"MINILLM_BACKEND": "python"}):
            self.assertEqual(backend.get_backend().name, "python")

    def test_auto_falls_back_when_library_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-library"
            with patch.dict(os.environ, {"MINILLM_BACKEND": "auto"}):
                with patch("minillm.backend_cpp.default_library_path", return_value=missing):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        selected = backend.get_backend()
        self.assertEqual(selected.name, "python")
        self.assertTrue(any("используется Python" in str(item.message) for item in caught))

    def test_explicit_cpp_reports_missing_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-library"
            with patch.dict(os.environ, {"MINILLM_BACKEND": "cpp"}):
                with patch("minillm.backend_cpp.default_library_path", return_value=missing):
                    with self.assertRaisesRegex(RuntimeError, "недоступен"):
                        backend.get_backend()


if __name__ == "__main__":
    unittest.main()
