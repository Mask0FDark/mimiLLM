"""End-to-end acceptance test for answer-only SFT."""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from mimillm.diagnostics import DEFAULT_ANSWER, run_one_pair_sft_acceptance


class OnePairSftAcceptanceTests(unittest.TestCase):
    def test_python_backend_overfits_and_survives_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                report = run_one_pair_sft_acceptance(
                    backend="python",
                    steps=100,
                    output_dir=Path(directory),
                    fresh_process=True,
                )
            self.assertTrue(report["mask_passed"])
            self.assertLess(report["final_loss"], report["initial_loss"])
            self.assertIn(DEFAULT_ANSWER, report["response_after"])
            self.assertIn(
                DEFAULT_ANSWER, report["response_reloaded_in_fresh_process"],
            )
            self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
