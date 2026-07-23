"""Tests for the deterministic training benchmark harness."""

import unittest

from mimillm.backend_cuda import is_available as cuda_is_available
from tools.benchmark_training import benchmark_training_step, training_step_snapshot


class TrainingBenchmarkTests(unittest.TestCase):
    def test_snapshot_is_deterministic_for_python_backend(self) -> None:
        first = training_step_snapshot(
            backend="python", context_length=4, d_model=4, n_heads=1, d_mlp=8,
        )
        second = training_step_snapshot(
            backend="python", context_length=4, d_model=4, n_heads=1, d_mlp=8,
        )
        self.assertEqual(first["backend"], "python")
        self.assertEqual(first["tokens"], 8)
        self.assertEqual(first["parameters"], second["parameters"])
        self.assertAlmostEqual(first["loss"], second["loss"], places=7)
        self.assertAlmostEqual(first["grad_norm"], second["grad_norm"], places=7)
        self.assertAlmostEqual(
            first["parameter_checksum"], second["parameter_checksum"], places=6,
        )

    def test_benchmark_reports_timing_fields(self) -> None:
        result = benchmark_training_step(
            backend="python", repeats=2, warmup=0,
            context_length=4, d_model=4, n_heads=1, d_mlp=8,
        )
        self.assertEqual(result["repeats"], 2)
        self.assertEqual(result["warmup"], 0)
        self.assertGreater(result["seconds"], 0.0)
        self.assertGreater(result["tokens_per_second"], 0.0)
        self.assertGreaterEqual(result["mean_seconds"], result["best_seconds"])
        self.assertGreaterEqual(result["median_seconds"], result["best_seconds"])
        self.assertGreater(result["mean_tokens_per_second"], 0.0)
        self.assertGreater(result["median_tokens_per_second"], 0.0)
        self.assertEqual(len(result["samples_seconds"]), 2)
        self.assertEqual(
            set(result["mean_phases_seconds"]),
            {"forward", "loss", "backward", "clipping", "optimizer", "zero_grad"},
        )
        self.assertEqual(
            set(result["mean_phases_percent"]),
            set(result["mean_phases_seconds"]),
        )

    @unittest.skipUnless(cuda_is_available(), "CUDA backend is unavailable")
    def test_cuda_training_step_matches_cpp_backend(self) -> None:
        options = {
            "context_length": 8,
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "d_mlp": 16,
            "batch_size": 2,
        }
        expected = training_step_snapshot(backend="cpp", **options)
        actual = training_step_snapshot(backend="cuda", **options)
        self.assertAlmostEqual(actual["loss"], expected["loss"], places=4)
        self.assertAlmostEqual(actual["grad_norm"], expected["grad_norm"], places=4)
        self.assertAlmostEqual(
            actual["parameter_checksum"],
            expected["parameter_checksum"],
            places=3,
        )

    @unittest.skipUnless(cuda_is_available(), "CUDA backend is unavailable")
    def test_cuda_repeated_steps_release_autograd_device_buffers(self) -> None:
        result = benchmark_training_step(
            backend="cuda", repeats=12, warmup=2,
            context_length=8, d_model=8, n_layers=1, n_heads=2,
            d_mlp=16, batch_size=2,
        )
        stats = result["memory_stats"]
        self.assertIsNotNone(stats)
        self.assertLessEqual(
            stats["tensor_cache_entries"],
            stats["persistent_entries"] + 8,
        )
        self.assertLessEqual(stats["pool_bytes"], stats["pool_limit_bytes"])


if __name__ == "__main__":
    unittest.main()
