"""
Unit tests for Gate 4: 560-Condition Synthetic Benchmark Matrix.
PS 26052 — Adaptive Defence ANC.
"""

import unittest
from pathlib import Path
import csv
import numpy as np

from src.dataset.synthetic_benchmark_matrix import (
    generate_benchmark_matrix,
    get_benchmark_pair,
    SPEAKER_PROFILES,
    NOISE_PROFILES,
    SNR_LEVELS,
)


class TestSyntheticBenchmarkMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta_csv = "data/benchmark_matrix/matrix_metadata.csv"
        cls.records = generate_benchmark_matrix(cls.meta_csv)

    def test_matrix_dimensions(self):
        """Matrix must have exactly 10 * 8 * 7 = 560 clips."""
        expected = len(SPEAKER_PROFILES) * len(NOISE_PROFILES) * len(SNR_LEVELS)
        self.assertEqual(len(self.records), 560)
        self.assertEqual(len(self.records), expected)

    def test_audio_synthesis_determinism(self):
        """Audio pair generation must be deterministic, 16 kHz, and finite."""
        clean1, prim1, ref1 = get_benchmark_pair(self.records[0], seed=42)
        clean2, prim2, ref2 = get_benchmark_pair(self.records[0], seed=42)

        self.assertEqual(len(clean1), 16000)
        self.assertEqual(len(prim1), 16000)
        self.assertEqual(len(ref1), 16000)

        # Determinism
        np.testing.assert_allclose(clean1, clean2, rtol=1e-5)
        np.testing.assert_allclose(prim1, prim2, rtol=1e-5)

        # Finite and clean
        self.assertFalse(np.isnan(clean1).any())
        self.assertFalse(np.isnan(prim1).any())
        self.assertFalse(np.isnan(ref1).any())

    def test_snr_gradient(self):
        """-10 dB clip must have higher noise power than +20 dB clip."""
        c_low, p_low, _ = get_benchmark_pair(self.records[0], seed=42)  # -10 dB
        c_high, p_high, _ = get_benchmark_pair(self.records[6], seed=42)  # +20 dB

        noise_pow_low = np.mean((p_low - c_low) ** 2)
        noise_pow_high = np.mean((p_high - c_high) ** 2)

        self.assertGreater(noise_pow_low, noise_pow_high * 100.0)

    def test_provenance_explicitly_labeled(self):
        """All entries must declare synthetic provenance."""
        for r in self.records:
            self.assertEqual(r["provenance"], "synthetic_procedural_mathematical")


if __name__ == "__main__":
    unittest.main()
