"""
Unit Tests for Gate 6: Causality and Anti-Cheating Verification.
PS 26052 — Adaptive Defence ANC.
"""

import unittest
import numpy as np

from src.streaming.causality_verifier import CausalityVerifier


class TestCausalityAudit(unittest.TestCase):
    def setUp(self):
        self.verifier = CausalityVerifier(sr=16000, frame_size=512, hop_size=128)

    def test_vss_nlms_is_strictly_causal(self):
        """VSS-NLMS must produce identical output for t <= T0 regardless of future t > T0."""
        rep = self.verifier.verify_dsp_filter_causality(n_samples=8000, t_cutoff=4000)
        self.assertTrue(rep["is_strictly_causal"])
        self.assertLess(rep["max_diff_past"], 1e-6)
        self.assertTrue(rep["future_is_distinct"])

    def test_streaming_engine_is_strictly_causal(self):
        """CausalStreamingEngine must not leak future frames into past outputs."""
        rep = self.verifier.verify_streaming_engine_causality(n_samples=8000, t_cutoff=4096)
        self.assertTrue(rep["is_strictly_causal"])
        self.assertLess(rep["max_diff_past"], 1e-5)


if __name__ == "__main__":
    unittest.main()
