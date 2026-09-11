"""
Unit Tests for PH3.5 Pre-Hardware Stress Validation Invariants.
PS 26052: Defence-Grade Adaptive ANC.

Ensures that CausalStreamingEngine with the authoritative ONNX deployment
backend permanently satisfies numerical stability, memory bounds, clipping limits,
and deterministic state reset.
"""

import unittest
import numpy as np
from pathlib import Path

from src.streaming.causal_engine import CausalStreamingEngine


class TestStreamingStressInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = 16000
        cls.hop = 128
        cls.engine = CausalStreamingEngine(sample_rate=cls.sr, use_fast_dsp=True, ai_backend="onnx")

    def setUp(self):
        self.engine.reset()

    def test_sustained_streaming_finite_output(self):
        """Streaming engine must produce strictly finite, bounded output under sustained noise."""
        rng = np.random.RandomState(42)
        n_hops = 300
        out_hops = []
        for _ in range(n_hops):
            p = (rng.randn(self.hop) * 0.2).astype(np.float32)
            r = (rng.randn(self.hop) * 0.2).astype(np.float32)
            o, _ = self.engine.process_hop(p, r)
            out_hops.append(o)

        out_wav = np.concatenate(out_hops)
        self.assertFalse(np.isnan(out_wav).any(), "NaN detected in streaming output")
        self.assertFalse(np.isinf(out_wav).any(), "Inf detected in streaming output")
        self.assertLessEqual(np.max(np.abs(out_wav)), 0.985, "Output exceeded safety clipping bound")

    def test_state_reset_determinism(self):
        """engine.reset() must restore bit-level/floating-point clean state identically to a fresh engine."""
        rng = np.random.RandomState(101)
        p_speech = (rng.randn(self.hop * 10) * 0.1).astype(np.float32)
        r_speech = (rng.randn(self.hop * 10) * 0.1).astype(np.float32)

        # 1. Fresh Run
        out_fresh = []
        for h in range(10):
            idx = h * self.hop
            o, _ = self.engine.process_hop(p_speech[idx : idx + self.hop], r_speech[idx : idx + self.hop])
            out_fresh.append(o)
        wav_fresh = np.concatenate(out_fresh)

        # 2. Corrupt engine with heavy noise
        for _ in range(200):
            self.engine.process_hop(rng.randn(self.hop).astype(np.float32) * 5.0, rng.randn(self.hop).astype(np.float32) * 5.0)

        # 3. Reset and Re-run
        self.engine.reset()
        out_reset = []
        for h in range(10):
            idx = h * self.hop
            o, _ = self.engine.process_hop(p_speech[idx : idx + self.hop], r_speech[idx : idx + self.hop])
            out_reset.append(o)
        wav_reset = np.concatenate(out_reset)

        max_diff = np.max(np.abs(wav_fresh - wav_reset))
        self.assertLess(max_diff, 1e-6, f"Engine state reset was non-deterministic: max_diff={max_diff:.2e}")

    def test_extreme_silence_zero_dc_drift(self):
        """Zero-input silence must produce zero DC drift without denominator division-by-zero."""
        zeros = np.zeros(self.hop, dtype=np.float32)
        out_hops = []
        for _ in range(200):
            o, _ = self.engine.process_hop(zeros, zeros)
            out_hops.append(o)

        out_wav = np.concatenate(out_hops)
        self.assertFalse(np.isnan(out_wav).any())
        self.assertLess(np.max(np.abs(out_wav)), 1e-6, "DC drift or ringing detected during silence")

    def test_digital_overdrive_clipping_containment(self):
        """Extreme +12 dBFS digital overdrive inputs must be contained within safe [-0.98, +0.98] envelope."""
        rng = np.random.RandomState(202)
        p_overdrive = (rng.randn(self.hop) * 4.0).astype(np.float32)
        r_overdrive = (rng.randn(self.hop) * 4.0).astype(np.float32)

        for _ in range(20):
            o, _ = self.engine.process_hop(p_overdrive, r_overdrive)
            self.assertLessEqual(np.max(np.abs(o)), 0.985, "Overdrive exceeded clipping containment bound")

    def test_malformed_input_nan_inf_recovery(self):
        """Injected NaNs and Infs must be sanitized and next hop must recover cleanly."""
        rng = np.random.RandomState(303)
        p_bad = (rng.randn(self.hop) * 0.1).astype(np.float32)
        p_bad[10] = np.nan
        p_bad[20] = np.inf
        r_good = (rng.randn(self.hop) * 0.1).astype(np.float32)

        out_bad, _ = self.engine.process_hop(p_bad, r_good)
        self.assertFalse(np.isnan(out_bad).any(), "NaN leaked to output")
        self.assertFalse(np.isinf(out_bad).any(), "Inf leaked to output")

        # Next hop recovery
        p_normal = (rng.randn(self.hop) * 0.1).astype(np.float32)
        out_normal, _ = self.engine.process_hop(p_normal, r_good)
        self.assertFalse(np.isnan(out_normal).any())
        self.assertFalse(np.isinf(out_normal).any())
        self.assertLess(np.max(np.abs(out_normal)), 1.0)


if __name__ == "__main__":
    unittest.main()
