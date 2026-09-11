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

    def test_tiny_enhancer_is_strictly_causal(self):
        """TinyEnhancerNet must not leak future temporal frames into past outputs."""
        import torch
        from src.ai.tiny_enhancer import TinyEnhancerNet

        torch.manual_seed(42)
        model = TinyEnhancerNet().eval()
        B, C, F_dim, T_dim = 1, 1, 129, 100
        x = torch.randn(B, C, F_dim, T_dim)

        t0 = 50
        x_mutated = x.clone()
        x_mutated[:, :, :, t0 + 1:] += torch.randn_like(x_mutated[:, :, :, t0 + 1:]) * 500.0

        with torch.no_grad():
            out_orig = model(x)
            out_mut = model(x_mutated)

        diff_past = torch.max(torch.abs(out_orig[:, :, :, :t0 + 1] - out_mut[:, :, :, :t0 + 1])).item()
        diff_future = torch.max(torch.abs(out_orig[:, :, :, t0 + 1:] - out_mut[:, :, :, t0 + 1:])).item()

        self.assertLess(diff_past, 1e-6, f"Causality leak in TinyEnhancer: diff_past={diff_past:.2e}")
        self.assertGreater(diff_future, 1e-2, f"Future was not perturbed: diff_future={diff_future:.2e}")

    def test_tiny_enhancer_batch_streaming_equivalence(self):
        """Stateful frame-by-frame streaming must match full-clip batch evaluation to machine precision."""
        import torch
        from src.ai.tiny_enhancer import TinyEnhancerNet

        torch.manual_seed(42)
        model = TinyEnhancerNet().eval()
        B, C, F_dim, T_dim = 1, 1, 129, 80
        x = torch.randn(B, C, F_dim, T_dim)

        with torch.no_grad():
            out_batch = model(x, stateful=False)

            model.reset_state()
            stream_frames = []
            for t in range(T_dim):
                x_t = x[:, :, :, t:t + 1]
                out_t = model(x_t, stateful=True)
                stream_frames.append(out_t)
            out_stream = torch.cat(stream_frames, dim=-1)

        diff = torch.max(torch.abs(out_batch - out_stream)).item()
        self.assertLess(diff, 1e-5, f"Batch vs streaming mismatch: diff={diff:.2e}")


if __name__ == "__main__":
    unittest.main()

