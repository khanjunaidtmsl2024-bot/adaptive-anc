"""
PH0.5 — Numba vs Python Correctness Equivalence Tests.
PS 26052 — Adaptive Defence ANC.

Validates that Numba JIT-compiled implementations produce numerically
identical output (within floating-point tolerance) to the reference
Python implementations. This is a HARD correctness invariant.

Evidence Tier: SOFTWARE VERIFIED
"""

import numpy as np
import pytest


class TestNLMSNumericalEquivalence:
    """Verify VSSNLMSFilterFast == VSSNLMSFilter to within 1e-5."""

    def _make_filters(self):
        from src.dsp.vss_nlms import VSSNLMSFilter
        from src.dsp.vss_nlms_fast import VSSNLMSFilterFast

        params = dict(
            filter_length=64, mu_init=0.05, mu_min=0.001, mu_max=0.4,
            alpha=0.97, beta=0.95, gamma=0.5, eps=1e-6,
            leakage=0.9999, max_weight=5.0,
        )
        py_filter = VSSNLMSFilter(**params)
        fast_filter = VSSNLMSFilterFast(**params)
        return py_filter, fast_filter

    def test_identical_output_on_white_noise(self):
        """White noise input: both backends must produce identical error signal."""
        py_f, fast_f = self._make_filters()

        rng = np.random.RandomState(42)
        primary = rng.randn(512).astype(np.float32) * 0.3
        reference = rng.randn(512).astype(np.float32) * 0.3

        e_py, est_py, mu_py = py_f.filter_block(primary, reference)
        e_fast, est_fast, mu_fast = fast_f.filter_block(primary, reference)

        np.testing.assert_allclose(e_py, e_fast, atol=1e-5,
            err_msg="NLMS error signal mismatch (Python vs Numba)")
        np.testing.assert_allclose(est_py, est_fast, atol=1e-5,
            err_msg="NLMS estimated signal mismatch (Python vs Numba)")
        np.testing.assert_allclose(mu_py, mu_fast, atol=1e-5,
            err_msg="NLMS mu trajectory mismatch (Python vs Numba)")

    def test_identical_output_on_correlated_signal(self):
        """Correlated noise + delayed copy: realistic ANC scenario."""
        py_f, fast_f = self._make_filters()

        rng = np.random.RandomState(99)
        noise = rng.randn(640).astype(np.float32) * 0.2
        speech = np.sin(2 * np.pi * 300 * np.arange(640) / 16000).astype(np.float32) * 0.1
        primary = speech + noise
        reference = np.roll(noise, 5) * 0.9  # delayed, attenuated copy

        e_py, _, _ = py_f.filter_block(primary, reference)
        e_fast, _, _ = fast_f.filter_block(primary, reference)

        np.testing.assert_allclose(e_py, e_fast, atol=1e-5,
            err_msg="NLMS error mismatch on correlated signal")

    def test_identical_weights_after_convergence(self):
        """After processing, filter weights must be identical."""
        py_f, fast_f = self._make_filters()

        rng = np.random.RandomState(7)
        for _ in range(5):
            p = rng.randn(128).astype(np.float32) * 0.1
            r = rng.randn(128).astype(np.float32) * 0.1
            py_f.filter_block(p, r)
            fast_f.filter_block(p, r)

        np.testing.assert_allclose(py_f.weights, fast_f.weights, atol=1e-5,
            err_msg="NLMS weights diverged (Python vs Numba)")

    def test_adapt_false_passthrough(self):
        """With adapt=False, both should produce identical non-adapting output."""
        py_f, fast_f = self._make_filters()

        rng = np.random.RandomState(123)
        p = rng.randn(256).astype(np.float32) * 0.1
        r = rng.randn(256).astype(np.float32) * 0.1

        e_py, _, _ = py_f.filter_block(p, r, adapt=False)
        e_fast, _, _ = fast_f.filter_block(p, r, adapt=False)

        np.testing.assert_allclose(e_py, e_fast, atol=1e-5,
            err_msg="NLMS non-adapting output mismatch")


class TestImpulseNumericalEquivalence:
    """Verify ImpulseProtectionControllerFast == ImpulseProtectionController."""

    def _make_controllers(self):
        from src.dsp.impulse_protection import ImpulseProtectionController
        from src.dsp.impulse_protection_fast import ImpulseProtectionControllerFast

        params = dict(
            sample_rate=16000, spike_threshold_sigmas=3.5,
            var_smoothing=0.995, hold_samples=80,
            recovery_rate=0.05, max_output_limit=0.95,
        )
        py_ctl = ImpulseProtectionController(**params)
        fast_ctl = ImpulseProtectionControllerFast(**params)
        return py_ctl, fast_ctl

    def test_identical_output_on_clean_signal(self):
        """Clean signal: both controllers must produce identical output."""
        py_c, fast_c = self._make_controllers()

        rng = np.random.RandomState(42)
        signal = rng.randn(512).astype(np.float32) * 0.05

        p_py, s_py, _ = py_c.filter_block_protection(signal)
        p_fast, s_fast, _ = fast_c.filter_block_protection(signal)

        np.testing.assert_allclose(p_py, p_fast, atol=1e-5,
            err_msg="Impulse protected signal mismatch (clean)")
        np.testing.assert_allclose(s_py, s_fast, atol=1e-5,
            err_msg="Impulse scale factors mismatch (clean)")

    def test_identical_output_on_impulsive_signal(self):
        """Signal with spike: both must detect and suppress identically."""
        py_c, fast_c = self._make_controllers()

        rng = np.random.RandomState(42)
        signal = rng.randn(512).astype(np.float32) * 0.05
        # Inject spike at sample 200
        signal[200] = 5.0
        signal[201] = -4.0

        p_py, s_py, stats_py = py_c.filter_block_protection(signal)
        p_fast, s_fast, stats_fast = fast_c.filter_block_protection(signal)

        np.testing.assert_allclose(p_py, p_fast, atol=1e-5,
            err_msg="Impulse protected signal mismatch (impulsive)")
        np.testing.assert_allclose(s_py, s_fast, atol=1e-5,
            err_msg="Impulse scale factors mismatch (impulsive)")

    def test_multi_block_state_consistency(self):
        """Process multiple blocks: state must track identically."""
        py_c, fast_c = self._make_controllers()

        rng = np.random.RandomState(77)
        for block_idx in range(10):
            signal = rng.randn(128).astype(np.float32) * 0.05
            if block_idx == 3:
                signal[50] = 8.0  # Spike in block 3

            p_py, _, _ = py_c.filter_block_protection(signal)
            p_fast, _, _ = fast_c.filter_block_protection(signal)

            np.testing.assert_allclose(p_py, p_fast, atol=1e-5,
                err_msg=f"Impulse mismatch at block {block_idx}")


class TestFastBackendProperty:
    """Verify backend property reports correctly."""

    def test_nlms_backend_is_numba(self):
        from src.dsp.vss_nlms_fast import VSSNLMSFilterFast, NUMBA_AVAILABLE
        f = VSSNLMSFilterFast()
        expected = "numba" if NUMBA_AVAILABLE else "python_fallback"
        assert f.backend == expected

    def test_impulse_backend_is_numba(self):
        from src.dsp.impulse_protection_fast import ImpulseProtectionControllerFast, NUMBA_AVAILABLE
        c = ImpulseProtectionControllerFast()
        expected = "numba" if NUMBA_AVAILABLE else "python_fallback"
        assert c.backend == expected
