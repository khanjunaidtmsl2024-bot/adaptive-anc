"""
Unit tests for Advanced DSP Subsystems:
VSS-NLMS, Delay Alignment, Speech Leakage Detection, and Impulse Protection.
PS 26052 — Adaptive Defence ANC.
"""

import numpy as np
import pytest

from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.delay_alignment import DelayAligner
from src.dsp.leakage_detector import SpeechLeakageDetector
from src.dsp.impulse_protection import ImpulseProtectionController


def test_vss_nlms_convergence():
    """Verify VSS-NLMS dynamically adapts step size and cancels correlated noise."""
    n_samples = 4000
    ref = np.random.randn(n_samples).astype(np.float32)
    # Target system: 2-tap FIR filter
    desired = 0.6 * ref + 0.3 * np.roll(ref, 1)

    vss = VSSNLMSFilter(filter_length=16, mu_init=0.1)
    err, est, mu_hist = vss.filter_block(desired, ref)

    assert len(err) == n_samples
    assert np.all(np.isfinite(err))
    # Error energy in second half should be significantly lower than first half
    early_err = np.mean(err[:500] ** 2)
    late_err = np.mean(err[-500:] ** 2)
    assert late_err < early_err * 0.3
    # Step size should have varied
    assert np.max(mu_hist) > np.min(mu_hist)


def test_delay_alignment():
    """Verify GCC delay estimation accurately detects integer sample shifts."""
    sr = 16000
    aligner = DelayAligner(max_lag_samples=16, sample_rate=sr)

    ref = np.random.randn(2000).astype(np.float32)
    # Primary is delayed by 4 samples relative to reference (reference leads primary by 4)
    delay_true = 4
    prim = np.roll(ref, delay_true)

    est_delay = aligner.estimate_delay(prim, ref)
    assert est_delay == delay_true

    p_align, r_align, applied = aligner.align(prim, ref)
    assert applied == delay_true
    # Cross correlation at lag 0 after alignment should be maximum
    corr_zero = np.dot(p_align[delay_true:], r_align[delay_true:])
    assert corr_zero > 0.8 * len(ref)


def test_leakage_detector():
    """Verify speech leakage detector reduces gating factor when primary has speech."""
    sr = 16000
    detector = SpeechLeakageDetector(sample_rate=sr)

    # 1. Noise-dominated frame (no speech)
    prim_noise = np.random.randn(512).astype(np.float32) * 0.1
    ref_noise = np.random.randn(512).astype(np.float32) * 0.1
    gate_clean, info_clean = detector.compute_frame_gating(prim_noise, ref_noise)
    assert gate_clean > 0.8
    assert info_clean["leakage_detected"] is False

    # 2. Speech-dominated frame with 1000 Hz formant tone (voice band)
    t = np.linspace(0, 512 / sr, 512, endpoint=False)
    prim_speech = 0.8 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    ref_leaked = prim_noise + 0.15 * prim_speech

    gate_leaked, info_leaked = detector.compute_frame_gating(prim_speech, ref_leaked)
    assert gate_leaked < gate_clean


def test_impulse_protection():
    """Verify impulse protection detects ballistic spikes and attenuates step size."""
    controller = ImpulseProtectionController(sample_rate=16000, spike_threshold_sigmas=3.5)

    # Train baseline variance on bounded steady-state noise (no accidental Gaussian 3.5-sigma spikes)
    normal_err = np.random.uniform(-0.05, 0.05, 500).astype(np.float32)
    for sample in normal_err:
        controller.process_sample(sample)

    assert controller.impulse_active is False

    # Inject extreme gunfire shockwave (10x amplitude)
    spike = 1.5
    protected_val, scale, is_impulse = controller.process_sample(spike)

    assert is_impulse is True
    assert scale < 0.2  # Step size severely attenuated
    assert abs(protected_val) <= controller.limit

