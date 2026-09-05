"""Unit tests for Objective Audio Metrics."""

import numpy as np
from src.evaluation.metrics import compute_snr, compute_si_snr, evaluate_all_metrics


def test_snr_pure_signal():
    """Verify SNR of near-identical signals is very high."""
    clean = np.sin(np.linspace(0, 10, 16000)).astype(np.float32)
    noisy = clean + 1e-4 * np.random.normal(0, 1, 16000).astype(np.float32)
    snr = compute_snr(clean, noisy)
    assert snr > 50.0


def test_si_snr_invariance():
    """Verify SI-SNR is invariant to arbitrary linear scale."""
    ref = np.random.normal(0, 1, 8000).astype(np.float32)
    est = 3.5 * ref  # scaled version
    si = compute_si_snr(ref, est)
    assert si > 100.0, f"Expected high SI-SNR for scaled signal, got {si}"


def test_evaluate_all():
    """Verify evaluate_all_metrics returns compliant dictionary."""
    clean = np.random.normal(0, 1, 16000).astype(np.float32)
    res = evaluate_all_metrics(clean, clean, sample_rate=16000)
    assert "snr_db" in res
    assert "si_snr_db" in res
    assert "compliance" in res
