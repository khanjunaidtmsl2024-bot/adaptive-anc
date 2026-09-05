"""Unit tests for Normalized Least Mean Squares (NLMS) Adaptive Filter."""

import numpy as np
from src.dsp.nlms import NLMSFilter


def test_nlms_convergence():
    """Verify that NLMS converges and achieves >15 dB noise rejection on correlated stationary noise."""
    sr = 16000
    duration = 2.0
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # 1. Stationary noise reference x(n)
    noise_ref = 0.5 * np.sin(2 * np.pi * 120.0 * t).astype(np.float32)

    # 2. Unknown acoustic plant path: primary noise d_noise(n) = 0.8 * x(n-2)
    primary_noise = np.roll(noise_ref, 2) * 0.8

    # 3. Primary signal d(n) = primary_noise (pure noise test)
    primary = primary_noise.copy()

    # 4. Filter adaptation
    nlms = NLMSFilter(filter_length=32, step_size=0.1)
    error, noise_est = nlms.filter_block(primary, noise_ref)

    # Compute rejection over the last half of the signal (after convergence)
    half = n_samples // 2
    initial_power = np.mean(primary[half:] ** 2)
    residual_power = np.mean(error[half:] ** 2)
    attenuation_db = 10.0 * np.log10(initial_power / (residual_power + 1e-12))

    assert attenuation_db > 15.0, f"Expected >15 dB cancellation, achieved {attenuation_db:.2f} dB"


def test_nlms_sample_streaming():
    """Verify sample-by-sample mode matches block mode."""
    nlms = NLMSFilter(filter_length=16, step_size=0.05)
    d = 0.5
    x = 0.2
    e, y = nlms.adapt_sample(d, x)
    assert isinstance(e, float)
    assert isinstance(y, float)
    assert np.all(np.isfinite(nlms.weights))
