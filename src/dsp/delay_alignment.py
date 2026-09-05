"""
Acoustic Delay Alignment & Synchronization Engine.
PS 26052 — Adaptive Defence ANC.

Estimates the acoustic propagation delay between primary and reference microphones
using generalized cross-correlation (GCC) and aligns the signals before adaptive filtering.
Prevents non-causality where reference noise arrives later than primary noise.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.signal import correlate, correlation_lags


class DelayAligner:
    """
    Cross-correlation delay estimator and time-domain signal aligner.
    """

    def __init__(self, max_lag_samples: int = 32, sample_rate: int = 16000):
        """
        Args:
            max_lag_samples: Maximum search window for delay (32 samples = 2.0 ms @ 16 kHz).
            sample_rate: Audio sampling frequency in Hz.
        """
        self.max_lag = int(max_lag_samples)
        self.sr = int(sample_rate)

    def estimate_delay(self, primary: np.ndarray, reference: np.ndarray) -> int:
        """
        Estimates integer sample lag tau between primary and reference:
        Positive lag means reference LEADS primary (reference noise arrived earlier).
        Negative lag means reference LAGS primary.
        """
        n = min(len(primary), len(reference))
        if n < 64:
            return 0

        p = primary[:n] - np.mean(primary[:n])
        r = reference[:n] - np.mean(reference[:n])

        # Cross-correlation
        xcorr = correlate(p, r, mode="full", method="fft")
        lags = correlation_lags(len(p), len(r), mode="full")

        # Restrict to search window [-max_lag, max_lag]
        mask = (lags >= -self.max_lag) & (lags <= self.max_lag)
        valid_lags = lags[mask]
        valid_xcorr = xcorr[mask]

        if len(valid_xcorr) == 0:
            return 0

        best_lag = int(valid_lags[np.argmax(valid_xcorr)])
        return best_lag

    def align(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        delay_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Aligns reference to primary based on estimated or specified delay.
        Returns: (primary_aligned, reference_aligned, applied_delay)
        """
        n = min(len(primary), len(reference))
        if delay_samples is None:
            delay = self.estimate_delay(primary[:n], reference[:n])
        else:
            delay = int(delay_samples)

        p = primary[:n].copy()
        r = reference[:n].copy()

        if delay > 0:
            # Reference leads primary -> delay reference by shifting right
            r_aligned = np.zeros_like(r)
            r_aligned[delay:] = r[:-delay]
        elif delay < 0:
            # Reference lags primary -> advance reference by shifting left
            shift = abs(delay)
            r_aligned = np.zeros_like(r)
            r_aligned[:-shift] = r[shift:]
        else:
            r_aligned = r

        return p, r_aligned, delay


# Convenience alias
DelayAlignment = DelayAligner

