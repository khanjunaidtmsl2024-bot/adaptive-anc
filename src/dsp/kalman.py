"""
Impulsive-Robust Adaptive Filter & M-Estimate Kalman Filter.
PS 26052 — Adaptive Defence ANC.

Provides robust filtering when high-energy impulsive transients (gunshots, artillery,
shockwaves) strike the microphones, preventing the divergence that collapses classic LMS.
"""

from typing import Tuple
import numpy as np


class RobustImpulseFilter:
    """
    Adaptive filter with Huber-M error clipping and energy-variance thresholding
    to withstand defence impulsive transients exceeding 110-130 dB SPL.
    """

    def __init__(
        self,
        filter_length: int = 64,
        step_size: float = 0.05,
        threshold_factor: float = 3.0,
        eps: float = 1e-6,
    ):
        self.M = int(filter_length)
        self.mu = float(step_size)
        self.k_thresh = float(threshold_factor)
        self.eps = float(eps)

        self.weights = np.zeros(self.M, dtype=np.float32)
        self.buf = np.zeros(self.M, dtype=np.float32)

        # Running estimate of background noise standard deviation
        self.running_var = 1e-4
        self.alpha_var = 0.98

    def adapt_sample(self, d_n: float, x_n: float) -> Tuple[float, bool]:
        """
        Process sample with impulsive detection and robust M-estimate update.

        Returns:
            (error_sample, is_impulse_flag)
        """
        self.buf[1:] = self.buf[:-1]
        self.buf[0] = x_n

        y_n = float(np.dot(self.weights, self.buf))
        e_n = d_n - y_n

        # Track noise variance
        self.running_var = self.alpha_var * self.running_var + (1 - self.alpha_var) * (e_n ** 2)
        sigma = np.sqrt(self.running_var) + self.eps

        # Impulsive detection via threshold
        is_impulse = abs(e_n) > (self.k_thresh * sigma)

        # Robust score function psi(e) (Huber M-estimator)
        # Suppresses gradient update during massive impulsive spikes to prevent divergence
        if is_impulse:
            psi_e = np.sign(e_n) * self.k_thresh * sigma
            effective_mu = self.mu * 0.1  # Down-scale adaptation during blast
        else:
            psi_e = e_n
            effective_mu = self.mu

        norm = float(np.dot(self.buf, self.buf)) + self.eps
        self.weights += (effective_mu / norm) * psi_e * self.buf

        return e_n, is_impulse
