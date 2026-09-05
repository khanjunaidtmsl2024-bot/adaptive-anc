"""
Filtered-x Least Mean Squares (FxLMS) and Secondary Path Modeling Module.
PS 26052 — Adaptive Defence ANC.

Used for:
1. Physical Acoustic Anti-Noise Cancellation modeling (earcup acoustic transfer).
2. Offline / Online secondary path estimation S_hat(z).
3. Compensating for DAC/ADC, amplifier, and speaker acoustic delay.
"""

from typing import Tuple, Optional
import numpy as np


class SecondaryPathModel:
    """Models the electro-acoustic secondary path S(z) from actuator to error mic."""

    def __init__(self, impulse_response: np.ndarray):
        """
        Args:
            impulse_response: 1D FIR coefficients representing S(z).
        """
        self.h = np.asarray(impulse_response, dtype=np.float32)
        self.L = len(self.h)
        self.buf = np.zeros(self.L, dtype=np.float32)

    def filter_sample(self, x_n: float) -> float:
        """Pass one sample through S(z)."""
        self.buf[1:] = self.buf[:-1]
        self.buf[0] = x_n
        return float(np.dot(self.h, self.buf))

    def filter_signal(self, signal: np.ndarray) -> np.ndarray:
        """Convolve signal with S(z) FIR filter."""
        return np.convolve(signal, self.h, mode="same").astype(np.float32)


class FxLMSFilter:
    """
    Filtered-x Least Mean Squares Filter.
    Filters the reference signal x(n) through S_hat(z) before adaptation:
        x_filtered(n) = S_hat(z) * x(n)
        W(n+1) = W(n) + mu * e(n) * x_filtered(n)
    """

    def __init__(
        self,
        filter_length: int = 64,
        step_size: float = 0.01,
        secondary_path_estimate: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ):
        self.M = int(filter_length)
        self.mu = float(step_size)
        self.eps = float(eps)

        if secondary_path_estimate is None:
            # Default to direct delta impulse with slight delay
            self.s_hat = np.zeros(16, dtype=np.float32)
            self.s_hat[1] = 1.0
        else:
            self.s_hat = np.asarray(secondary_path_estimate, dtype=np.float32)

        self.sec_model = SecondaryPathModel(self.s_hat)
        self.weights = np.zeros(self.M, dtype=np.float32)
        self.ref_buf = np.zeros(self.M, dtype=np.float32)
        self.filtered_ref_buf = np.zeros(self.M, dtype=np.float32)

    def adapt_sample(self, error_n: float, ref_n: float) -> float:
        """
        Adapt filter based on error signal e(n) and reference input x(n).

        Returns:
            anti_noise_sample: y(n) generated for speaker actuator.
        """
        # Update raw reference buffer
        self.ref_buf[1:] = self.ref_buf[:-1]
        self.ref_buf[0] = ref_n

        # Generate anti-noise: y(n) = W^T * X(n)
        y_n = float(np.dot(self.weights, self.ref_buf))

        # Filter reference through secondary path model: x'(n) = S_hat(z) * x(n)
        filtered_ref = self.sec_model.filter_sample(ref_n)
        self.filtered_ref_buf[1:] = self.filtered_ref_buf[:-1]
        self.filtered_ref_buf[0] = filtered_ref

        # Update weights: W(n+1) = W(n) + mu * e(n) * X'(n) / (||X'(n)||^2 + eps)
        norm = float(np.dot(self.filtered_ref_buf, self.filtered_ref_buf)) + self.eps
        self.weights += (self.mu / norm) * error_n * self.filtered_ref_buf

        return y_n
