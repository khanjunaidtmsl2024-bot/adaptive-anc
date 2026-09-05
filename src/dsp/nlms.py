"""
Normalized Least Mean Squares (NLMS) Adaptive Filter Module.
PS 26052 — Adaptive Defence ANC.

Implements the Stage 1 Classical DSP Pre-AI Reference Canceller (Config A):
    d(n) = primary microphone (speech + ambient noise)
    x(n) = reference microphone (ambient noise reference)
    y(n) = W^T * x(n) (estimated noise in primary channel)
    e(n) = d(n) - y(n) (partially enhanced speech passed to Stage 2 AI)
"""

from typing import Tuple, Optional
import numpy as np


class NLMSFilter:
    """
    Normalized Least Mean Squares Adaptive Filter with power normalization,
    optional leakage factor for drift prevention, and coefficient clamping.
    """

    def __init__(
        self,
        filter_length: int = 64,
        step_size: float = 0.05,
        eps: float = 1e-6,
        leakage: float = 0.9999,
        max_weight: float = 5.0,
        normalized: bool = True,
    ):
        """
        Args:
            filter_length: Number of FIR filter taps (M).
            step_size: Learning rate (mu).
            eps: Regularization constant preventing division by zero.
            leakage: Leaky LMS factor (1.0 = standard, <1.0 = leaky).
            max_weight: Maximum allowable absolute value per filter weight.
            normalized: If True, normalize by input power (NLMS); if False, standard LMS.
        """
        self.M = int(filter_length)
        self.mu = float(step_size)
        self.eps = float(eps)
        self.leakage = float(leakage)
        self.max_weight = float(max_weight)
        self.normalized = bool(normalized)

        # Internal state
        self.weights = np.zeros(self.M, dtype=np.float32)
        self.buffer = np.zeros(self.M, dtype=np.float32)

    def reset(self) -> None:
        """Reset internal filter weights and delay line buffer."""
        self.weights.fill(0.0)
        self.buffer.fill(0.0)

    def adapt_sample(self, d_n: float, x_n: float) -> Tuple[float, float]:
        """
        Process a single audio sample (sample-by-sample streaming mode).

        Args:
            d_n: Primary microphone sample.
            x_n: Reference microphone sample.

        Returns:
            Tuple of (enhanced_error_sample, estimated_noise_sample)
        """
        # Shift delay line and insert newest reference sample
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = x_n

        # Filter output: y(n) = W^T * X(n)
        y_n = float(np.dot(self.weights, self.buffer))

        # Error signal: e(n) = d(n) - y(n)
        e_n = d_n - y_n

        # Power of reference buffer: ||X(n)||^2
        norm = float(np.dot(self.buffer, self.buffer)) + self.eps

        # Weight update with leakage:
        # If normalized (NLMS): W(n+1) = leakage * W(n) + (mu / norm) * e(n) * X(n)
        # If unnormalized (LMS): W(n+1) = leakage * W(n) + mu * e(n) * X(n)
        if self.normalized:
            factor = (self.mu / norm) * e_n
        else:
            factor = self.mu * e_n
        self.weights = self.leakage * self.weights + factor * self.buffer

        # Prevent weight divergence / clamp weights
        np.clip(self.weights, -self.max_weight, self.max_weight, out=self.weights)

        return e_n, y_n

    def update(self, x_n: float, d_n: float) -> float:
        """
        Convenience single-sample update method.
        Args:
            x_n: Reference noise sample.
            d_n: Primary desired signal sample.
        Returns:
            e_n: Filter error / enhanced speech sample.
        """
        e_n, _ = self.adapt_sample(d_n=d_n, x_n=x_n)
        return e_n

    def filter_block(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        adapt: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a contiguous block of audio samples.

        Args:
            primary: 1D array of primary microphone samples.
            reference: 1D array of reference microphone samples.
            adapt: If True, update filter coefficients; if False, freeze weights.

        Returns:
            Tuple of (error_signal, estimated_noise_signal)
        """
        n_samples = min(len(primary), len(reference))
        error = np.zeros(n_samples, dtype=np.float32)
        estimated = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            # Shift buffer
            self.buffer[1:] = self.buffer[:-1]
            self.buffer[0] = reference[i]

            # Output
            y_i = float(np.dot(self.weights, self.buffer))
            e_i = primary[i] - y_i

            estimated[i] = y_i
            error[i] = e_i

            if adapt:
                norm = float(np.dot(self.buffer, self.buffer)) + self.eps
                if self.normalized:
                    factor = (self.mu / norm) * e_i
                else:
                    factor = self.mu * e_i
                self.weights = self.leakage * self.weights + factor * self.buffer
                np.clip(self.weights, -self.max_weight, self.max_weight, out=self.weights)

        return error, estimated
