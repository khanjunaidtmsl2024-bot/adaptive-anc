"""
Variable Step-Size Normalized Least Mean Squares (VSS-NLMS) Filter.
PS 26052 — Adaptive Defence ANC.

Implements the Aboulnasr & Mayyas (1997) / Benveniste VSS formulation:
Step-size mu(n) dynamically adapts according to the estimated error-reference
correlation p(n), achieving fast convergence during non-stationary transitions
and low misadjustment during steady-state periods.

Equations:
    y(n) = W^T(n) * X(n)
    e(n) = d(n) - y(n)
    p(n) = beta * p(n-1) + (1 - beta) * e(n) * x(n)
    mu(n+1) = alpha * mu(n) + gamma * p^2(n)
    mu(n+1) = clamp(mu(n+1), mu_min, mu_max)
    W(n+1) = leakage * W(n) + [mu(n+1) / (eps + ||X(n)||^2)] * e(n) * X(n)
"""

from typing import Tuple, Optional
import numpy as np


class VSSNLMSFilter:
    """
    Variable Step-Size Normalized Least Mean Squares Adaptive Filter.
    """

    def __init__(
        self,
        filter_length: int = 64,
        mu_init: float = 0.05,
        mu_min: float = 0.001,
        mu_max: float = 0.4,
        alpha: float = 0.97,
        beta: float = 0.95,
        gamma: float = 0.5,
        eps: float = 1e-6,
        leakage: float = 0.9999,
        max_weight: float = 5.0,
    ):
        """
        Args:
            filter_length: Number of FIR filter taps (M).
            mu_init: Initial step-size.
            mu_min: Minimum allowable step-size (steady-state floor).
            mu_max: Maximum allowable step-size (tracking ceiling).
            alpha: Step-size smoothing parameter (0 < alpha < 1).
            beta: Error-input correlation smoothing parameter (0 < beta < 1).
            gamma: Correlation-to-step gain factor.
            eps: Regularization constant.
            leakage: Weight leakage factor.
            max_weight: Coefficient clamping limit.
        """
        self.M = int(filter_length)
        self.mu = float(mu_init)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.leakage = float(leakage)
        self.max_weight = float(max_weight)

        # State vectors
        self.weights = np.zeros(self.M, dtype=np.float32)
        self.buffer = np.zeros(self.M, dtype=np.float32)
        self.p_cor = 0.0  # Smoothed error-reference correlation

    def reset(self) -> None:
        """Reset internal filter coefficients and correlation memory."""
        self.weights.fill(0.0)
        self.buffer.fill(0.0)
        self.p_cor = 0.0

    def adapt_sample(self, d_n: float, x_n: float) -> Tuple[float, float, float]:
        """
        Process a single sample and return (error, estimated_noise, current_mu).
        """
        # Delay line update
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = x_n

        # Output estimate
        y_n = float(np.dot(self.weights, self.buffer))
        e_n = d_n - y_n

        # Update smoothed correlation: p(n) = beta * p(n-1) + (1 - beta) * e(n) * x(n)
        self.p_cor = self.beta * self.p_cor + (1.0 - self.beta) * (e_n * x_n)

        # Update variable step-size: mu(n+1) = alpha * mu(n) + gamma * p(n)^2
        self.mu = self.alpha * self.mu + self.gamma * (self.p_cor ** 2)
        self.mu = float(np.clip(self.mu, self.mu_min, self.mu_max))

        # Weight update with power normalization
        norm = float(np.dot(self.buffer, self.buffer)) + self.eps
        factor = (self.mu / norm) * e_n
        self.weights = self.leakage * self.weights + factor * self.buffer

        # Clamping
        np.clip(self.weights, -self.max_weight, self.max_weight, out=self.weights)

        return e_n, y_n, self.mu

    def filter_block(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        adapt: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter a contiguous block of audio samples.
        Returns: (error_signal, estimated_noise_signal, mu_trajectory)
        """
        n_samples = min(len(primary), len(reference))
        error = np.zeros(n_samples, dtype=np.float32)
        estimated = np.zeros(n_samples, dtype=np.float32)
        mu_hist = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            self.buffer[1:] = self.buffer[:-1]
            self.buffer[0] = reference[i]

            y_i = float(np.dot(self.weights, self.buffer))
            e_i = primary[i] - y_i

            estimated[i] = y_i
            error[i] = e_i

            if adapt:
                self.p_cor = self.beta * self.p_cor + (1.0 - self.beta) * (e_i * reference[i])
                self.mu = self.alpha * self.mu + self.gamma * (self.p_cor ** 2)
                self.mu = float(np.clip(self.mu, self.mu_min, self.mu_max))

                norm = float(np.dot(self.buffer, self.buffer)) + self.eps
                factor = (self.mu / norm) * e_i
                self.weights = self.leakage * self.weights + factor * self.buffer
                np.clip(self.weights, -self.max_weight, self.max_weight, out=self.weights)

            mu_hist[i] = self.mu

        return error, estimated, mu_hist
