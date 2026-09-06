"""
Numba JIT-Compiled Variable Step-Size NLMS Filter.
PS 26052 — Adaptive Defence ANC.

PERFORMANCE OPTIMIZATION — NOT AN ALGORITHM CHANGE.

This module compiles the identical VSS-NLMS mathematics from vss_nlms.py
into native machine code using Numba's @njit decorator. The inner
sample-by-sample loop eliminates Python interpreter dispatch overhead,
achieving ~10-50× speedup on the adaptive filter hot path.

Mathematical equivalence with VSSNLMSFilter is a hard correctness invariant.
"""

from typing import Tuple, Optional
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _nlms_filter_block_numba(
        primary: np.ndarray,
        reference: np.ndarray,
        weights: np.ndarray,
        buffer: np.ndarray,
        p_cor: float,
        mu: float,
        mu_min: float,
        mu_max: float,
        alpha: float,
        beta: float,
        gamma: float,
        eps: float,
        leakage: float,
        max_weight: float,
        adapt: bool,
    ) -> Tuple:
        """
        Numba-compiled VSS-NLMS block processing.

        Returns: (error, estimated, mu_hist, weights_out, buffer_out, p_cor_out, mu_out)
        """
        n = min(len(primary), len(reference))
        M = len(weights)
        error = np.zeros(n, dtype=np.float32)
        estimated = np.zeros(n, dtype=np.float32)
        mu_hist = np.zeros(n, dtype=np.float32)

        # Work on copies to avoid mutating input arrays unexpectedly
        w = weights.copy()
        buf = buffer.copy()
        pc = p_cor
        mu_val = mu

        for i in range(n):
            # Delay line shift
            for j in range(M - 1, 0, -1):
                buf[j] = buf[j - 1]
            buf[0] = reference[i]

            # Output estimate: y = W^T * X
            y_i = np.float32(0.0)
            for j in range(M):
                y_i += w[j] * buf[j]

            e_i = primary[i] - y_i

            estimated[i] = y_i
            error[i] = e_i

            if adapt:
                # Update smoothed correlation
                pc = beta * pc + (1.0 - beta) * (e_i * reference[i])

                # Update variable step-size
                mu_val = alpha * mu_val + gamma * (pc * pc)
                if mu_val < mu_min:
                    mu_val = mu_min
                elif mu_val > mu_max:
                    mu_val = mu_max

                # Weight update with power normalization
                norm = np.float32(0.0)
                for j in range(M):
                    norm += buf[j] * buf[j]
                norm += eps

                factor = (mu_val / norm) * e_i

                for j in range(M):
                    w[j] = leakage * w[j] + factor * buf[j]
                    # Clamping
                    if w[j] > max_weight:
                        w[j] = max_weight
                    elif w[j] < -max_weight:
                        w[j] = -max_weight

            mu_hist[i] = mu_val

        return error, estimated, mu_hist, w, buf, pc, mu_val


class VSSNLMSFilterFast:
    """
    Numba JIT-accelerated VSS-NLMS Adaptive Filter.

    Drop-in replacement for VSSNLMSFilter with identical API.
    Falls back to Python VSSNLMSFilter if Numba is not available.
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

        self.weights = np.zeros(self.M, dtype=np.float32)
        self.buffer = np.zeros(self.M, dtype=np.float32)
        self.p_cor = 0.0

        self._fallback = None
        if not NUMBA_AVAILABLE:
            from src.dsp.vss_nlms import VSSNLMSFilter
            self._fallback = VSSNLMSFilter(
                filter_length=filter_length,
                mu_init=mu_init,
                mu_min=mu_min,
                mu_max=mu_max,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                eps=eps,
                leakage=leakage,
                max_weight=max_weight,
            )

    def reset(self) -> None:
        """Reset internal filter coefficients and correlation memory."""
        self.weights.fill(0.0)
        self.buffer.fill(0.0)
        self.p_cor = 0.0
        if self._fallback is not None:
            self._fallback.reset()

    def filter_block(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        adapt: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter a contiguous block of audio samples.
        Returns: (error_signal, estimated_noise_signal, mu_trajectory)
        """
        if self._fallback is not None:
            return self._fallback.filter_block(primary, reference, adapt)

        p = np.ascontiguousarray(primary, dtype=np.float32)
        r = np.ascontiguousarray(reference, dtype=np.float32)

        error, estimated, mu_hist, w_out, buf_out, pc_out, mu_out = \
            _nlms_filter_block_numba(
                p, r,
                self.weights, self.buffer,
                self.p_cor, self.mu,
                self.mu_min, self.mu_max,
                self.alpha, self.beta, self.gamma,
                self.eps, self.leakage, self.max_weight,
                adapt,
            )

        # Update state
        self.weights = w_out
        self.buffer = buf_out
        self.p_cor = pc_out
        self.mu = mu_out

        return error, estimated, mu_hist

    def freeze_adaptation(self, freeze_duration_samples: int = 128) -> None:
        """Records adaptation freeze duration in samples."""
        self._freeze_counter = getattr(self, "_freeze_counter", 0) + freeze_duration_samples

    @property
    def backend(self) -> str:
        """Returns the active backend name."""
        return "numba" if self._fallback is None else "python_fallback"


# Convenience alias
VSSNLMSFast = VSSNLMSFilterFast
