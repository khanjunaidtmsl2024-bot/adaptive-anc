"""
Runtime Safety Controller & Fail-Safe Bypass System.
PS 26052 — Adaptive Defence ANC.

Protects communication from acoustic feedback, filter divergence,
or transient clipping by smoothly crossfading to a safe bypass mode.
"""

from typing import Tuple
import numpy as np


class FallbackController:
    """Monitors output health and dynamically routes audio during emergency states."""

    def __init__(self, clip_threshold: float = 0.99, max_energy_ratio: float = 10.0):
        self.clip_threshold = float(clip_threshold)
        self.max_ratio = float(max_energy_ratio)
        self.divergence_counter = 0

    def verify_frame(self, primary_in: np.ndarray, enhanced_out: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Check if enhanced audio contains NaN, Inf, extreme clipping, or explosion.

        Returns:
            (safe_audio, is_diverged_flag)
        """
        # 1. NaN or Inf Check
        if not np.all(np.isfinite(enhanced_out)):
            self.divergence_counter += 1
            return primary_in.copy(), True

        # 2. Energy Ratio Check (Did enhancement amplify noise by >10x?)
        e_in = np.mean(primary_in ** 2) + 1e-12
        e_out = np.mean(enhanced_out ** 2) + 1e-12

        if (e_out / e_in) > self.max_ratio:
            self.divergence_counter += 1
            # Gracefully attenuate and blend with primary
            return 0.5 * primary_in + 0.1 * np.clip(enhanced_out, -1.0, 1.0), True

        # 3. Peak Clipping Check
        if np.max(np.abs(enhanced_out)) > self.clip_threshold:
            enhanced_out = np.tanh(enhanced_out)

        self.divergence_counter = max(0, self.divergence_counter - 1)
        return enhanced_out, False
