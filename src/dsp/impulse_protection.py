"""
Impulsive Noise Protection & Filter Divergence Controller.
PS 26052 — Adaptive Defence ANC.

Prevents adaptive filter weight explosion during extreme acoustic shockwaves
(artillery, gunfire, door slams, ballistic shockwaves) by detecting error spikes,
attenuating or freezing adaptation step-size mu, and enforcing controlled recovery.
"""

from typing import Tuple, Dict, Any
import numpy as np


class ImpulseProtectionController:
    """
    Error-spike detector and adaptive step-size protector.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        spike_threshold_sigmas: float = 3.5,
        var_smoothing: float = 0.995,
        hold_samples: int = 80,       # 5.0 ms hold at 16 kHz
        recovery_rate: float = 0.05,   # Linear ramp-up per sample
        max_output_limit: float = 0.95,
    ):
        """
        Args:
            sample_rate: Sampling frequency in Hz.
            spike_threshold_sigmas: Multiplier on running standard deviation for spike trigger.
            var_smoothing: Moving average factor for error variance tracking.
            hold_samples: Number of samples to hold adaptation suppressed after a transient.
            recovery_rate: Rate of step-size recovery back to 1.0 per sample.
            max_output_limit: Hard limiter threshold to protect hearing and DAC.
        """
        self.sr = int(sample_rate)
        self.theta = float(spike_threshold_sigmas)
        self.lambda_var = float(var_smoothing)
        self.hold_samples = int(hold_samples)
        self.recovery_rate = float(recovery_rate)
        self.limit = float(max_output_limit)

        # Internal state
        self.running_var = 0.001
        self.hold_counter = 0
        self.current_scale = 1.0
        self.impulse_active = False
        self.total_impulses_detected = 0

    def reset(self) -> None:
        """Reset detector state."""
        self.running_var = 0.001
        self.hold_counter = 0
        self.current_scale = 1.0
        self.impulse_active = False
        self.total_impulses_detected = 0

    def process_sample(self, error_sample: float) -> Tuple[float, float, bool]:
        """
        Processes sample-by-sample:
        Returns:
            (protected_error_sample, step_size_scale_factor, is_impulse)
        """
        e = float(error_sample)
        e_sq = e ** 2

        # Running standard deviation
        sigma_e = float(np.sqrt(self.running_var + 1e-12))

        # Check for error spike: |e| > theta * sigma_e
        if abs(e) > self.theta * sigma_e:
            self.impulse_active = True
            self.hold_counter = self.hold_samples
            self.total_impulses_detected += 1
            # Step size scale drops dramatically
            ratio = abs(e) / (self.theta * sigma_e + 1e-12)
            self.current_scale = float(1.0 / (1.0 + ratio ** 2))
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
                self.impulse_active = True
            else:
                self.impulse_active = False
                # Smooth recovery ramp
                if self.current_scale < 1.0:
                    self.current_scale = min(1.0, self.current_scale + self.recovery_rate)

        # Update variance estimate only during non-impulsive intervals to avoid polluting baseline
        if not self.impulse_active:
            self.running_var = self.lambda_var * self.running_var + (1.0 - self.lambda_var) * e_sq

        # Output limiter protection
        e_protected = float(np.clip(e, -self.limit, self.limit))

        return e_protected, float(self.current_scale), self.impulse_active

    def filter_block_protection(
        self,
        error_signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Vectorized/block processing with impulse detection and step-size scaling.
        Returns: (protected_error, step_size_multipliers, stats)
        """
        n = len(error_signal)
        protected = np.zeros(n, dtype=np.float32)
        scales = np.zeros(n, dtype=np.float32)
        flags = np.zeros(n, dtype=bool)

        for i in range(n):
            p_i, s_i, flag_i = self.process_sample(error_signal[i])
            protected[i] = p_i
            scales[i] = s_i
            flags[i] = flag_i

        stats = {
            "impulses_detected": int(np.sum(np.diff(flags.astype(int)) > 0)),
            "samples_suppressed": int(np.sum(flags)),
            "min_scale": float(np.min(scales)),
            "mean_scale": float(np.mean(scales)),
        }

        return protected, scales, stats
