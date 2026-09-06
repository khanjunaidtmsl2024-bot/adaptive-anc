"""
Numba JIT-Compiled Impulsive Noise Protection Controller.
PS 26052 — Adaptive Defence ANC.

PERFORMANCE OPTIMIZATION — NOT AN ALGORITHM CHANGE.

Compiles the identical impulse detection and step-size scaling logic from
impulse_protection.py into native machine code via Numba @njit. The
sample-by-sample Python loop is the second-largest bottleneck after NLMS.

Mathematical equivalence with ImpulseProtectionController is a hard
correctness invariant.
"""

from typing import Tuple, Dict, Any
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _impulse_block_numba(
        error_signal: np.ndarray,
        running_var: float,
        hold_counter: int,
        current_scale: float,
        impulse_active: bool,
        total_impulses: int,
        theta: float,
        lambda_var: float,
        hold_samples: int,
        recovery_rate: float,
        limit: float,
    ) -> Tuple:
        """
        Numba-compiled impulse protection block processing.

        Returns: (protected, scales, flags, running_var_out, hold_counter_out,
                  current_scale_out, impulse_active_out, total_impulses_out)
        """
        n = len(error_signal)
        protected = np.zeros(n, dtype=np.float32)
        scales = np.zeros(n, dtype=np.float32)
        flags = np.zeros(n, dtype=np.bool_)

        rv = running_var
        hc = hold_counter
        cs = current_scale
        ia = impulse_active
        ti = total_impulses

        for i in range(n):
            e = float(error_signal[i])
            e_sq = e * e

            # Running standard deviation
            sigma_e = np.sqrt(rv + 1e-12)

            # Check for error spike
            if abs(e) > theta * sigma_e:
                ia = True
                hc = hold_samples
                ti += 1
                ratio = abs(e) / (theta * sigma_e + 1e-12)
                cs = 1.0 / (1.0 + ratio * ratio)
            else:
                if hc > 0:
                    hc -= 1
                    ia = True
                else:
                    ia = False
                    if cs < 1.0:
                        cs = min(1.0, cs + recovery_rate)

            # Update variance estimate only during non-impulsive intervals
            if not ia:
                rv = lambda_var * rv + (1.0 - lambda_var) * e_sq

            # Output limiter
            if e > limit:
                e_protected = limit
            elif e < -limit:
                e_protected = -limit
            else:
                e_protected = e

            protected[i] = np.float32(e_protected)
            scales[i] = np.float32(cs)
            flags[i] = ia

        return protected, scales, flags, rv, hc, cs, ia, ti


class ImpulseProtectionControllerFast:
    """
    Numba JIT-accelerated Impulse Protection Controller.

    Drop-in replacement for ImpulseProtectionController with identical API.
    Falls back to Python ImpulseProtectionController if Numba is not available.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        spike_threshold_sigmas: float = 3.5,
        var_smoothing: float = 0.995,
        hold_samples: int = 80,
        recovery_rate: float = 0.05,
        max_output_limit: float = 0.95,
    ):
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

        self._fallback = None
        if not NUMBA_AVAILABLE:
            from src.dsp.impulse_protection import ImpulseProtectionController
            self._fallback = ImpulseProtectionController(
                sample_rate=sample_rate,
                spike_threshold_sigmas=spike_threshold_sigmas,
                var_smoothing=var_smoothing,
                hold_samples=hold_samples,
                recovery_rate=recovery_rate,
                max_output_limit=max_output_limit,
            )

    def reset(self) -> None:
        """Reset detector state."""
        self.running_var = 0.001
        self.hold_counter = 0
        self.current_scale = 1.0
        self.impulse_active = False
        self.total_impulses_detected = 0
        if self._fallback is not None:
            self._fallback.reset()

    def filter_block_protection(
        self,
        error_signal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Block processing with impulse detection and step-size scaling.
        Returns: (protected_error, step_size_multipliers, stats)
        """
        if self._fallback is not None:
            result = self._fallback.filter_block_protection(error_signal)
            # Sync detector state; the fallback owns the authoritative state
            # while Numba is unavailable, and leaving these at their initial
            # values would desync wrapper attributes from real filter state.
            fb = self._fallback
            self.running_var = fb.running_var
            self.hold_counter = fb.hold_counter
            self.current_scale = fb.current_scale
            self.impulse_active = fb.impulse_active
            self.total_impulses_detected = fb.total_impulses_detected
            return result

        sig = np.ascontiguousarray(error_signal, dtype=np.float32)

        protected, scales, flags, rv, hc, cs, ia, ti = _impulse_block_numba(
            sig,
            self.running_var,
            self.hold_counter,
            self.current_scale,
            self.impulse_active,
            self.total_impulses_detected,
            self.theta,
            self.lambda_var,
            self.hold_samples,
            self.recovery_rate,
            self.limit,
        )

        # Update state
        self.running_var = float(rv)
        self.hold_counter = int(hc)
        self.current_scale = float(cs)
        self.impulse_active = bool(ia)
        self.total_impulses_detected = int(ti)

        # Compute stats from flags
        flag_diff = np.diff(flags.astype(np.int32))
        stats = {
            "impulses_detected": int(np.sum(flag_diff > 0)),
            "samples_suppressed": int(np.sum(flags)),
            "min_scale": float(np.min(scales)),
            "mean_scale": float(np.mean(scales)),
        }

        return protected, scales, stats

    @property
    def backend(self) -> str:
        """Returns the active backend name."""
        return "numba" if self._fallback is None else "python_fallback"


# Convenience alias
ImpulseProtectionFast = ImpulseProtectionControllerFast
