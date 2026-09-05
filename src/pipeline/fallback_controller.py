"""
Runtime Safety Controller & Fail-Safe Bypass System.
PS 26052 — Adaptive Defence ANC.

Protects communication from acoustic feedback, filter divergence,
or transient clipping by smoothly crossfading to a safe bypass mode.
Includes crest-factor and spectral-flux based impulse blast detection
to immediately freeze adaptive filter updates during gunfire or explosions.
"""

from typing import Tuple, Dict, Any
import numpy as np


class FallbackController:
    """Monitors output health and dynamically routes audio during emergency states."""

    def __init__(
        self,
        clip_threshold: float = 0.95,
        max_energy_ratio: float = 10.0,
        crest_factor_threshold: float = 6.0,
        spectral_flux_threshold: float = 0.3
    ):
        self.clip_threshold = float(clip_threshold)
        self.max_ratio = float(max_energy_ratio)
        self.cf_thresh = float(crest_factor_threshold)
        self.flux_thresh = float(spectral_flux_threshold)
        self.divergence_counter = 0
        self.prev_magnitude_spectrum = None

    def detect_impulse(self, frame: np.ndarray) -> Tuple[bool, float, float]:
        """
        Detect sharp ballistic muzzle blasts or explosions using Crest Factor and Spectral Flux.

        Returns:
            Tuple of (is_impulse, crest_factor, spectral_flux)
        """
        rms = float(np.sqrt(np.mean(frame ** 2))) + 1e-12
        peak = float(np.max(np.abs(frame)))
        crest_factor = peak / rms

        # Spectral flux: positive difference in normalized magnitude spectrum
        spec = np.abs(np.fft.rfft(frame))
        spec_norm = spec / (np.sum(spec) + 1e-12)

        if self.prev_magnitude_spectrum is None:
            spectral_flux = 0.0
        else:
            diff = spec_norm - self.prev_magnitude_spectrum
            spectral_flux = float(np.sum(diff[diff > 0]))

        self.prev_magnitude_spectrum = spec_norm

        is_impulse = (crest_factor > self.cf_thresh) or (spectral_flux > self.flux_thresh) or (peak > 0.7 and crest_factor > 2.2)
        return is_impulse, crest_factor, spectral_flux

    def monitor_frame(
        self,
        primary_frame: np.ndarray,
        enhanced_frame: np.ndarray,
        reference_frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive frame monitor returning detailed health diagnostics.
        """
        # 1. Impulse check on primary / reference
        is_impulse_prim, cf_prim, flux_prim = self.detect_impulse(primary_frame)
        is_impulse_ref, cf_ref, _ = self.detect_impulse(reference_frame)
        impulse_active = is_impulse_prim or is_impulse_ref

        # 2. Divergence check
        safe_audio, is_diverged = self.verify_frame(primary_frame, enhanced_frame)

        # 3. Peak limiting
        peak = float(np.max(np.abs(safe_audio)))
        limiter_active = peak > self.clip_threshold
        if limiter_active:
            safe_audio = np.tanh(safe_audio / self.clip_threshold) * self.clip_threshold

        return {
            "impulse_detected": impulse_active,
            "freeze_adaptation": impulse_active or is_diverged,
            "crest_factor": max(cf_prim, cf_ref),
            "spectral_flux": flux_prim,
            "is_diverged": is_diverged,
            "limiter_active": limiter_active,
            "safe_audio": safe_audio.astype(np.float32)
        }

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
