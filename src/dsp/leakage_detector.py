"""
Reference-Microphone Speech Leakage & Coherence Gating Engine.
PS 26052 — Adaptive Defence ANC.

Protects against catastrophic speech cancellation when speech leaks into the
reference microphone channel (x[n] = v_ref[n] + alpha * s[n]).
Uses cross-channel power ratios, voice-band coherence, and speech energy tracking
to dynamically attenuate the adaptation rate mu rather than naively freezing via binary VAD.
"""

from typing import Tuple, Dict, Any
import numpy as np


class SpeechLeakageDetector:
    """
    Evaluates speech leakage probability and calculates adaptation gating factor.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        leakage_threshold_ratio: float = 1.8,
        voice_band: Tuple[int, int] = (300, 3400),
        smoothing_alpha: float = 0.85,
    ):
        """
        Args:
            sample_rate: Sampling frequency in Hz.
            leakage_threshold_ratio: Primary/Reference energy ratio above which leakage is suspected.
            voice_band: Frequency range (Hz) of human speech formants.
            smoothing_alpha: Time smoothing parameter for gating factor.
        """
        self.sr = int(sample_rate)
        self.threshold = float(leakage_threshold_ratio)
        self.f_min, self.f_max = voice_band
        self.alpha = float(smoothing_alpha)

        self.smooth_gate = 1.0

    def compute_frame_gating(
        self,
        primary_frame: np.ndarray,
        reference_frame: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates adaptation scaling factor g in [0.0, 1.0]:
        1.0 = Clean noise reference -> Full adaptation (mu = mu0).
        0.0 = High speech leakage -> Adaptation frozen (mu = 0) to protect speech.
        Intermediate values smoothly attenuate adaptation.
        """
        ep = float(np.mean(primary_frame ** 2) + 1e-12)
        er = float(np.mean(reference_frame ** 2) + 1e-12)

        # Energy ratio: Ep / Er
        energy_ratio = ep / er

        # Voice-band spectral energy check on primary
        n = len(primary_frame)
        fft_p = np.abs(np.fft.rfft(primary_frame))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sr)
        voice_mask = (freqs >= self.f_min) & (freqs <= self.f_max)

        voice_energy = np.sum(fft_p[voice_mask] ** 2)
        total_energy = np.sum(fft_p ** 2) + 1e-12
        voice_ratio = float(voice_energy / total_energy)

        # Calculate instantaneous gating factor
        # If primary has strong speech (voice_ratio > 0.4) AND primary energy dominates reference:
        if energy_ratio > self.threshold and voice_ratio > 0.35:
            # Speech is active on primary and potentially leaking into reference
            # Attenuate adaptation proportionally
            excess = energy_ratio / self.threshold
            inst_gate = float(np.clip(1.0 / (1.0 + excess), 0.0, 1.0))
        else:
            # Noise dominated: reference is pure environmental noise
            inst_gate = 1.0

        # Exponential time smoothing to prevent abrupt tap oscillations
        self.smooth_gate = self.alpha * self.smooth_gate + (1.0 - self.alpha) * inst_gate

        info = {
            "primary_energy": round(ep, 6),
            "reference_energy": round(er, 6),
            "energy_ratio": round(energy_ratio, 2),
            "voice_energy_ratio": round(voice_ratio, 3),
            "gating_factor": round(self.smooth_gate, 3),
            "leakage_detected": bool(self.smooth_gate < 0.6),
        }

        return float(self.smooth_gate), info
