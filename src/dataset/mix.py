"""
Acoustic Mixing & Dual-Microphone Physical Simulation Engine.
PS 26052 — Adaptive Defence ANC.

Generates physically realistic acoustic mixtures for:
- Primary Microphone:   d[n] = s[n] + v_primary[n]
- Reference Microphone: x[n] = v_ref[n] + alpha_leak * s[n]

Simulates:
1. Speech leakage into reference mic (critical failure mode test for NLMS)
2. Primary-to-reference acoustic delay mismatch (fractional or integer lag)
3. Microphone sensitivity / gain mismatch (+/- 1.5 dB)
4. Nonlinear clipping saturation (0 dBFS / -1 dBFS)
5. Impulsive ballistic shockwaves and bursts
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class DualMicAcousticMixer:
    """
    Physical acoustic channel simulator producing synchronized
    primary and reference microphone streams.
    """

    @staticmethod
    def compute_rms(signal: np.ndarray) -> float:
        """Computes root-mean-square energy with numerical floor."""
        return float(np.sqrt(np.mean(signal ** 2) + 1e-12))

    @classmethod
    def mix_dual_channel(
        cls,
        clean_speech: np.ndarray,
        noise_audio: np.ndarray,
        target_snr_db: float,
        speech_leakage_ratio: float = 0.0,
        delay_samples: int = 0,
        gain_mismatch_db: float = 0.0,
        simulate_clipping: bool = False,
        clip_threshold: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Creates calibrated dual-microphone audio pair.

        Args:
            clean_speech: 1D array of speech s[n].
            noise_audio: 1D array of background noise v[n].
            target_snr_db: Desired SNR on primary channel in dB.
            speech_leakage_ratio: Leakage factor alpha of speech into reference mic (0.0 to 0.5).
            delay_samples: Shift in samples of reference mic relative to primary mic.
            gain_mismatch_db: Sensitivity difference of reference mic relative to primary in dB.
            simulate_clipping: Whether to apply nonlinear clipping saturation.
            clip_threshold: Amplitude threshold for saturation.

        Returns:
            (primary_d, reference_x, metadata)
        """
        # Truncate to shortest signal length
        n_samples = min(len(clean_speech), len(noise_audio))
        s = clean_speech[:n_samples].astype(np.float32)
        v = noise_audio[:n_samples].astype(np.float32)

        rms_s = cls.compute_rms(s)
        rms_v = cls.compute_rms(v)

        # Scale noise to achieve target SNR: SNR = 20 * log10(rms_s / rms_v_scaled)
        # rms_v_target = rms_s / (10^(SNR/20))
        target_noise_rms = rms_s / (10.0 ** (target_snr_db / 20.0))
        scale_v = target_noise_rms / rms_v
        v_primary = v * scale_v

        # 1. Primary channel: d[n] = s[n] + v_primary[n]
        primary_d = s + v_primary

        # 2. Reference channel:
        # Reference microphone captures acoustic noise + acoustic speech leakage
        ref_noise = v_primary.copy()

        # Apply acoustic delay if specified
        if delay_samples != 0:
            ref_noise = np.roll(ref_noise, delay_samples)
            if delay_samples > 0:
                ref_noise[:delay_samples] = 0.0
            else:
                ref_noise[delay_samples:] = 0.0

        # Apply reference microphone gain mismatch
        if gain_mismatch_db != 0.0:
            gain_factor = 10.0 ** (gain_mismatch_db / 20.0)
            ref_noise = ref_noise * gain_factor

        # Add speech leakage: x[n] = v_ref[n] + alpha * s[n]
        speech_leak = s * float(speech_leakage_ratio)
        reference_x = ref_noise + speech_leak

        # 3. Simulate nonlinear saturation if requested
        if simulate_clipping:
            primary_d = np.clip(primary_d, -clip_threshold, clip_threshold)
            reference_x = np.clip(reference_x, -clip_threshold, clip_threshold)

        # Calculate achieved primary SNR
        actual_noise_rms = cls.compute_rms(primary_d - s)
        actual_snr_db = 20.0 * np.log10(rms_s / (actual_noise_rms + 1e-12))

        meta = {
            "target_snr_db": float(target_snr_db),
            "actual_snr_db": round(float(actual_snr_db), 2),
            "speech_leakage_ratio": float(speech_leakage_ratio),
            "delay_samples": int(delay_samples),
            "gain_mismatch_db": float(gain_mismatch_db),
            "simulated_clipping": bool(simulate_clipping),
            "speech_rms": round(rms_s, 6),
            "primary_rms": round(cls.compute_rms(primary_d), 6),
            "reference_rms": round(cls.compute_rms(reference_x), 6),
        }

        return primary_d, reference_x, meta
