"""
PS 26052: DRDO Adaptive ANC — Open-Loop Anti-Noise & Plant Verification (PH5.6)
================================================================================
Validates the physical electro-acoustic chain before closing the adaptive loop:
1. Verifies the secondary speaker reliably produces physical acoustic emission.
2. Verifies the error microphone detects the emission with adequate SNR.
3. Tests polarity / sign convention (software phase inversion -> acoustic inversion).
4. Compares measured error microphone waveform with model prediction (s_measured * y).
5. Quantifies Total Harmonic Distortion (THD) of the transducer path under drive.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .duplex_audio import DuplexAudioEngine


class OpenLoopAntiNoiseTester:
    """
    Executes open-loop diagnostic anti-noise tests to confirm hardware integrity
    and sign/phase convention prior to closed-loop adaptation.
    """

    def __init__(
        self,
        engine: DuplexAudioEngine,
        secondary_path_fir: np.ndarray,
        test_frequency: float = 200.0,
        test_duration_sec: float = 1.0,
    ):
        self.engine = engine
        self.sr = engine.sample_rate
        self.s_fir = np.asarray(secondary_path_fir, dtype=np.float32)
        self.f0 = float(test_frequency)
        self.duration = float(test_duration_sec)

    def generate_tone(self, amplitude: float = 0.5, phase_deg: float = 0.0) -> np.ndarray:
        """Generate pure sinusoidal anti-noise test tone."""
        n_samples = int(self.sr * self.duration)
        t = np.linspace(0, self.duration, n_samples, endpoint=False)
        phase_rad = np.deg2rad(phase_deg)
        tone = amplitude * np.sin(2.0 * np.pi * self.f0 * t + phase_rad)
        # Apply 20 ms Hann fade in/out
        fade = int(0.02 * self.sr)
        if fade > 0:
            window = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade) / fade))
            tone[:fade] *= window
            tone[-fade:] *= window[::-1]
        return tone.astype(np.float32)

    @staticmethod
    def compute_thd(signal: np.ndarray, sample_rate: int, fundamental_freq: float) -> float:
        """Compute Total Harmonic Distortion (THD) percentage of a recorded tone."""
        sig = np.asarray(signal, dtype=np.float64)
        if len(sig) < 256:
            return 0.0
        # Windowed FFT
        win = np.hanning(len(sig))
        sig_win = sig * win
        n_fft = len(sig)
        fft_mag = np.abs(np.fft.rfft(sig_win))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        # Fundamental bin
        k1 = int(np.argmin(np.abs(freqs - fundamental_freq)))
        # Search around k1 for peak
        k1_peak = k1 - 2 + int(np.argmax(fft_mag[max(0, k1 - 2) : min(len(fft_mag), k1 + 3)]))
        v1 = fft_mag[k1_peak]

        if v1 < 1e-12:
            return 0.0

        # Harmonics (2nd through 5th)
        harmonic_powers = 0.0
        for h in range(2, 6):
            fh = h * fundamental_freq
            if fh >= sample_rate * 0.48:
                break
            kh = int(np.argmin(np.abs(freqs - fh)))
            kh_peak = kh - 2 + int(np.argmax(fft_mag[max(0, kh - 2) : min(len(fft_mag), kh + 3)]))
            harmonic_powers += fft_mag[kh_peak] ** 2

        thd = np.sqrt(harmonic_powers) / v1
        return float(thd * 100.0)

    def test_open_loop_response(
        self, amplitude: float = 0.5
    ) -> Dict[str, Union[float, bool, np.ndarray, Dict]]:
        """
        Execute open-loop anti-noise tests with normal (0 deg) and inverted (180 deg) phases.
        """
        # 1. Measure ambient noise baseline (silence)
        silence_probe = np.zeros(int(0.5 * self.sr), dtype=np.float32)
        ambient_rec, _ = self.engine.play_and_record(silence_probe, record_extra_seconds=0.1)
        ambient_rms = float(np.sqrt(np.mean(ambient_rec ** 2))) + 1e-12

        # 2. Test Tone at Phase = 0 deg
        tone_0 = self.generate_tone(amplitude=amplitude, phase_deg=0.0)
        rec_0, tele_0 = self.engine.play_and_record(tone_0, record_extra_seconds=0.2)

        # 3. Test Tone at Phase = 180 deg (Acoustic Inversion check)
        tone_180 = self.generate_tone(amplitude=amplitude, phase_deg=180.0)
        rec_180, tele_180 = self.engine.play_and_record(tone_180, record_extra_seconds=0.2)

        # Signal levels
        rec_0_rms = float(np.sqrt(np.mean(rec_0 ** 2)))
        rec_180_rms = float(np.sqrt(np.mean(rec_180 ** 2)))
        snr_db = 20.0 * np.log10(rec_0_rms / ambient_rms)

        # 4. Predict expected response using the identified plant S(z)
        pred_0 = np.convolve(tone_0, self.s_fir, mode="full")

        # Trim / align predicted and measured responses
        min_len = min(len(rec_0), len(pred_0))
        rec_trim = rec_0[:min_len]
        pred_trim = pred_0[:min_len]

        # Normalized cross-correlation between predicted and measured
        norm_rec = np.linalg.norm(rec_trim) + 1e-12
        norm_pred = np.linalg.norm(pred_trim) + 1e-12
        xcorr = np.correlate(rec_trim, pred_trim, mode="full") / (norm_rec * norm_pred)
        model_match_correlation = float(np.max(np.abs(xcorr)))

        # 5. Verify Polarity / Inversion symmetry: rec_0 and rec_180 should be anti-phase
        min_rec_len = min(len(rec_0), len(rec_180))
        rec_0_cut = rec_0[:min_rec_len]
        rec_180_cut = rec_180[:min_rec_len]
        inversion_correlation = float(
            np.dot(rec_0_cut, rec_180_cut)
            / (np.linalg.norm(rec_0_cut) * np.linalg.norm(rec_180_cut) + 1e-12)
        )

        # A perfect phase inversion gives correlation close to -1.0
        polarity_verified = inversion_correlation < -0.70

        # 6. THD calculation
        thd_percent = self.compute_thd(rec_0, self.sr, self.f0)

        # Hardware integrity check
        hardware_ok = (snr_db >= 10.0) and polarity_verified and (thd_percent < 25.0)

        return {
            "test_frequency_hz": float(self.f0),
            "test_amplitude": float(amplitude),
            "ambient_rms": float(ambient_rms),
            "signal_rms": float(rec_0_rms),
            "snr_db": float(snr_db),
            "inversion_correlation": float(inversion_correlation),
            "polarity_verified": bool(polarity_verified),
            "model_match_correlation": float(model_match_correlation),
            "thd_percent": float(thd_percent),
            "hardware_integrity_passed": bool(hardware_ok),
            "tone_stimulus": tone_0,
            "response_0_deg": rec_0,
            "response_180_deg": rec_180,
            "predicted_response": pred_0,
        }
