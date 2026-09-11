"""
PS 26052: DRDO Adaptive ANC — Physical Secondary-Path Identification (PH5.3)
=============================================================================
Implements Angelo Farina's Logarithmic Swept-Sine deconvolution technique
to identify the secondary acoustic path S(z) = DAC -> Amp -> Speaker -> Cavity -> Mic -> ADC.
Extracts:
1. Physical causal impulse response s(t) (64 and 128 taps).
2. Frequency magnitude response |S(f)| in dB.
3. Unwrapped phase response ∠S(f) in degrees.
4. Total Harmonic Distortion (THD) separation from the linear response.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .duplex_audio import DuplexAudioEngine


class SecondaryPathMeasurer:
    """
    Identifies the secondary transfer function S(z) using Farina logarithmic swept-sine.
    """

    def __init__(
        self,
        engine: DuplexAudioEngine,
        f_start: float = 50.0,
        f_end: float = 4000.0,
        sweep_duration_sec: float = 1.0,
        filter_taps: int = 128,
    ):
        self.engine = engine
        self.sr = engine.sample_rate
        self.f1 = float(f_start)
        self.f2 = float(min(f_end, self.sr * 0.45))
        self.duration = float(sweep_duration_sec)
        self.filter_taps = int(filter_taps)

    def generate_log_sweep_and_inverse(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Farina logarithmic swept-sine and its analytic inverse filter.

        Returns
        -------
        sweep : np.ndarray (1D float32)
        inverse_filter : np.ndarray (1D float32)
        """
        sr = self.sr
        T = self.duration
        f1 = self.f1
        f2 = self.f2

        N = int(sr * T)
        t = np.linspace(0, T, N, endpoint=False)

        # Farina parameter L
        w1 = 2.0 * np.pi * f1
        w2 = 2.0 * np.pi * f2
        L = T / np.log(w2 / w1)

        # Instantaneous phase
        phi = w1 * L * (np.exp(t / L) - 1.0)
        sweep = np.sin(phi)

        # Smooth taper at edges (Hann fade-in / fade-out over 20 ms)
        fade_samples = int(0.02 * sr)
        if fade_samples > 0 and 2 * fade_samples < N:
            fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
            sweep[:fade_samples] *= fade_in
            sweep[-fade_samples:] *= fade_in[::-1]

        # Inverse filter: time-reversed sweep with exponential amplitude decay (-6 dB/octave)
        m = np.exp(-t / L)
        inv_filter = sweep[::-1] * m

        # Fast FFT convolution to normalize so direct conv has peak = 1.0
        n_fft = 2 ** int(np.ceil(np.log2(2 * N)))
        S = np.fft.rfft(sweep, n=n_fft)
        K = np.fft.rfft(inv_filter, n=n_fft)
        conv_direct = np.fft.irfft(S * K, n=n_fft)
        peak_val = np.max(np.abs(conv_direct))
        if peak_val > 1e-12:
            inv_filter = inv_filter / peak_val

        return sweep.astype(np.float32), inv_filter.astype(np.float32)

    @staticmethod
    def deconvolve(
        response: np.ndarray, inverse_filter: np.ndarray
    ) -> np.ndarray:
        """
        Convolve measured microphone response with inverse filter via FFT.
        """
        y = np.asarray(response, dtype=np.float64)
        k = np.asarray(inverse_filter, dtype=np.float64)

        n_fft = 2 ** int(np.ceil(np.log2(len(y) + len(k))))
        Y = np.fft.rfft(y, n=n_fft)
        K = np.fft.rfft(k, n=n_fft)
        ir = np.fft.irfft(Y * K, n=n_fft)
        return ir

    def measure(
        self,
        record_extra_sec: float = 0.3,
        amplitude: float = 0.5,
    ) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """
        Play Farina swept-sine through secondary speaker, record error microphone,
        and extract impulse response S(z).
        """
        sweep, inv_filter = self.generate_log_sweep_and_inverse()
        playback = (amplitude * sweep).astype(np.float32)

        # Full-duplex synchronous play and record
        response, telemetry = self.engine.play_and_record(
            playback, record_extra_seconds=record_extra_sec
        )

        # Deconvolve to obtain full response
        full_ir = self.deconvolve(response, inv_filter)

        # Find the theoretical zero-latency Dirac delta arrival
        # Compute direct conv peak of sweep and inv_filter
        n_fft = 2 ** int(np.ceil(np.log2(len(sweep) + len(inv_filter))))
        dir_conv = np.fft.irfft(np.fft.rfft(sweep, n_fft) * np.fft.rfft(inv_filter, n_fft), n_fft)
        t_ref_zero = int(np.argmax(np.abs(dir_conv)))

        # Find the physical onset peak in the response following t_ref_zero
        search_start = max(0, t_ref_zero - 16)
        search_end = min(len(full_ir), t_ref_zero + int(0.3 * self.sr))
        search_region = np.abs(full_ir[search_start:search_end])
        local_peak_idx = int(np.argmax(search_region))
        global_peak_idx = search_start + local_peak_idx

        # Find first sample exceeding 15% peak to define physical acoustic arrival
        threshold = 0.15 * np.abs(full_ir[global_peak_idx])
        onset_idx = global_peak_idx
        for i in range(global_peak_idx, max(0, global_peak_idx - 64), -1):
            if np.abs(full_ir[i]) < threshold:
                onset_idx = i
                break

        # Extract 64-tap and 128-tap causal secondary-path FIR filters
        s_64 = full_ir[onset_idx : onset_idx + 64].copy()
        s_128 = full_ir[onset_idx : onset_idx + 128].copy()
        if len(s_64) < 64:
            s_64 = np.pad(s_64, (0, 64 - len(s_64)))
        if len(s_128) < 128:
            s_128 = np.pad(s_128, (0, 128 - len(s_128)))

        # Selected FIR model based on user tap count
        s_fir = s_128 if self.filter_taps >= 128 else s_64
        # Window FIR tail gently with Hann half-window to avoid truncation splatter
        tail_len = min(16, len(s_fir) // 4)
        s_fir_windowed = s_fir.copy()
        s_fir_windowed[-tail_len:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(tail_len) / tail_len))

        # Frequency Response S(f)
        n_freq = 512
        freqs = np.fft.rfftfreq(n_freq, d=1.0 / self.sr)
        S_fft = np.fft.rfft(s_fir_windowed, n=n_freq)
        mag_db = 20.0 * np.log10(np.abs(S_fft) + 1e-12)
        phase_deg = np.rad2deg(np.unwrap(np.angle(S_fft)))

        # Acoustic transport delay (samples between t_ref_zero and onset)
        transport_delay_samples = max(0, onset_idx - t_ref_zero)
        transport_delay_ms = (transport_delay_samples / self.sr) * 1000.0

        # Impulse response SNR
        signal_energy = np.sum(s_fir ** 2)
        noise_window = full_ir[:max(10, t_ref_zero - 100)]
        noise_energy = np.mean(noise_window ** 2) if len(noise_window) > 0 else 1e-12
        ir_snr_db = 10.0 * np.log10((signal_energy / len(s_fir)) / (noise_energy + 1e-12))

        return {
            "sample_rate": float(self.sr),
            "filter_taps": int(self.filter_taps),
            "sweep_stimulus": playback,
            "recorded_response": response,
            "full_ir": full_ir,
            "t_ref_zero": int(t_ref_zero),
            "onset_index": int(onset_idx),
            "transport_delay_samples": int(transport_delay_samples),
            "transport_delay_ms": float(transport_delay_ms),
            "s_fir_64": s_64.astype(np.float32),
            "s_fir_128": s_128.astype(np.float32),
            "s_fir": s_fir_windowed.astype(np.float32),
            "freqs": freqs.astype(np.float32),
            "mag_db": mag_db.astype(np.float32),
            "phase_deg": phase_deg.astype(np.float32),
            "ir_snr_db": float(ir_snr_db),
            "telemetry": telemetry,
        }
