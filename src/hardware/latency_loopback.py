"""
PS 26052: DRDO Adaptive ANC — Physical ADC/DAC Loopback Latency Measurement (PH5.2)
=====================================================================================
Measures physical round-trip hardware latency (DAC buffer -> DAC -> physical loopback
-> ADC -> ADC buffer) using cross-correlation of a calibrated chirp / pulse probe.
Computes latency statistics (P50, P95, min, max, jitter std) across multiple trials.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .duplex_audio import DuplexAudioEngine


class LoopbackLatencyMeasurer:
    """
    Measures physical ADC/DAC round-trip latency using cross-correlation.
    """

    def __init__(
        self,
        engine: DuplexAudioEngine,
        probe_duration_sec: float = 0.05,
        f_low: float = 300.0,
        f_high: float = 3000.0,
    ):
        self.engine = engine
        self.sr = engine.sample_rate
        self.probe_duration = probe_duration_sec
        self.f_low = f_low
        self.f_high = f_high

    def generate_probe_signal(self) -> np.ndarray:
        """
        Generate a windowed linear chirp probe signal.
        Chirps provide high signal-to-noise ratio and immunity to DC-blocking
        filters compared to single-sample delta impulses.
        """
        n_samples = int(self.sr * self.probe_duration)
        t = np.linspace(0, self.probe_duration, n_samples, endpoint=False)
        # Linear frequency chirp
        phase = 2 * np.pi * (self.f_low * t + 0.5 * (self.f_high - self.f_low) * (t ** 2) / self.probe_duration)
        chirp = np.sin(phase)
        # Apply Tukey / Hann window to prevent spectral splatter
        window = np.hanning(n_samples)
        probe = (chirp * window).astype(np.float32)
        # Normalize to 0.7 to avoid any DAC clipping
        probe = 0.7 * probe / (np.max(np.abs(probe)) + 1e-12)
        return probe

    @staticmethod
    def compute_cross_correlation_delay(
        stimulus: np.ndarray, response: np.ndarray, sample_rate: int
    ) -> Tuple[float, float, np.ndarray]:
        """
        Compute lag between stimulus and response using normalized cross-correlation.

        Returns
        -------
        delay_samples : float (with sub-sample parabolic interpolation)
        delay_ms : float
        xcorr : np.ndarray (normalized cross-correlation array)
        """
        x = np.asarray(stimulus, dtype=np.float64)
        y = np.asarray(response, dtype=np.float64)

        if len(y) < len(x):
            pad = np.zeros(len(x) - len(y), dtype=np.float64)
            y = np.concatenate([y, pad])

        # Energy normalization
        norm_x = np.sqrt(np.sum(x ** 2)) + 1e-12
        norm_y = np.sqrt(np.sum(y ** 2)) + 1e-12

        # Fast FFT-based cross correlation
        n_fft = 2 ** int(np.ceil(np.log2(len(x) + len(y))))
        X = np.fft.rfft(x, n=n_fft)
        Y = np.fft.rfft(y, n=n_fft)
        xcorr = np.fft.irfft(Y * np.conj(X), n=n_fft)[:len(y)] / (norm_x * norm_y)

        peak_idx = int(np.argmax(xcorr))

        # Sub-sample parabolic interpolation if not on boundary
        sub_sample_delay = float(peak_idx)
        if 0 < peak_idx < len(xcorr) - 1:
            alpha = xcorr[peak_idx - 1]
            beta = xcorr[peak_idx]
            gamma = xcorr[peak_idx + 1]
            denom = 2.0 * (2.0 * beta - alpha - gamma)
            if abs(denom) > 1e-12:
                delta = (gamma - alpha) / denom
                sub_sample_delay = peak_idx + delta

        delay_ms = (sub_sample_delay / sample_rate) * 1000.0
        return sub_sample_delay, delay_ms, xcorr

    def measure_single_trial(
        self, silence_prefix_sec: float = 0.05
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Execute one physical play-and-record trial and extract loopback latency.
        """
        probe = self.generate_probe_signal()
        prefix_samples = int(silence_prefix_sec * self.sr)
        playback = np.zeros(prefix_samples + len(probe), dtype=np.float32)
        playback[prefix_samples : prefix_samples + len(probe)] = probe

        # Record physical response
        response, telemetry = self.engine.play_and_record(playback, record_extra_seconds=0.25)

        # Match probe within response
        sub_delay, delay_ms, xcorr = self.compute_cross_correlation_delay(
            probe, response, self.sr
        )
        # Deduct the silence prefix
        actual_delay_samples = sub_delay - prefix_samples
        actual_delay_ms = (actual_delay_samples / self.sr) * 1000.0

        peak_corr = float(np.max(xcorr))

        return {
            "delay_samples": actual_delay_samples,
            "delay_ms": actual_delay_ms,
            "peak_correlation": peak_corr,
            "probe": probe,
            "playback": playback,
            "response": response,
            "xcorr": xcorr,
            "telemetry": telemetry,
        }

    def measure_multi_trial(
        self, num_trials: int = 10, pause_between_sec: float = 0.05
    ) -> Dict[str, Union[float, List[float], np.ndarray]]:
        """
        Execute multiple loopback trials and compute statistical distribution.
        """
        delays_ms = []
        delays_samples = []
        correlations = []
        first_trial_data = None

        for trial in range(num_trials):
            data = self.measure_single_trial()
            if first_trial_data is None:
                first_trial_data = data
            delays_ms.append(data["delay_ms"])
            delays_samples.append(data["delay_samples"])
            correlations.append(data["peak_correlation"])
            if pause_between_sec > 0:
                time.sleep(pause_between_sec)

        arr_ms = np.array(delays_ms)
        arr_samp = np.array(delays_samples)

        summary = {
            "num_trials": int(num_trials),
            "sample_rate": float(self.sr),
            "p50_latency_ms": float(np.median(arr_ms)),
            "p95_latency_ms": float(np.percentile(arr_ms, 95)),
            "min_latency_ms": float(np.min(arr_ms)),
            "max_latency_ms": float(np.max(arr_ms)),
            "std_latency_ms": float(np.std(arr_ms)),
            "p50_latency_samples": float(np.median(arr_samp)),
            "mean_peak_correlation": float(np.mean(correlations)),
            "trials_ms": [float(x) for x in delays_ms],
            "first_trial_stimulus": first_trial_data["playback"],
            "first_trial_response": first_trial_data["response"],
            "first_trial_xcorr": first_trial_data["xcorr"],
        }
        return summary
