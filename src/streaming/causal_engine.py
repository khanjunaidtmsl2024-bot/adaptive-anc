"""
Causal Streaming Inference Engine.
PS 26052 -- Adaptive Defence ANC.

Processes audio in real-time (causal) mode:
  - Fixed frame size: 256 samples (16 ms @ 16 kHz) for sub-20ms latency
  - Overlap-Add with 50% overlap: hop_size = 128 samples (8 ms)
  - Total algorithmic latency: frame_size + processing = ~18 ms target
  - No future-frame lookahead (strictly causal)

Integrates:
  1. Noise Regime Detector (frame-by-frame)
  2. DSP Stage (VSS-NLMS with adaptive parameters from regime)
  3. AI Stage (spectral mask via selected backend)
  4. Performance profiling (per-frame timing)
"""

import time
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.leakage_detector import SpeechLeakageDetector
from src.dsp.impulse_protection import ImpulseProtectionController
from src.dsp.noise_regime_detector import NoiseRegimeDetector, NoiseRegime, REGIME_PRESETS
from src.ai.tiny_enhancer import TinyEnhancerWrapper


class CausalStreamingEngine:
    """
    Real-time causal streaming speech enhancement engine.

    Processes audio hop-by-hop with no lookahead.
    Dynamically adapts DSP parameters based on noise regime detection.
    """

    def __init__(
        self,
        frame_size: int = 256,
        hop_size: int = 128,
        sample_rate: int = 16000,
        filter_length: int = 64,
        step_size: float = 0.05,
        ai_backend: Optional[Any] = None,
        enable_regime_adaptation: bool = True,
    ):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sr = sample_rate
        self.enable_adaptation = enable_regime_adaptation

        # DSP components
        self.nlms = VSSNLMSFilter(filter_length=filter_length, mu_init=step_size)
        self.leakage_detector = SpeechLeakageDetector(sample_rate=sample_rate)
        self.impulse_controller = ImpulseProtectionController(sample_rate=sample_rate)
        self.regime_detector = NoiseRegimeDetector(sample_rate=sample_rate, frame_size=frame_size)

        # AI backend
        self.ai_backend = ai_backend if ai_backend is not None else TinyEnhancerWrapper()

        # Streaming buffers
        self.input_buf_primary = np.zeros(frame_size, dtype=np.float32)
        self.input_buf_reference = np.zeros(frame_size, dtype=np.float32)
        self.overlap_buf = np.zeros(frame_size, dtype=np.float32)

        # Analysis/synthesis window
        self.window = np.hanning(frame_size).astype(np.float32)

        # Performance tracking
        self.frame_count = 0
        self.total_dsp_time_ms = 0.0
        self.total_ai_time_ms = 0.0
        self.total_regime_time_ms = 0.0
        self.latency_samples = []
        self.current_regime = NoiseRegime.STATIONARY

    def reset(self) -> None:
        """Reset all internal state."""
        self.input_buf_primary[:] = 0
        self.input_buf_reference[:] = 0
        self.overlap_buf[:] = 0
        self.nlms.reset()
        self.impulse_controller.reset()
        self.regime_detector.reset()
        self.frame_count = 0
        self.total_dsp_time_ms = 0.0
        self.total_ai_time_ms = 0.0
        self.total_regime_time_ms = 0.0
        self.latency_samples = []

    def process_hop(
        self,
        primary_hop: np.ndarray,
        reference_hop: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process one hop of audio (strictly causal, no lookahead).

        Args:
            primary_hop: (hop_size,) float32 primary mic samples.
            reference_hop: (hop_size,) float32 reference mic samples.

        Returns:
            (output_hop, diagnostics) where output_hop is (hop_size,) float32.
        """
        t_total_start = time.perf_counter()
        self.frame_count += 1

        # Sanitize inputs: protect against NaNs, Infs, and extreme type drifts
        primary_hop = np.nan_to_num(primary_hop, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        reference_hop = np.nan_to_num(reference_hop, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        # Shift input buffers (FIFO)
        self.input_buf_primary[:-self.hop_size] = self.input_buf_primary[self.hop_size:]
        self.input_buf_primary[-self.hop_size:] = primary_hop

        self.input_buf_reference[:-self.hop_size] = self.input_buf_reference[self.hop_size:]
        self.input_buf_reference[-self.hop_size:] = reference_hop

        # Step 1: Noise Regime Detection
        t0 = time.perf_counter()
        regime, regime_diag = self.regime_detector.classify_frame(self.input_buf_primary)
        self.current_regime = regime
        t_regime = (time.perf_counter() - t0) * 1000.0
        self.total_regime_time_ms += t_regime

        # Adapt parameters if enabled
        if self.enable_adaptation:
            params = REGIME_PRESETS[regime]
            self.nlms.mu = params["nlms_mu"]

        # Step 2: DSP -- NLMS adaptive filtering
        t0 = time.perf_counter()
        dsp_out, _, _ = self.nlms.filter_block(self.input_buf_primary, self.input_buf_reference)
        # Impulse protection
        dsp_out, _, imp_stats = self.impulse_controller.filter_block_protection(dsp_out)
        t_dsp = (time.perf_counter() - t0) * 1000.0
        self.total_dsp_time_ms += t_dsp

        # Step 3: AI -- Spectral mask enhancement (causal: single-frame STFT)
        t0 = time.perf_counter()
        windowed = dsp_out * self.window
        stft_frame = np.fft.rfft(windowed)
        freq_bins = len(stft_frame)
        mag = np.abs(stft_frame).reshape(-1, 1)  # (F, 1) single frame
        phase = np.angle(stft_frame).reshape(-1, 1)

        enh_mag, enh_phase = self.ai_backend.enhance_spectrogram(mag, phase)

        # Ensure output is exactly (freq_bins,) for iRFFT
        enh_mag_1d = enh_mag.flatten()[:freq_bins]
        enh_phase_1d = enh_phase.flatten()[:freq_bins]
        enh_stft = enh_mag_1d * np.exp(1j * enh_phase_1d)
        recon = np.fft.irfft(enh_stft, n=self.frame_size) * self.window
        t_ai = (time.perf_counter() - t0) * 1000.0
        self.total_ai_time_ms += t_ai

        # Overlap-add
        self.overlap_buf += recon
        output_hop = self.overlap_buf[:self.hop_size].copy()
        self.overlap_buf[:-self.hop_size] = self.overlap_buf[self.hop_size:]
        self.overlap_buf[-self.hop_size:] = 0.0

        t_total = (time.perf_counter() - t_total_start) * 1000.0
        self.latency_samples.append(t_total)

        diagnostics = {
            "frame": self.frame_count,
            "regime": regime.value,
            "time_total_ms": round(t_total, 3),
            "time_dsp_ms": round(t_dsp, 3),
            "time_ai_ms": round(t_ai, 3),
            "time_regime_ms": round(t_regime, 3),
            "impulses_suppressed": imp_stats.get("impulses_detected", 0),
            "rtf_frame": round(t_total / 1000.0 / (self.hop_size / self.sr), 4),
        }

        return output_hop.astype(np.float32), diagnostics

    def process_signal(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a complete signal hop-by-hop in causal streaming mode.

        Returns (enhanced_signal, aggregate_diagnostics).
        """
        self.reset()
        n = min(len(primary), len(reference))
        n_hops = (n - self.frame_size) // self.hop_size + 1

        output = np.zeros(n, dtype=np.float32)
        regime_log = []

        for i in range(n_hops):
            start = i * self.hop_size
            p_hop = primary[start:start + self.hop_size]
            r_hop = reference[start:start + self.hop_size]

            if len(p_hop) < self.hop_size:
                p_hop = np.pad(p_hop, (0, self.hop_size - len(p_hop)))
                r_hop = np.pad(r_hop, (0, self.hop_size - len(r_hop)))

            out_hop, diag = self.process_hop(p_hop, r_hop)
            output[start:start + self.hop_size] = out_hop
            regime_log.append(diag["regime"])

        # Peak-limit
        peak = np.max(np.abs(output)) + 1e-12
        if peak > 0.98:
            output = output * 0.98 / peak

        # Aggregate stats
        latencies = np.array(self.latency_samples) if self.latency_samples else np.array([0.0])
        aggregate = {
            "total_frames": self.frame_count,
            "avg_latency_ms": round(float(np.mean(latencies)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
            "max_latency_ms": round(float(np.max(latencies)), 3),
            "avg_rtf": round(float(np.mean(latencies) / 1000.0 / (self.hop_size / self.sr)), 4),
            "avg_dsp_ms": round(self.total_dsp_time_ms / max(self.frame_count, 1), 3),
            "avg_ai_ms": round(self.total_ai_time_ms / max(self.frame_count, 1), 3),
            "avg_regime_ms": round(self.total_regime_time_ms / max(self.frame_count, 1), 3),
            "regime_distribution": {
                r.value: regime_log.count(r.value) / max(len(regime_log), 1)
                for r in NoiseRegime
            },
            "algorithmic_latency_ms": round(self.frame_size / self.sr * 1000, 1),
        }

        return output[:n], aggregate

    def get_latency_budget(self) -> Dict[str, float]:
        """Return the latency budget breakdown."""
        algo_lat = self.frame_size / self.sr * 1000.0
        latencies = np.array(self.latency_samples) if self.latency_samples else np.array([0.0])
        compute_lat = float(np.mean(latencies))
        return {
            "algorithmic_latency_ms": round(algo_lat, 1),
            "compute_latency_ms": round(compute_lat, 3),
            "total_e2e_latency_ms": round(algo_lat + compute_lat, 1),
            "target_met": (algo_lat + compute_lat) < 20.0,
        }
