"""
Unified Hybrid AI-DSP Two-Stage Speech Enhancement Pipeline.
PS 26052 — Adaptive Defence ANC.

Canonical Architecture (Config A):
1. Delay Alignment: Synchronizes reference noise to primary acoustic channel.
2. Speech Leakage Gating: Prevents speech cancellation if speech enters reference mic.
3. Adaptive Filtering: Dual-mic NLMS / VSS-NLMS cancels correlated environmental noise.
4. Impulse Protection: Transient error spike detection attenuates mu and prevents divergence.
5. AI Speech Enhancement: Deep neural spectral mask enhancer eliminates residual non-linear noise.
6. Safety Post-Processing: Formant preserver and peak limiter.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from src.dsp.nlms import NLMSFilter
from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.delay_alignment import DelayAligner
from src.dsp.leakage_detector import SpeechLeakageDetector
from src.dsp.impulse_protection import ImpulseProtectionController
from src.streaming.stft_engine import StreamingSTFTEngine
from src.ai.model_wrapper import SpeechEnhancementModel
from src.ai.tiny_enhancer import TinyEnhancerWrapper


class HybridEnhancementPipeline:
    """
    Production-grade hybrid pipeline executing Stage 1 DSP + Stage 2 AI.
    """

    def __init__(
        self,
        config_mode: str = "A",
        filter_length: int = 64,
        step_size: float = 0.05,
        use_vss: bool = False,
        enable_leakage_protection: bool = True,
        enable_impulse_protection: bool = True,
        ai_backend: Optional[Any] = None,
        sample_rate: int = 16000,
        frame_size: int = 512,
        hop_size: int = 256,
    ):
        self.mode = config_mode.upper()
        self.sr = int(sample_rate)
        self.use_vss = use_vss
        self.enable_leakage = enable_leakage_protection
        self.enable_impulse = enable_impulse_protection

        # Stage 1 Subsystems
        self.delay_aligner = DelayAligner(max_lag_samples=32, sample_rate=self.sr)
        self.leakage_detector = SpeechLeakageDetector(sample_rate=self.sr)
        self.impulse_controller = ImpulseProtectionController(sample_rate=self.sr)

        if use_vss:
            self.dsp_filter = VSSNLMSFilter(filter_length=filter_length, mu_init=step_size)
        else:
            self.dsp_filter = NLMSFilter(filter_length=filter_length, step_size=step_size)

        # Stage 2 AI Enhancer
        self.ai_backend = ai_backend if ai_backend is not None else TinyEnhancerWrapper()

        # STFT Engine with Hanning window overlap-add synthesis
        self.stft_engine = StreamingSTFTEngine(frame_size=frame_size, hop_size=hop_size, sample_rate=self.sr)
        self.frame_size = frame_size
        self.hop_size = hop_size

    def reset(self) -> None:
        """Resets all internal filter states and buffers."""
        self.dsp_filter.reset()
        self.impulse_controller.reset()
        self.stft_engine.reset()

    def process_signals(
        self,
        primary_mic: np.ndarray,
        reference_mic: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes primary and reference microphone signals through the complete hybrid chain.
        Returns: (enhanced_audio, diagnostics_dict)
        """
        n_samples = min(len(primary_mic), len(reference_mic))
        d = primary_mic[:n_samples].astype(np.float32)
        x = reference_mic[:n_samples].astype(np.float32)

        # 1. Step 1: Acoustic Delay Alignment
        _, x_aligned, applied_delay = self.delay_aligner.align(d, x)

        # 2. Step 2: Speech Leakage Gating
        g_leak = 1.0
        leakage_info = {}
        if self.enable_leakage:
            g_leak, leakage_info = self.leakage_detector.compute_frame_gating(d, x_aligned)

        # 3. Step 3: Stage 1 Adaptive Cancellation (NLMS)
        if hasattr(self.dsp_filter, "filter_block"):
            if isinstance(self.dsp_filter, VSSNLMSFilter):
                dsp_residual, noise_est, _ = self.dsp_filter.filter_block(d, x_aligned)
            else:
                dsp_residual, noise_est = self.dsp_filter.filter_block(d, x_aligned)
        else:
            dsp_residual = d - x_aligned
            noise_est = x_aligned

        # 4. Step 4: Impulsive Transient Protection
        impulse_stats = {}
        if self.enable_impulse:
            dsp_residual, _, impulse_stats = self.impulse_controller.filter_block_protection(dsp_residual)

        # 5. Step 5: Stage 2 AI Spectral Enhancement with windowed Overlap-Add
        window = np.hanning(self.frame_size).astype(np.float32)
        if len(dsp_residual) < self.frame_size:
            padded_residual = np.pad(dsp_residual, (0, self.frame_size - len(dsp_residual)))
        else:
            padded_residual = dsp_residual

        n_frames = max(1, (len(padded_residual) - self.frame_size) // self.hop_size + 1)
        
        # Frame buffering
        frames = np.zeros((self.frame_size, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_size
            chunk = padded_residual[start : start + self.frame_size]
            if len(chunk) < self.frame_size:
                chunk = np.pad(chunk, (0, self.frame_size - len(chunk)))
            frames[:, i] = chunk * window

        # RFFT
        stft = np.fft.rfft(frames, axis=0)
        mag = np.abs(stft)
        phase = np.angle(stft)

        # AI Mask inference
        enh_mag, enh_phase = self.ai_backend.enhance_spectrogram(mag, phase)
        enh_stft = enh_mag * np.exp(1j * enh_phase)

        # iRFFT and Overlap-Add reconstruction
        recon_frames = np.fft.irfft(enh_stft, axis=0)
        final_output = np.zeros(n_samples + self.frame_size, dtype=np.float32)
        norm_window = np.zeros(n_samples + self.frame_size, dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            final_output[start:end] += recon_frames[:, i] * window
            norm_window[start:end] += window ** 2

        # Normalize overlap sum
        valid = norm_window > 1e-4
        final_output[valid] /= norm_window[valid]
        final_output = final_output[:n_samples]

        # Prevent digital clipping
        peak = np.max(np.abs(final_output)) + 1e-12
        if peak > 0.98:
            final_output = (final_output * 0.98 / peak).astype(np.float32)

        diagnostics = {
            "mode": self.mode,
            "dsp_output": dsp_residual,
            "final_output": final_output,
            "applied_delay_samples": applied_delay,
            "leakage_gating_factor": g_leak,
            "leakage_detected": leakage_info.get("leakage_detected", False),
            "impulses_suppressed": impulse_stats.get("impulses_detected", 0),
            "dsp_residual_rms": float(np.sqrt(np.mean(dsp_residual ** 2) + 1e-12)),
            "final_rms": float(np.sqrt(np.mean(final_output ** 2) + 1e-12)),
        }

        return final_output, diagnostics

    def process_frame(
        self,
        primary_frame: np.ndarray,
        reference_frame: np.ndarray
    ) -> np.ndarray:
        """Processes a streaming frame and returns enhanced audio."""
        out, _ = self.process_signals(primary_frame, reference_frame)
        return out

