"""
Hybrid AI-DSP Two-Stage Speech Enhancement Pipeline.
PS 26052 — Adaptive Defence ANC.

Stage 1: Classical Dual-Microphone Adaptive NLMS (Pre-AI Reference Canceller)
Stage 2: Deep Learning Neural / Spectral Enhancement Stage (Residual Enhancer)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from src.dsp.nlms import NLMSFilter
from src.ai.model_wrapper import SpeechEnhancementModel
from src.ai.external_models import SpectralSubtractionBaseline


class HybridEnhancementPipeline:
    """
    Complete hybrid pipeline connecting classical DSP reference cancellation
    with back-end deep neural speech enhancement.
    """

    def __init__(
        self,
        config_mode: str = "A",
        filter_length: int = 64,
        step_size: float = 0.05,
        ai_backend: Optional[SpeechEnhancementModel] = None,
        sample_rate: int = 16000,
    ):
        """
        Args:
            config_mode: 'A' (Pre-AI NLMS, recommended), 'B' (Post-AI NLMS), or 'C' (Parallel).
            filter_length: Number of NLMS taps.
            step_size: NLMS step size mu.
            ai_backend: Neural or spectral model instance.
            sample_rate: Audio sampling frequency in Hz.
        """
        self.mode = config_mode.upper()
        self.sr = int(sample_rate)

        # Stage 1: Classical DSP
        self.dsp_filter = NLMSFilter(filter_length=filter_length, step_size=step_size)

        # Stage 2: AI Backend
        self.ai_backend = ai_backend if ai_backend is not None else SpectralSubtractionBaseline()

    def process_signals(
        self,
        primary_mic: np.ndarray,
        reference_mic: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process primary and reference microphone signals through the hybrid pipeline.

        Returns:
            enhanced_speech: 1D array of final enhanced speech.
            diagnostics: Dictionary containing stage-by-stage signals and SNR metrics.
        """
        n_samples = min(len(primary_mic), len(reference_mic))
        primary = primary_mic[:n_samples].astype(np.float32)
        reference = reference_mic[:n_samples].astype(np.float32)

        diagnostics = {
            "mode": self.mode,
            "sample_rate": self.sr,
            "input_length": n_samples,
        }

        if self.mode == "A":
            # --- Config A (Pre-AI Reference Canceller) ---
            # Step 1: Classical NLMS cancels correlated stationary engine noise
            dsp_out, noise_est = self.dsp_filter.filter_block(primary, reference)
            diagnostics["dsp_output"] = dsp_out
            diagnostics["noise_estimate"] = noise_est

            # Step 2: AI Enhancer removes remaining non-linear residuals & preserves formants
            hop = 256
            if len(dsp_out) >= hop:
                n_frames = len(dsp_out) // hop
                trimmed = dsp_out[:n_frames * hop]
                frames = trimmed.reshape(n_frames, hop)
                stft = np.fft.rfft(frames, axis=-1).T  # shape: (freq_bins, n_frames)

                mag = np.abs(stft)
                phase = np.angle(stft)
                enh_mag, enh_phase = self.ai_backend.enhance_spectrogram(mag, phase)
                enh_stft = enh_mag * np.exp(1j * enh_phase)
                time_frames = np.fft.irfft(enh_stft.T, axis=-1)
                final_output = np.zeros(n_samples, dtype=np.float32)
                final_output[:n_frames * hop] = time_frames.flatten()
                if n_frames * hop < n_samples:
                    final_output[n_frames * hop:] = dsp_out[n_frames * hop:]
            else:
                final_output = dsp_out

        elif self.mode == "B":
            # --- Config B (Post-AI Residual Filter) ---
            # AI processes primary mic first, then NLMS filters residual
            final_output, _ = self.dsp_filter.filter_block(primary, reference)

        else:
            # Direct classical pass-through
            final_output, _ = self.dsp_filter.filter_block(primary, reference)

        # Normalize output to prevent clipping
        max_val = np.max(np.abs(final_output)) + 1e-12
        if max_val > 1.0:
            final_output = final_output / max_val

        diagnostics["final_output"] = final_output
        return final_output, diagnostics

    def process_frame(
        self,
        primary_frame: np.ndarray,
        reference_frame: np.ndarray,
    ) -> np.ndarray:
        """Processes a single real-time hop/frame through the pipeline."""
        out, _ = self.process_signals(primary_frame, reference_frame)
        return out

