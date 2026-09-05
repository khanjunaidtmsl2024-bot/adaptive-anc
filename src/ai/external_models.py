"""
External Model Loader & Baseline Neural / Spectral Enhancer.
PS 26052 — Adaptive Defence ANC.
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np

from src.ai.model_wrapper import SpeechEnhancementModel
from src.ai.tiny_enhancer import TinyEnhancerWrapper, TORCH_AVAILABLE


class SpectralSubtractionBaseline(SpeechEnhancementModel):
    """
    Classical spectral subtraction baseline (Boll 1979) used for verification
    and zero-dependency benchmarking when deep learning models are not loaded.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 0.02):
        self.alpha = float(alpha)  # Over-subtraction factor
        self.beta = float(beta)    # Spectral floor factor
        self.noise_profile = None

    def estimate_noise(self, noise_mag: np.ndarray) -> None:
        self.noise_profile = np.mean(noise_mag, axis=-1, keepdims=True)

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def enhance_spectrogram(
        self,
        mag_spec: np.ndarray,
        phase: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.noise_profile is None:
            # Estimate noise from initial 5 frames
            self.noise_profile = np.mean(mag_spec[:, :min(5, mag_spec.shape[1])], axis=-1, keepdims=True)

        subtracted = mag_spec - self.alpha * self.noise_profile
        floor = self.beta * mag_spec
        enhanced_mag = np.maximum(subtracted, floor)

        return enhanced_mag.astype(np.float32), phase


def load_model_checkpoint(model_name: str, checkpoint_path: Optional[str] = None) -> SpeechEnhancementModel:
    """Factory function to load neural or baseline speech enhancement model."""
    if model_name.lower() == "tiny_enhancer" and TORCH_AVAILABLE and checkpoint_path and Path(checkpoint_path).exists():
        return TinyEnhancerWrapper(checkpoint_path=checkpoint_path)
    return SpectralSubtractionBaseline()
