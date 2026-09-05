"""
Abstract Base Class for Neural Speech Enhancement Models.
PS 26052 — Adaptive Defence ANC.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class SpeechEnhancementModel(ABC):
    """Unified interface for deep learning speech enhancement backends."""

    @abstractmethod
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a single time-domain frame or STFT spectrum.

        Args:
            frame: 1D array of audio samples (frame_length).

        Returns:
            enhanced_frame: 1D array of enhanced audio samples.
        """
        pass

    @abstractmethod
    def enhance_spectrogram(self, mag_spec: np.ndarray, phase: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply spectral mask or complex enhancement.

        Args:
            mag_spec: 2D magnitude spectrogram (freq_bins, time_frames).
            phase: 2D phase angle spectrogram.

        Returns:
            Tuple of (enhanced_mag_spec, enhanced_phase).
        """
        pass
