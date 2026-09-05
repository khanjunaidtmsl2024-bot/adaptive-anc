"""
Audio IO Abstraction Layer — Base Interfaces.
PS 26052 — Adaptive Defence ANC.

Decouples the DSP/AI ANC processing core from underlying audio capture
and playback backends (WAV files, synthetic generators, ALSA/I2S on Pi 4).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class HardwareUnavailableError(RuntimeError):
    """Raised when physical hardware backend (e.g. ALSA/WM8960) is invoked on non-target OS."""
    pass


class AudioSource(ABC):
    """Abstract audio input source providing synchronized dual-channel microphone chunks."""

    @abstractmethod
    def open(self) -> None:
        """Initialize and open the audio stream."""
        pass

    @abstractmethod
    def read_chunk(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the next chunk of audio.
        Returns:
            (primary_chunk, reference_chunk) as float32 arrays of shape (n_samples,).
        """
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Returns True if the stream has more audio to deliver."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release audio stream and hardware resources."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AudioSink(ABC):
    """Abstract audio output sink consuming processed single-channel speech chunks."""

    @abstractmethod
    def open(self) -> None:
        """Initialize and open the output audio stream."""
        pass

    @abstractmethod
    def write_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Write a processed audio chunk to the sink.
        Args:
            audio_chunk: (n_samples,) float32 enhanced speech samples.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Flush buffers and release output stream resources."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
