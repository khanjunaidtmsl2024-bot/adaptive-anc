"""
Real-Time Overlap-Add STFT / iSTFT Streaming Engine.
PS 26052 — Adaptive Defence ANC.

Ensures continuous streaming audio execution with exactly 50% overlap:
    frame_size = 512 samples (32 ms @ 16 kHz)
    hop_size   = 256 samples (16 ms @ 16 kHz)
    Algorithmic latency = 32.0 ms.
"""

from typing import Callable, Optional
import numpy as np


class StreamingSTFTEngine:
    """Frame-by-frame STFT/iSTFT processor with overlap-add synthesis."""

    def __init__(
        self,
        frame_size: int = 512,
        hop_size: int = 256,
        sample_rate: int = 16000,
        window_type: str = "hann"
    ):
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        self.sr = int(sample_rate)

        # Window functions
        if window_type == "hann":
            self.analysis_window = np.hanning(self.frame_size).astype(np.float32)
        else:
            self.analysis_window = np.ones(self.frame_size, dtype=np.float32)

        # Synthesis window normalization
        self.synthesis_window = self.analysis_window.copy()

        # Input buffer: holds past (frame_size - hop_size) samples
        self.input_buf = np.zeros(self.frame_size, dtype=np.float32)

        # Output accumulation buffer for overlap-add
        self.overlap_buf = np.zeros(self.frame_size, dtype=np.float32)

    def reset(self) -> None:
        """Reset all internal buffers to zero state."""
        self.input_buf[:] = 0.0
        self.overlap_buf[:] = 0.0

    def process_hop(
        self,
        new_samples: np.ndarray,
        spectrum_callback: Optional[Callable[[np.ndarray, np.ndarray], tuple]] = None
    ) -> np.ndarray:
        """
        Process exactly one hop_size chunk of incoming audio.

        Args:
            new_samples: 1D array of length hop_size.
            spectrum_callback: Function receiving (mag, phase) and returning enhanced (mag, phase).

        Returns:
            out_hop: 1D array of length hop_size (synthesized enhanced audio).
        """
        assert len(new_samples) == self.hop_size, f"Expected {self.hop_size} samples, got {len(new_samples)}"

        # Shift input buffer and insert new hop
        self.input_buf[:-self.hop_size] = self.input_buf[self.hop_size:]
        self.input_buf[-self.hop_size:] = new_samples

        # Apply analysis window
        windowed_frame = self.input_buf * self.analysis_window

        # Real FFT
        spectrum = np.fft.rfft(windowed_frame)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Optional neural callback
        if spectrum_callback is not None:
            mag, phase = spectrum_callback(mag, phase)

        # Inverse FFT
        reconstructed = mag * np.exp(1j * phase)
        time_frame = np.fft.irfft(reconstructed, n=self.frame_size)

        # Apply synthesis window
        time_frame = time_frame * self.synthesis_window

        # Overlap-add into accumulation buffer
        self.overlap_buf += time_frame

        # Extract synthesized hop
        out_hop = self.overlap_buf[:self.hop_size].copy()

        # Shift overlap buffer
        self.overlap_buf[:-self.hop_size] = self.overlap_buf[self.hop_size:]
        self.overlap_buf[-self.hop_size:] = 0.0

        return out_hop.astype(np.float32)
