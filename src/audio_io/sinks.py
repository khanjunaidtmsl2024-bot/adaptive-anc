"""
Audio IO Concrete Sinks: WAVSink, NullSink, ALSASink.
PS 26052 — Adaptive Defence ANC.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

from src.audio_io.base import AudioSink, HardwareUnavailableError


class WAVSink(AudioSink):
    """Writes processed audio chunks to a standard WAV audio file."""

    def __init__(self, output_path: str, sr: int = 16000):
        self.output_path = Path(output_path)
        self.sr = sr
        self.buffer = []

    def open(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer = []

    def write_chunk(self, audio_chunk: np.ndarray) -> None:
        self.buffer.append(audio_chunk.astype(np.float32))

    def close(self) -> None:
        if self.buffer:
            full_audio = np.concatenate(self.buffer, axis=0)
            # Clip to [-1.0, 1.0] for safety
            full_audio = np.clip(full_audio, -1.0, 1.0)
            sf.write(str(self.output_path), full_audio, self.sr)
        self.buffer = []


class NullSink(AudioSink):
    """Discards audio chunks. Ideal for latency profiling, stress testing, and headless benchmarks."""

    def __init__(self):
        self.total_chunks = 0
        self.total_samples = 0

    def open(self) -> None:
        self.total_chunks = 0
        self.total_samples = 0

    def write_chunk(self, audio_chunk: np.ndarray) -> None:
        self.total_chunks += 1
        self.total_samples += len(audio_chunk)

    def close(self) -> None:
        pass


class ALSASink(AudioSink):
    """
    Hardware ALSA I2S playback sink for Raspberry Pi 4 + WM8960 DAC output.
    Acts as a verified stub in pre-hardware simulation (PH0).
    """

    def __init__(self, device: str = "hw:1,0", sample_rate: int = 16000):
        self.device = device
        self.sr = sample_rate
        self._is_open = False

    def open(self) -> None:
        try:
            import alsaaudio
            self.stream = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK, alsaaudio.PCM_NORMAL, device=self.device)
            self.stream.setchannels(1)  # Mono enhanced speech output
            self.stream.setrate(self.sr)
            self.stream.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            self.stream.setperiodsize(128)
            self._is_open = True
        except (ImportError, Exception) as e:
            raise HardwareUnavailableError(
                f"ALSA hardware audio sink '{self.device}' is unavailable on this host platform. "
                f"Requires physical Linux Raspberry Pi 4 with WM8960 I2S driver: {e}"
            )

    def write_chunk(self, audio_chunk: np.ndarray) -> None:
        if not self._is_open:
            raise HardwareUnavailableError("ALSA playback stream is not open.")
        int16_samples = (np.clip(audio_chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        self.stream.write(int16_samples.tobytes())

    def close(self) -> None:
        if self._is_open:
            self.stream.close()
            self._is_open = False
