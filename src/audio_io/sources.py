"""
Audio IO Concrete Sources: WAVSource, SyntheticSource, LoopbackSource, ALSASource.
PS 26052 — Adaptive Defence ANC.
"""

from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, Any
import numpy as np
import soundfile as sf

from src.audio_io.base import AudioSource, HardwareUnavailableError
from src.dataset.synthetic_benchmark_matrix import (
    synthesize_speech_profile,
    synthesize_noise_profile,
    get_benchmark_pair,
)


class WAVSource(AudioSource):
    """Feeds primary and reference channels from dual WAV files."""

    def __init__(self, primary_path: str, reference_path: Optional[str] = None, sr: int = 16000):
        self.primary_path = Path(primary_path)
        self.reference_path = Path(reference_path) if reference_path else None
        self.target_sr = sr
        self.primary_data = None
        self.ref_data = None
        self.cursor = 0
        self.total_samples = 0

    def open(self) -> None:
        if not self.primary_path.exists():
            raise FileNotFoundError(f"Primary WAV file not found: {self.primary_path}")
        
        p_wav, p_sr = sf.read(str(self.primary_path), dtype="float32")
        self.primary_data = p_wav.flatten()
        self.total_samples = len(self.primary_data)

        if self.reference_path and self.reference_path.exists():
            r_wav, _ = sf.read(str(self.reference_path), dtype="float32")
            self.ref_data = r_wav.flatten()
        else:
            # Fallback: silence reference
            self.ref_data = np.zeros(self.total_samples, dtype=np.float32)

        self.cursor = 0

    def read_chunk(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_active():
            return np.zeros(n_samples, dtype=np.float32), np.zeros(n_samples, dtype=np.float32)

        start = self.cursor
        end = min(start + n_samples, self.total_samples)
        chunk_p = self.primary_data[start:end]
        chunk_r = self.ref_data[start:end]

        if len(chunk_p) < n_samples:
            pad = n_samples - len(chunk_p)
            chunk_p = np.pad(chunk_p, (0, pad))
            chunk_r = np.pad(chunk_r, (0, pad))

        self.cursor += n_samples
        return chunk_p.astype(np.float32), chunk_r.astype(np.float32)

    def is_active(self) -> bool:
        return self.cursor < self.total_samples

    def close(self) -> None:
        self.primary_data = None
        self.ref_data = None


class SyntheticSource(AudioSource):
    """Generates continuous calibrated speech and noise on-the-fly."""

    def __init__(self, clip_meta: Dict[str, Any], sr: int = 16000, seed: int = 42):
        self.clip_meta = clip_meta
        self.sr = sr
        self.seed = seed
        self.clean = None
        self.primary = None
        self.reference = None
        self.cursor = 0
        self.total_samples = 0

    def open(self) -> None:
        self.clean, self.primary, self.reference = get_benchmark_pair(self.clip_meta, seed=self.seed)
        self.total_samples = len(self.primary)
        self.cursor = 0

    def read_chunk(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_active():
            return np.zeros(n_samples, dtype=np.float32), np.zeros(n_samples, dtype=np.float32)

        start = self.cursor
        end = min(start + n_samples, self.total_samples)
        chunk_p = self.primary[start:end]
        chunk_r = self.reference[start:end]

        if len(chunk_p) < n_samples:
            pad = n_samples - len(chunk_p)
            chunk_p = np.pad(chunk_p, (0, pad))
            chunk_r = np.pad(chunk_r, (0, pad))

        self.cursor += n_samples
        return chunk_p.astype(np.float32), chunk_r.astype(np.float32)

    def is_active(self) -> bool:
        return self.cursor < self.total_samples

    def close(self) -> None:
        self.clean = None
        self.primary = None
        self.reference = None


class LoopbackSource(AudioSource):
    """Generates chunks from a callable generator function."""

    def __init__(self, generator_fn: Callable[[int], Tuple[np.ndarray, np.ndarray]], max_chunks: int = 100):
        self.generator_fn = generator_fn
        self.max_chunks = max_chunks
        self.chunk_count = 0

    def open(self) -> None:
        self.chunk_count = 0

    def read_chunk(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        self.chunk_count += 1
        return self.generator_fn(n_samples)

    def is_active(self) -> bool:
        return self.chunk_count < self.max_chunks

    def close(self) -> None:
        pass


class ALSASource(AudioSource):
    """
    Hardware ALSA I2S capture source for Raspberry Pi 4 + WM8960 Audio HAT.
    Acts as a verified stub in pre-hardware simulation (PH0).
    """

    def __init__(self, device: str = "hw:1,0", sample_rate: int = 16000):
        self.device = device
        self.sr = sample_rate
        self._is_open = False

    def open(self) -> None:
        try:
            import alsaaudio  # pyalsaaudio on Linux Pi
            self.stream = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=self.device)
            self.stream.setchannels(2)  # Stereo: Ch0 Primary mic, Ch1 Reference mic
            self.stream.setrate(self.sr)
            self.stream.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            self.stream.setperiodsize(128)
            self._is_open = True
        except (ImportError, Exception) as e:
            raise HardwareUnavailableError(
                f"ALSA hardware audio source '{self.device}' is unavailable on this host platform. "
                f"Requires physical Linux Raspberry Pi 4 with WM8960 I2S driver: {e}"
            )

    def read_chunk(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._is_open:
            raise HardwareUnavailableError("ALSA capture stream is not open.")
        length, raw_data = self.stream.read()
        interleaved = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        primary = interleaved[0::2]
        reference = interleaved[1::2]
        return primary[:n_samples], reference[:n_samples]

    def is_active(self) -> bool:
        return self._is_open

    def close(self) -> None:
        if self._is_open:
            self.stream.close()
            self._is_open = False
