"""
PS 26052: DRDO Adaptive ANC — Physical Audio Full-Duplex Engine (PH5.1)
========================================================================
Synchronous full-duplex audio stream wrapper using PyAudio.
Facilitates simultaneous DAC playback (secondary speaker) and ADC capture
(error / reference microphone) with sample-aligned buffer management.
"""

import os
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PYAUDIO_AVAILABLE = False


class DuplexAudioEngine:
    """
    Synchronous full-duplex audio I/O manager.
    Coordinates playback of excitation signals while simultaneously capturing
    microphone responses with minimal operating system buffer jitter.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 256,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
    ):
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio is not installed. Run 'pip install pyaudio'.")

        self.sample_rate = int(sample_rate)
        self.chunk_size = int(chunk_size)
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.pa = pyaudio.PyAudio()

        # Resolve device indices if not specified
        if self.input_device_index is None:
            try:
                def_in = self.pa.get_default_input_device_info()
                self.input_device_index = def_in["index"]
            except IOError:
                self.input_device_index = None

        if self.output_device_index is None:
            try:
                def_out = self.pa.get_default_output_device_info()
                self.output_device_index = def_out["index"]
            except IOError:
                self.output_device_index = None

    def close(self):
        """Cleanly terminate PyAudio instance."""
        if self.pa is not None:
            self.pa.terminate()
            self.pa = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def list_devices(cls) -> List[Dict[str, Union[int, str, float]]]:
        """Enumerate all available audio input and output devices."""
        if not PYAUDIO_AVAILABLE:
            return []
        pa = pyaudio.PyAudio()
        devices = []
        try:
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                    devices.append({
                        "index": i,
                        "name": info.get("name", "Unknown"),
                        "host_api": info.get("hostApi", 0),
                        "max_input_channels": info.get("maxInputChannels", 0),
                        "max_output_channels": info.get("maxOutputChannels", 0),
                        "default_sample_rate": info.get("defaultSampleRate", 0),
                    })
                except Exception:
                    continue
        finally:
            pa.terminate()
        return devices

    def play_and_record(
        self,
        playback_signal: np.ndarray,
        record_extra_seconds: float = 0.2,
        input_channels: int = 1,
        output_channels: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Simultaneously play an output buffer and record the physical microphone response.

        Parameters
        ----------
        playback_signal : np.ndarray
            1D float32 or float64 array normalized in [-1.0, 1.0].
        record_extra_seconds : float
            Post-playback capture window to accommodate physical acoustic decay
            and pipeline buffer delays.
        input_channels : int
            Number of recording channels (default 1).
        output_channels : int
            Number of playback channels (default 1).

        Returns
        -------
        recorded_signal : np.ndarray
            1D float32 array of captured samples.
        telemetry : dict
            Diagnostic metrics including timing, buffer overrun/underrun count.
        """
        sig_float = np.asarray(playback_signal, dtype=np.float32).ravel()
        # Soft-limit to prevent DAC digital clipping
        sig_float = np.clip(sig_float, -1.0, 1.0)

        total_playback_samples = len(sig_float)
        extra_samples = int(record_extra_seconds * self.sample_rate)
        total_record_samples = total_playback_samples + extra_samples

        # Zero-pad playback signal to match total record duration
        padded_playback = np.zeros(total_record_samples, dtype=np.float32)
        padded_playback[:total_playback_samples] = sig_float

        # Pre-convert to 16-bit PCM for universal driver compatibility
        pcm_playback = (padded_playback * 32767.0).astype(np.int16)
        if output_channels == 2:
            pcm_playback = np.repeat(pcm_playback[:, None], 2, axis=1)

        raw_play_bytes = pcm_playback.tobytes()

        # Open full-duplex stream
        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=input_channels,
            rate=self.sample_rate,
            input=True,
            output=True,
            input_device_index=self.input_device_index,
            output_device_index=self.output_device_index,
            frames_per_buffer=self.chunk_size,
        )

        recorded_chunks = []
        bytes_per_play_frame = self.chunk_size * output_channels * 2  # 16-bit = 2 bytes
        play_ptr = 0
        total_bytes = len(raw_play_bytes)

        t_start = time.perf_counter()
        overflows = 0

        try:
            while play_ptr < total_bytes:
                chunk_bytes = raw_play_bytes[play_ptr : play_ptr + bytes_per_play_frame]
                if len(chunk_bytes) < bytes_per_play_frame:
                    chunk_bytes = chunk_bytes + b"\x00" * (bytes_per_play_frame - len(chunk_bytes))

                # Write to DAC
                stream.write(chunk_bytes)

                # Read from ADC
                try:
                    in_data = stream.read(self.chunk_size, exception_on_overflow=False)
                    recorded_chunks.append(in_data)
                except Exception:
                    overflows += 1

                play_ptr += bytes_per_play_frame

        finally:
            stream.stop_stream()
            stream.close()

        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        # Assemble captured buffer
        raw_rec_bytes = b"".join(recorded_chunks)
        rec_int16 = np.frombuffer(raw_rec_bytes, dtype=np.int16)
        if input_channels > 1:
            rec_int16 = rec_int16.reshape(-1, input_channels)[:, 0]

        recorded_signal = rec_int16.astype(np.float32) / 32767.0

        # Trim to total expected samples
        if len(recorded_signal) > total_record_samples:
            recorded_signal = recorded_signal[:total_record_samples]

        telemetry = {
            "sample_rate": float(self.sample_rate),
            "chunk_size": float(self.chunk_size),
            "playback_samples": float(total_playback_samples),
            "recorded_samples": float(len(recorded_signal)),
            "elapsed_ms": float(t_total_ms),
            "overflow_count": float(overflows),
            "input_peak_amplitude": float(np.max(np.abs(recorded_signal))) if len(recorded_signal) > 0 else 0.0,
            "input_rms": float(np.sqrt(np.mean(recorded_signal ** 2))) if len(recorded_signal) > 0 else 0.0,
        }

        return recorded_signal, telemetry

    @staticmethod
    def save_wav_pair(
        output_path_stimulus: Union[str, Path],
        output_path_response: Union[str, Path],
        stimulus: np.ndarray,
        response: np.ndarray,
        sample_rate: int = 16000,
    ):
        """Save stimulus and captured response as synchronized WAV files."""
        for p, sig in [(output_path_stimulus, stimulus), (output_path_response, response)]:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            norm_sig = np.clip(sig, -1.0, 1.0)
            int16_sig = (norm_sig * 32767.0).astype(np.int16)
            with wave.open(str(p), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int16_sig.tobytes())
