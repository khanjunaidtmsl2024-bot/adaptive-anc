"""
Unit Tests for Gate 7: Audio IO Abstraction Layer.
PS 26052 — Adaptive Defence ANC.
"""

import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

from src.audio_io import (
    AudioSource,
    AudioSink,
    HardwareUnavailableError,
    WAVSource,
    SyntheticSource,
    LoopbackSource,
    ALSASource,
    WAVSink,
    NullSink,
    ALSASink,
)
from src.streaming.causal_engine import CausalStreamingEngine
from src.dataset.synthetic_benchmark_matrix import generate_benchmark_matrix


class TestAudioIOAbstraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("scratch/test_audio_io")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.sr = 16000
        self.chunk_size = 128

        # Create dummy WAV files
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        self.primary_wav = self.test_dir / "primary.wav"
        self.ref_wav = self.test_dir / "ref.wav"
        self.out_wav = self.test_dir / "out.wav"

        sf.write(str(self.primary_wav), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), self.sr)
        sf.write(str(self.ref_wav), (0.1 * np.random.randn(8000)).astype(np.float32), self.sr)

    def test_wav_source_and_sink(self):
        """Verify reading from WAVSource and writing to WAVSink."""
        source = WAVSource(str(self.primary_wav), str(self.ref_wav), sr=self.sr)
        sink = WAVSink(str(self.out_wav), sr=self.sr)

        with source, sink:
            while source.is_active():
                p, r = source.read_chunk(self.chunk_size)
                self.assertEqual(len(p), self.chunk_size)
                self.assertEqual(len(r), self.chunk_size)
                # Write back dummy
                sink.write_chunk(p)

        self.assertTrue(self.out_wav.exists())
        data, sr = sf.read(str(self.out_wav))
        self.assertEqual(sr, self.sr)
        self.assertGreater(len(data), 0)

    def test_synthetic_source_and_null_sink(self):
        """Verify SyntheticSource reading with NullSink."""
        meta = generate_benchmark_matrix()[0]
        source = SyntheticSource(meta, sr=self.sr)
        sink = NullSink()

        with source, sink:
            count = 0
            while source.is_active() and count < 20:
                p, r = source.read_chunk(self.chunk_size)
                self.assertEqual(len(p), self.chunk_size)
                sink.write_chunk(p)
                count += 1

        self.assertEqual(sink.total_chunks, 20)
        self.assertEqual(sink.total_samples, 20 * self.chunk_size)

    def test_loopback_source(self):
        """Verify LoopbackSource with generator."""
        def dummy_gen(n):
            return np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

        source = LoopbackSource(dummy_gen, max_chunks=10)
        chunks_read = 0
        with source:
            while source.is_active():
                p, r = source.read_chunk(self.chunk_size)
                self.assertEqual(p[0], 1.0)
                chunks_read += 1
        self.assertEqual(chunks_read, 10)

    def test_alsa_stubs_raise_gracefully(self):
        """ALSASource and ALSASink must raise HardwareUnavailableError on non-target platform."""
        src = ALSASource(device="hw:1,0")
        with self.assertRaises(HardwareUnavailableError):
            src.open()

        sink = ALSASink(device="hw:1,0")
        with self.assertRaises(HardwareUnavailableError):
            sink.open()


if __name__ == "__main__":
    unittest.main()
