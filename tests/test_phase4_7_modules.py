"""
Unit Tests for Phase 4-7 Modules.
PS 26052 -- Adaptive Defence ANC.

Tests:
  1. DTLN model — forward pass, parameter count, wrapper interface
  2. CRN model — forward pass, parameter count, wrapper interface
  3. Noise Regime Detector — classification accuracy on synthetic signals
  4. Causal Streaming Engine — hop-by-hop processing, latency tracking
  5. Training harness — single-step gradient flow
"""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestDTLN(unittest.TestCase):
    """Test DTLN model architecture and wrapper."""

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_dtln_forward_shape(self):
        from src.ai.dtln import DTLNNet
        model = DTLNNet(frame_size=512, hop_size=128, hidden_size=64, encoder_size=128)
        x = torch.randn(1, 1, 16000)  # 1 second @ 16kHz
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, x.shape, "DTLN output shape must match input")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_dtln_parameter_count(self):
        from src.ai.dtln import DTLNNet
        # Contract-geometry default (frame 256); legacy frame-512 variants are
        # covered by test_dtln_forward_shape.
        model = DTLNNet(hidden_size=64, encoder_size=128)
        params = model.count_parameters()
        self.assertGreater(params, 10000, "DTLN should have >10K params")
        self.assertLess(params, 5_000_000, "DTLN should have <5M params")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_dtln_wrapper_spectrogram(self):
        from src.ai.dtln import DTLNWrapper
        # Legacy geometry: 512-frame wrapper consumes 257-bin spectrograms.
        wrapper = DTLNWrapper(frame_size=512, hop_size=128, hidden_size=64, encoder_size=128)
        mag = np.random.rand(257, 10).astype(np.float32)
        phase = np.random.rand(257, 10).astype(np.float32)
        enh_mag, enh_phase = wrapper.enhance_spectrogram(mag, phase)
        self.assertEqual(enh_mag.shape[0], 257)
        self.assertEqual(enh_phase.shape[0], 257)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_dtln_wrapper_waveform(self):
        from src.ai.dtln import DTLNWrapper
        wrapper = DTLNWrapper(hidden_size=64, encoder_size=128)
        wav = np.random.randn(16000).astype(np.float32) * 0.1
        enhanced = wrapper.enhance_waveform(wav)
        self.assertEqual(len(enhanced), len(wav))


class TestCRN(unittest.TestCase):
    """Test CRN model architecture and wrapper."""

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_crn_forward_shape(self):
        from src.ai.crn import CRNNet
        model = CRNNet(freq_bins=257, hidden_size=64, channels=(8, 16, 32, 64, 128))
        x = torch.randn(1, 1, 257, 50)
        with torch.no_grad():
            mask = model(x)
        self.assertEqual(mask.shape, x.shape, "CRN mask shape must match input")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_crn_mask_range(self):
        from src.ai.crn import CRNNet
        model = CRNNet(freq_bins=257, hidden_size=64, channels=(8, 16, 32, 64, 128))
        x = torch.randn(1, 1, 257, 50)
        with torch.no_grad():
            mask = model(x)
        self.assertTrue(torch.all(mask >= 0), "CRN mask must be >= 0")
        self.assertTrue(torch.all(mask <= 1), "CRN mask must be <= 1")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_crn_wrapper_spectrogram(self):
        from src.ai.crn import CRNWrapper
        # Legacy geometry: 257-bin wrapper consumes 257-bin spectrograms.
        wrapper = CRNWrapper(freq_bins=257, hidden_size=64, channels=(8, 16, 32, 64, 128))
        mag = np.random.rand(257, 20).astype(np.float32)
        phase = np.random.rand(257, 20).astype(np.float32)
        enh_mag, enh_phase = wrapper.enhance_spectrogram(mag, phase)
        self.assertEqual(enh_mag.shape, mag.shape)
        self.assertEqual(enh_phase.shape, phase.shape)


class TestNoiseRegimeDetector(unittest.TestCase):
    """Test noise regime classification on synthetic signals."""

    def test_stationary_noise(self):
        from src.dsp.noise_regime_detector import NoiseRegimeDetector, NoiseRegime
        detector = NoiseRegimeDetector(sample_rate=16000, frame_size=512)
        # White noise at moderate level
        signal = np.random.randn(16000).astype(np.float32) * 0.1
        regime, diag = detector.classify_signal(signal)
        # White noise should be DIFFUSE or STATIONARY (spectral flatness close to 1 for white noise)
        self.assertIn(regime, [NoiseRegime.DIFFUSE, NoiseRegime.STATIONARY])

    def test_impulsive_detection(self):
        from src.dsp.noise_regime_detector import NoiseRegimeDetector, NoiseRegime
        detector = NoiseRegimeDetector(sample_rate=16000, frame_size=512)
        # Mostly silence with large spikes
        signal = np.zeros(16000, dtype=np.float32)
        spike_positions = [1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000]
        for pos in spike_positions:
            signal[pos:pos + 8] = 10.0  # Very sharp spikes
        regime, diag = detector.classify_signal(signal)
        # Should detect as impulsive due to high kurtosis
        self.assertEqual(regime, NoiseRegime.IMPULSIVE,
                         f"Expected IMPULSIVE but got {regime.value}")

    def test_pipeline_params_exist(self):
        from src.dsp.noise_regime_detector import NoiseRegimeDetector, NoiseRegime, REGIME_PRESETS
        for regime in NoiseRegime:
            params = REGIME_PRESETS[regime]
            self.assertIn("nlms_mu", params)
            self.assertIn("filter_length", params)
            self.assertIn("enable_impulse_protection", params)
            self.assertIn("ai_aggressiveness", params)

    def test_regime_distribution(self):
        from src.dsp.noise_regime_detector import NoiseRegimeDetector
        detector = NoiseRegimeDetector(sample_rate=16000, frame_size=512)
        signal = np.random.randn(16000).astype(np.float32) * 0.1
        _, diag = detector.classify_signal(signal)
        dist = diag["regime_distribution"]
        total = sum(dist.values())
        self.assertAlmostEqual(total, 1.0, places=2,
                               msg="Regime distribution must sum to 1.0")


class TestCausalStreamingEngine(unittest.TestCase):
    """Test causal streaming engine processing."""

    def test_hop_by_hop_output_length(self):
        from src.streaming.causal_engine import CausalStreamingEngine
        engine = CausalStreamingEngine(frame_size=256, hop_size=128, sample_rate=16000)
        primary = np.random.randn(16000).astype(np.float32) * 0.1
        reference = np.random.randn(16000).astype(np.float32) * 0.1
        output, agg = engine.process_signal(primary, reference)
        self.assertEqual(len(output), len(primary),
                         "Causal engine output length must match input")

    def test_latency_budget(self):
        from src.streaming.causal_engine import CausalStreamingEngine
        engine = CausalStreamingEngine(frame_size=256, hop_size=128, sample_rate=16000)
        primary = np.random.randn(8000).astype(np.float32) * 0.1
        reference = np.random.randn(8000).astype(np.float32) * 0.1
        _, _ = engine.process_signal(primary, reference)
        budget = engine.get_latency_budget()
        self.assertEqual(budget["algorithmic_latency_ms"], 16.0,
                         "256 samples @ 16kHz = 16ms algorithmic latency")
        self.assertIn("total_e2e_latency_ms", budget)

    def test_regime_adaptation(self):
        from src.streaming.causal_engine import CausalStreamingEngine
        engine = CausalStreamingEngine(
            frame_size=256, hop_size=128,
            enable_regime_adaptation=True,
        )
        primary = np.random.randn(8000).astype(np.float32) * 0.1
        reference = np.random.randn(8000).astype(np.float32) * 0.1
        _, agg = engine.process_signal(primary, reference)
        self.assertIn("regime_distribution", agg)
        self.assertGreater(agg["total_frames"], 0)


class TestSTFTEngineReset(unittest.TestCase):
    """Test that STFT engine reset works."""

    def test_reset_clears_buffers(self):
        from src.streaming.stft_engine import StreamingSTFTEngine
        engine = StreamingSTFTEngine(frame_size=512, hop_size=256)
        # Process some data
        hop = np.random.randn(256).astype(np.float32)
        engine.process_hop(hop)
        # Reset
        engine.reset()
        self.assertTrue(np.all(engine.input_buf == 0),
                        "Reset must zero the input buffer")
        self.assertTrue(np.all(engine.overlap_buf == 0),
                        "Reset must zero the overlap buffer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
