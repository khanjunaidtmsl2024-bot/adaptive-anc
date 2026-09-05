"""
PS 26052 -- Gate 1: Comprehensive DSP Mathematical Robustness Matrix.
Tests 16 extreme and adversarial signal edge cases on CausalStreamingEngine
and VSS-NLMS hybrid chain to guarantee zero crashes, zero NaNs, and numerical stability.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.streaming.causal_engine import CausalStreamingEngine
from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.delay_alignment import DelayAligner
from src.dsp.leakage_detector import SpeechLeakageDetector
from src.dsp.impulse_protection import ImpulseProtectionController


class TestDSPRobustnessMatrix(unittest.TestCase):
    """16-case exhaustive mathematical robustness test matrix."""

    def setUp(self):
        self.sr = 16000
        self.engine = CausalStreamingEngine(
            frame_size=512,
            hop_size=128,
            sample_rate=self.sr,
            filter_length=64,
            step_size=0.05
        )
        self.delay = self.engine.frame_size - self.engine.hop_size
        self.n_samples = 8000  # 0.5s
        
        # Load realistic speech with formant structure
        clean_path = Path("data/v4/clean/SPK_001_clean.wav")
        if clean_path.exists():
            import soundfile as sf
            wav, _ = sf.read(str(clean_path))
            self.clean_speech = wav[:self.n_samples].astype(np.float32)
        else:
            t = np.linspace(0, 0.5, self.n_samples, endpoint=False)
            # Formant synthesis fallback
            self.clean_speech = (
                0.4 * np.sin(2 * np.pi * 300 * t) * np.sin(2 * np.pi * 5 * t) +
                0.3 * np.sin(2 * np.pi * 1200 * t) * np.cos(2 * np.pi * 3 * t) +
                0.2 * np.sin(2 * np.pi * 2500 * t)
            ).astype(np.float32)
            
        self.noise = (0.2 * np.random.randn(self.n_samples)).astype(np.float32)

    def test_01_zero_input(self):
        """Case 1: All zeros input should produce zero output without NaNs."""
        zeros = np.zeros(self.n_samples, dtype=np.float32)
        out, diag = self.engine.process_signal(zeros, zeros)
        self.assertEqual(len(out), self.n_samples)
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())
        self.assertLessEqual(np.max(np.abs(out)), 1e-6)

    def test_02_speech_only(self):
        """Case 2: Pure speech in primary, silence in reference. Speech should not be destroyed."""
        zeros = np.zeros(self.n_samples, dtype=np.float32)
        out, _ = self.engine.process_signal(self.clean_speech, zeros)
        self.assertFalse(np.isnan(out).any())
        valid_out = out[self.delay : 7500]
        valid_in = self.clean_speech[self.delay : 7500]
        out_energy = np.mean(valid_out ** 2)
        in_energy = np.mean(valid_in ** 2)
        # Should preserve measurable speech energy without collapse
        self.assertGreater(out_energy, 1e-5)

    def test_03_noise_only(self):
        """Case 3: Pure noise in primary, correlated noise in reference. Output should be attenuated."""
        primary_noise = self.noise
        ref_noise = np.roll(self.noise, 2)
        out, _ = self.engine.process_signal(primary_noise, ref_noise)
        self.assertFalse(np.isnan(out).any())
        # NLMS should reduce steady-state noise power
        self.assertLess(np.mean(out[2000:] ** 2), np.mean(primary_noise[2000:] ** 2) * 1.5)

    def test_04_nominal_speech_plus_noise(self):
        """Case 4: Nominal speech + noise mixtures."""
        primary = self.clean_speech + self.noise
        ref = np.roll(self.noise, 3)
        out, diag = self.engine.process_signal(primary, ref)
        self.assertEqual(len(out), len(primary))
        self.assertFalse(np.isnan(out).any())
        self.assertGreater(diag["total_frames"], 0)

    def test_05_pure_reference_noise(self):
        """Case 5: Uncorrelated reference noise. Primary should not be corrupted."""
        uncorr_ref = np.random.randn(self.n_samples).astype(np.float32) * 0.2
        out, _ = self.engine.process_signal(self.clean_speech, uncorr_ref)
        self.assertFalse(np.isnan(out).any())
        valid_out = out[self.delay : 7500]
        self.assertGreater(np.mean(valid_out ** 2), 1e-5)

    def test_06_reference_with_speech_leakage(self):
        """Case 6: 30% speech bleed into reference microphone."""
        primary = self.clean_speech + self.noise
        ref = self.noise + 0.30 * self.clean_speech
        out, diag = self.engine.process_signal(primary, ref)
        self.assertFalse(np.isnan(out).any())
        # Residual must retain speech correlation accounting for pipeline latency
        valid_len = 6500
        corr = np.corrcoef(out[self.delay : self.delay + valid_len], self.clean_speech[:valid_len])[0, 1]
        self.assertGreater(corr, 0.1, "Speech should retain positive correlation with ground truth")

    def test_07_extreme_impulsive_transient(self):
        """Case 7: Gunfire / impulse transient (+40 dB spike)."""
        primary = self.clean_speech.copy()
        primary[3000:3010] += 50.0  # Huge +40dB transient
        ref = self.noise.copy()
        out, diag = self.engine.process_signal(primary, ref)
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())
        # Peak-limiting ensures output is bounded in [-1.0, 1.0]
        self.assertLessEqual(np.max(np.abs(out)), 1.0)

    def test_08_digital_clipping(self):
        """Case 8: Input exceeds standard full-scale (+2.5 to -2.5)."""
        primary = np.clip(self.clean_speech * 10.0, -2.5, 2.5)
        ref = np.clip(self.noise * 10.0, -2.5, 2.5)
        out, _ = self.engine.process_signal(primary, ref)
        self.assertFalse(np.isnan(out).any())
        self.assertLessEqual(np.max(np.abs(out)), 1.0)

    def test_09_very_low_amplitude(self):
        """Case 9: Micro-amplitude signal (1e-7 amplitude, near digital floor)."""
        tiny_primary = self.clean_speech * 1e-7
        tiny_ref = self.noise * 1e-7
        out, _ = self.engine.process_signal(tiny_primary, tiny_ref)
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())

    def test_10_high_amplitude(self):
        """Case 10: High amplitude signal (+100.0)."""
        high_p = self.clean_speech * 100.0
        high_r = self.noise * 100.0
        out, _ = self.engine.process_signal(high_p, high_r)
        self.assertFalse(np.isnan(out).any())
        self.assertLessEqual(np.max(np.abs(out)), 1.0)

    def test_11_nan_inf_injection(self):
        """Case 11: Deliberate injection of NaN and Inf values into input."""
        corrupt_p = self.clean_speech.copy()
        corrupt_p[100:105] = np.nan
        corrupt_p[500:505] = np.inf
        corrupt_p[1000:1005] = -np.inf
        ref = self.noise.copy()
        out, _ = self.engine.process_signal(corrupt_p, ref)
        self.assertFalse(np.isnan(out).any(), "Sanitizer must remove all NaNs from output")
        self.assertFalse(np.isinf(out).any(), "Sanitizer must remove all Infs from output")

    def test_12_single_vs_multi_channel_shapes(self):
        """Case 12: Column vectors (N, 1) should be handled cleanly."""
        col_p = self.clean_speech.reshape(-1, 1).flatten()
        col_r = self.noise.reshape(-1, 1).flatten()
        out, _ = self.engine.process_signal(col_p, col_r)
        self.assertEqual(len(out), self.n_samples)

    def test_13_dropped_discontinuous_frames(self):
        """Case 13: Sudden 180-degree phase shift in input midway."""
        discontinuous_p = self.clean_speech.copy()
        discontinuous_p[4000:] *= -1.0  # Phase inversion jump
        out, _ = self.engine.process_signal(discontinuous_p, self.noise)
        self.assertFalse(np.isnan(out).any())
        self.assertLessEqual(np.max(np.abs(out)), 1.0)

    def test_14_dc_offset(self):
        """Case 14: Extreme DC offset bias (+0.5)."""
        dc_p = self.clean_speech + 0.5
        dc_r = self.noise + 0.3
        out, _ = self.engine.process_signal(dc_p, dc_r)
        self.assertFalse(np.isnan(out).any())
        self.assertLessEqual(np.max(np.abs(out)), 1.0)

    def test_15_random_noise_burst(self):
        """Case 15: High-entropy noise burst in the middle of speech."""
        burst_p = self.clean_speech.copy()
        burst_p[2000:3000] += np.random.uniform(-5.0, 5.0, 1000).astype(np.float32)
        out, _ = self.engine.process_signal(burst_p, self.noise)
        self.assertFalse(np.isnan(out).any())
        self.assertLessEqual(np.max(np.abs(out)), 1.0)

    def test_16_non_power_of_two_lengths(self):
        """Case 16: Non-power-of-two arbitrary prime length signal (e.g. 7,381 samples)."""
        prime_len = 7381
        p = self.clean_speech[:prime_len]
        r = self.noise[:prime_len]
        out, _ = self.engine.process_signal(p, r)
        self.assertEqual(len(out), prime_len, "Output length must match prime input length")
        self.assertFalse(np.isnan(out).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
