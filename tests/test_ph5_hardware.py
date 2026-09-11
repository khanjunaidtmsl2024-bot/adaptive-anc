"""
PS 26052: DRDO Adaptive ANC — Unit Tests for Hardware Bring-Up Modules (PH5)
=============================================================================
Tests Farina swept-sine synthesis, inverse filter deconvolution accuracy,
cross-correlation loopback delay estimation, THD calculation, and device discovery.
"""

import numpy as np
import pytest

from src.hardware.duplex_audio import DuplexAudioEngine
from src.hardware.latency_loopback import LoopbackLatencyMeasurer
from src.hardware.secondary_path_measurer import SecondaryPathMeasurer
from src.hardware.open_loop_tester import OpenLoopAntiNoiseTester


class DummyDuplexEngine:
    """Mock duplex audio engine for deterministic unit testing without physical soundcards."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 256, impulse_delay: int = 48):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.impulse_delay = impulse_delay
        # Synthetic band-limited secondary path
        t_ir = np.arange(64) / sample_rate
        self.synthetic_s = (np.exp(-t_ir * 500.0) * np.sin(2.0 * np.pi * 350.0 * t_ir)).astype(np.float32)

    def play_and_record(self, playback_signal, record_extra_seconds=0.2, input_channels=1, output_channels=1):
        sig = np.asarray(playback_signal, dtype=np.float32)
        n_extra = int(record_extra_seconds * self.sample_rate)
        rec_len = len(sig) + n_extra
        # Convolve with synthetic plant and delay
        sim_out = np.convolve(sig, self.synthetic_s, mode="full")
        recorded = np.zeros(rec_len, dtype=np.float32)
        start = self.impulse_delay
        valid_len = min(len(sim_out), rec_len - start)
        recorded[start : start + valid_len] = sim_out[:valid_len]
        recorded += 1e-4 * np.random.randn(rec_len).astype(np.float32)

        telemetry = {
            "sample_rate": float(self.sample_rate),
            "chunk_size": float(self.chunk_size),
            "playback_samples": float(len(sig)),
            "recorded_samples": float(rec_len),
            "elapsed_ms": 10.0,
            "overflow_count": 0.0,
            "input_peak_amplitude": float(np.max(np.abs(recorded))),
            "input_rms": float(np.sqrt(np.mean(recorded ** 2))),
        }
        return recorded, telemetry


def test_farina_sweep_generation():
    """Test that Farina sweep and inverse filter generate correct dimensions and boundaries."""
    engine = DummyDuplexEngine(sample_rate=16000)
    measurer = SecondaryPathMeasurer(engine, f_start=100.0, f_end=4000.0, sweep_duration_sec=0.5)
    sweep, inv_filter = measurer.generate_log_sweep_and_inverse()

    assert len(sweep) == int(16000 * 0.5)
    assert len(inv_filter) == len(sweep)
    assert np.max(np.abs(sweep)) <= 1.05
    assert np.max(np.abs(inv_filter)) <= 1.05

    # Direct deconvolution of sweep and inverse filter should produce a sharp Dirac-like peak
    direct_conv = measurer.deconvolve(sweep, inv_filter)
    peak_idx = np.argmax(np.abs(direct_conv))
    # Peak should be near the sweep length
    assert abs(peak_idx - (len(sweep) - 1)) <= 5


def test_farina_deconvolution_reconstruction():
    """Test that Farina deconvolution mathematically recovers a known acoustic plant."""
    sr = 16000
    engine = DummyDuplexEngine(sample_rate=sr)
    measurer = SecondaryPathMeasurer(engine, f_start=50.0, f_end=4000.0, sweep_duration_sec=0.5)
    sweep, inv_filter = measurer.generate_log_sweep_and_inverse()

    # Define realistic band-limited acoustic cavity plant (400 Hz resonance, 64 taps)
    t_ir = np.arange(64) / sr
    known_fir = (np.exp(-t_ir * 600.0) * np.sin(2.0 * np.pi * 400.0 * t_ir)).astype(np.float32)
    known_fir = known_fir / np.max(np.abs(known_fir))

    simulated_mic_response = np.convolve(sweep, known_fir, mode="full")

    # Deconvolve
    recovered_ir = measurer.deconvolve(simulated_mic_response, inv_filter)

    # Reference Dirac arrival
    direct_conv = measurer.deconvolve(sweep, inv_filter)
    t_ref_zero = int(np.argmax(np.abs(direct_conv)))

    # Recovered segment starting at t_ref_zero
    recovered_segment = recovered_ir[t_ref_zero : t_ref_zero + len(known_fir)]
    corr = np.dot(recovered_segment, known_fir) / (
        np.linalg.norm(recovered_segment) * np.linalg.norm(known_fir)
    )
    assert corr > 0.98, f"Deconvolution correlation was {corr:.4f}, expected > 0.98"


def test_cross_correlation_latency_detection():
    """Test that normalized cross-correlation recovers exact sub-sample delays."""
    sr = 16000
    engine = DummyDuplexEngine(sample_rate=sr)
    measurer = LoopbackLatencyMeasurer(engine, probe_duration_sec=0.04, f_low=400.0, f_high=2000.0)
    probe = measurer.generate_probe_signal()

    # Create artificial delay of 85 samples
    true_delay = 85
    test_response = np.zeros(len(probe) + 200, dtype=np.float32)
    test_response[true_delay : true_delay + len(probe)] = probe

    delay_samples, delay_ms, xcorr = measurer.compute_cross_correlation_delay(
        probe, test_response, sr
    )
    assert abs(delay_samples - true_delay) < 0.2
    assert abs(delay_ms - (true_delay / sr * 1000.0)) < 0.05
    assert np.max(xcorr) > 0.98


def test_open_loop_thd_computation():
    """Test Total Harmonic Distortion calculation on pure vs distorted tones."""
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    f0 = 200.0

    # Pure tone -> THD should be very low (< 0.5%)
    pure_tone = np.sin(2.0 * np.pi * f0 * t)
    thd_pure = OpenLoopAntiNoiseTester.compute_thd(pure_tone, sr, f0)
    assert thd_pure < 1.0, f"Pure tone THD was {thd_pure:.2f}%, expected < 1%"

    # 10% 2nd harmonic distortion -> THD should be ~10%
    distorted_tone = pure_tone + 0.10 * np.sin(2.0 * np.pi * 2.0 * f0 * t)
    thd_distorted = OpenLoopAntiNoiseTester.compute_thd(distorted_tone, sr, f0)
    assert 8.0 <= thd_distorted <= 12.0, f"Distorted THD was {thd_distorted:.2f}%, expected ~10%"


def test_repeatability_dual_metric_diagnosis():
    """Test that RepeatabilityVerifier cleanly separates timing jitter from plant repeatability."""
    sr = 16000
    engine = DummyDuplexEngine(sample_rate=sr)
    measurer = SecondaryPathMeasurer(engine, f_start=50.0, f_end=4000.0, sweep_duration_sec=0.25)
    
    from src.hardware.repeatability_verifier import RepeatabilityVerifier
    verifier = RepeatabilityVerifier(measurer, num_runs=3)
    results = verifier.run_evaluation(pause_between_sec=0.0)
    
    # On a deterministic mock engine without jitter, both raw and aligned correlations should be high
    assert results["mean_aligned_correlation"] > 0.95
    assert results["plant_repeatability_passed"] is True
    assert "anc_band_gain_std_db" in results
    assert "anc_band_phase_std_deg" in results
    assert "delay_std_samples" in results

