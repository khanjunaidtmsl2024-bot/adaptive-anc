"""
Unit tests for the Laptop Validation Suite (Tests 1 to 8) and ANC X-Ray Exporter.
Smart India Hackathon (SIH) 2026 - Problem Statement ID: 26052
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.evaluation.laptop_test_suite import (
    run_test_1_known_noise,
    run_test_2_quantitative_table,
    run_test_3_ai_separately,
    run_test_4_reference_leakage,
    run_test_5_impulsive_noise,
    run_test_6_nonstationary_tracking,
    run_test_7_causality,
    run_test_8_streaming_simulation,
)
from src.visualization.anc_xray import export_experiment_bundle


class TestLaptopValidationSuite:
    """Verifies all 8 canonical laptop tests execute deterministically."""

    def test_01_known_noise_cancellation(self):
        res = run_test_1_known_noise(sr=16000)
        assert res["passed"] is True, f"Known noise test failed: {res}"
        assert res["delta_snr"] > 3.0, f"Expected Delta SNR > 3 dB, got {res['delta_snr']}"

    def test_02_quantitative_table(self):
        res = run_test_2_quantitative_table(sr=16000)
        assert res["passed"] is True, f"Quantitative table failed: {res}"
        table = res["table"]
        assert "Noisy" in table and "Hybrid" in table
        assert table["Hybrid"]["stoi"] >= 0.70

    def test_03_ai_separately(self):
        res = run_test_3_ai_separately(sr=16000)
        assert res["passed"] is True, f"AI contribution test failed: {res}"

    def test_04_reference_leakage(self):
        res = run_test_4_reference_leakage(sr=16000)
        assert res["passed"] is True, f"Leakage test failed: {res}"
        assert len(res["curve"]) == 6

    def test_05_impulsive_noise_recovery(self):
        res = run_test_5_impulsive_noise(sr=16000)
        assert res["passed"] is True, f"Impulse recovery failed: {res}"
        assert res["recovery_time_ms"] < 150.0

    def test_06_nonstationary_tracking(self):
        res = run_test_6_nonstationary_tracking(sr=16000)
        assert res["passed"] is True, f"Nonstationary test failed: {res}"
        assert res["divergence_detected"] is False

    def test_07_causality(self):
        res = run_test_7_causality(sr=16000)
        assert res["passed"] is True, f"Causality test failed: {res}"
        assert res["max_future_leakage"] < 1e-5

    def test_08_streaming_simulation(self):
        res = run_test_8_streaming_simulation(sr=16000)
        assert res["test_8a_passed"] is True, f"Streaming mechanism (8a) failed: {res['verdict_8a']}"
        assert res["test_8b_passed"] is True, f"Computational budget (8b) failed: {res['verdict_8b']}"
        assert res["p95_ms"] <= 8.000, f"Expected P95 <= 8.000 ms, got {res['p95_ms']}"

    def test_09_experiment_bundle_export(self):
        tmp_dir = tempfile.mkdtemp(prefix="test_xray_")
        try:
            sr = 16000
            n = sr * 1  # 1 second
            clean = np.sin(2 * np.pi * 300 * np.arange(n) / sr).astype(np.float32)
            noise = np.random.normal(0, 0.2, n).astype(np.float32)
            primary = clean + noise
            reference = np.roll(noise, 2)
            residual = clean + 0.5 * noise
            enhanced = clean + 0.1 * noise
            noise_est = 0.5 * noise

            metrics = {
                "Noisy": {"snr_db": 0.0, "si_sdr_db": 0.0, "stoi": 0.65, "pesq": 1.4},
                "Hybrid": {"snr_db": 5.0, "si_sdr_db": 8.0, "stoi": 0.85, "pesq": 1.7},
            }
            latencies = [4.2, 4.5, 4.3, 5.1, 4.4]

            manifest = export_experiment_bundle(
                exp_dir=tmp_dir,
                config={"exp_id": "TEST_UNIT", "sample_rate": sr},
                clean=clean,
                primary=primary,
                reference=reference,
                noise_estimate=noise_est,
                residual=residual,
                enhanced=enhanced,
                metrics_dict=metrics,
                frame_times_ms=latencies,
                sr=sr,
            )

            # Check that all files exist
            expected_keys = [
                "config", "input.wav", "reference.wav", "nlms_output.wav",
                "hybrid_output.wav", "metrics", "latency", "waveform.png",
                "spectrogram_input.png", "spectrogram_output.png",
                "metrics_comparison.png", "latency_profile.png",
                "anc_xray_full_panel.png"
            ]
            for key in expected_keys:
                assert key in manifest, f"Missing {key} in manifest"
                assert Path(manifest[key]).exists(), f"File {manifest[key]} does not exist"

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
