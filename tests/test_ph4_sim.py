"""
Unit Tests for PH4-SIM Physical ANC Simulation and Plant Models.
PS 26052 — Adaptive Defence ANC.
"""

import unittest
import numpy as np

from src.dsp.acoustic_plant import AcousticPlantModel, StreamingPlantState
from src.dsp.fxlms import FxLMSFilter, SecondaryPathModel, run_fxnlms_simulation
from src.dsp.supervisory_coupling import SupervisoryGovernor, run_supervisory_hybrid_simulation


class TestPH4Simulation(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.plant = AcousticPlantModel(sample_rate=self.sr, fir_length=64)

    def test_plant_frequency_and_uncertainty_properties(self):
        """Plant must synthesize stable impulse responses and exact uncertainty percentages."""
        # 1. Finite, non-empty responses
        self.assertEqual(len(self.plant.p_nominal), 64)
        self.assertEqual(len(self.plant.s_nominal), 64)
        self.assertFalse(np.isnan(self.plant.s_nominal).any())
        self.assertFalse(np.isnan(self.plant.p_nominal).any())

        # 2. Uncertainty Generator exact L2 relative error
        for err_target in [5.0, 10.0, 20.0, 50.0]:
            s_hat = self.plant.generate_uncertain_secondary_path(mismatch_percent=err_target, seed=123)
            rel_err = float(np.linalg.norm(s_hat - self.plant.s_nominal) / np.linalg.norm(self.plant.s_nominal)) * 100.0
            self.assertAlmostEqual(rel_err, err_target, delta=0.5)

    def test_fxnlms_convergence_on_tonal_disturbance(self):
        """FxNLMS must converge and achieve substantial cancellation (> 15 dB) on low-frequency tones."""
        # Generate 250 Hz tone (within primary ANC target band)
        duration = 1.0
        n_samples = int(duration * self.sr)
        t = np.arange(n_samples) / self.sr
        ref = np.sin(2 * np.pi * 250.0 * t).astype(np.float32)

        # Primary disturbance
        prim = self.plant.convolve_primary(ref)
        s_true = self.plant.s_nominal
        s_hat = self.plant.s_nominal.copy()

        res = run_fxnlms_simulation(
            reference=ref,
            primary_disturbance=prim,
            s_true=s_true,
            s_hat=s_hat,
            filter_length=64,
            step_size=0.02,
            leakage=1e-4,
        )

        self.assertFalse(res["diverged"], "FxNLMS diverged on pure tone")
        self.assertGreater(res["cancellation_db"], 15.0, f"Insufficient cancellation: {res['cancellation_db']} dB")

    def test_fxnlms_under_modeling_uncertainty(self):
        """FxNLMS must remain stable under moderate (10-20%) secondary path modeling mismatch."""
        from scipy import signal
        duration = 1.0
        n_samples = int(duration * self.sr)
        rng = np.random.RandomState(42)
        # Band-limited disturbance in primary ANC target band (< 500 Hz)
        b, a = signal.butter(4, 500.0 / (self.sr / 2.0), btype="low")
        ref = signal.lfilter(b, a, rng.randn(n_samples).astype(np.float32)).astype(np.float32) * 0.1
        prim = self.plant.convolve_primary(ref)
        s_true = self.plant.s_nominal

        # 10% modeling error
        s_hat_10 = self.plant.generate_uncertain_secondary_path(mismatch_percent=10.0, seed=101)
        res_10 = run_fxnlms_simulation(ref, prim, s_true, s_hat_10, filter_length=64, step_size=0.01)
        self.assertFalse(res_10["diverged"])
        self.assertGreater(res_10["cancellation_db"], 3.0)

        # 20% modeling error
        s_hat_20 = self.plant.generate_uncertain_secondary_path(mismatch_percent=20.0, seed=202)
        res_20 = run_fxnlms_simulation(ref, prim, s_true, s_hat_20, filter_length=64, step_size=0.01)
        self.assertFalse(res_20["diverged"])

    def test_supervisory_governor_freezes_during_speech(self):
        """Supervisory governor must detect speech and freeze adaptation to protect voice."""
        governor = SupervisoryGovernor(sample_rate=self.sr, hop_size=128)

        # Frame 1: Quiet noise
        quiet_frame = np.zeros(128, dtype=np.float32) + 0.001
        diag_quiet = governor.analyze_frame(quiet_frame, quiet_frame, current_step_size=0.02)
        self.assertFalse(diag_quiet["freeze_adaptation"])
        self.assertEqual(diag_quiet["adjusted_mu"], 0.02)

        # Frame 2: User Speech Burst
        speech_frame = (np.sin(2 * np.pi * 300.0 * np.arange(128) / self.sr) * 0.5).astype(np.float32)
        diag_speech = governor.analyze_frame(speech_frame, quiet_frame, current_step_size=0.02)
        self.assertTrue(diag_speech["freeze_adaptation"])
        self.assertEqual(diag_speech["adjusted_mu"], 0.0)


if __name__ == "__main__":
    unittest.main()
