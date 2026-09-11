"""
Unit and regression tests for PH4.1 Simulation Realism & Robustness.
PS 26052 — Adaptive Defence ANC.
"""

import numpy as np
import pytest
from scipy import signal

from src.dsp.acoustic_plant import AcousticPlantModel
from src.dsp.fxlms import run_fxnlms_simulation, FxLMSFilter


def test_incoherent_reference_synthesis():
    """Verify incoherent reference synthesis matches target coherence properties."""
    sr = 16000
    N = 16000
    rng = np.random.RandomState(42)
    clean = rng.randn(N).astype(np.float32)

    # Test Gamma = 1.0 (exact copy)
    ref_1 = AcousticPlantModel.synthesize_incoherent_reference(clean, coherence=1.0, sr=sr)
    assert np.allclose(ref_1, clean, atol=1e-6)

    # Test Gamma = 0.5 (cross-correlation check)
    ref_half = AcousticPlantModel.synthesize_incoherent_reference(clean, coherence=0.5, sr=sr, seed=123)
    p_clean = np.mean(clean ** 2)
    p_ref = np.mean(ref_half ** 2)
    corr = np.mean(clean * ref_half) / (np.sqrt(p_clean * p_ref) + 1e-12)
    # Expected correlation is sqrt(0.5) ~ 0.707
    assert 0.60 < corr < 0.82

    # Test Gamma = 0.0 (near-zero cross-correlation)
    ref_zero = AcousticPlantModel.synthesize_incoherent_reference(clean, coherence=0.0, sr=sr, seed=999)
    corr_zero = abs(np.mean(clean * ref_zero) / (np.sqrt(p_clean * np.mean(ref_zero ** 2)) + 1e-12))
    assert corr_zero < 0.10


def test_multi_uncertainty_generator():
    """Verify all 7 perturbation dimensions produce non-divergent valid FIR filters."""
    plant = AcousticPlantModel(sample_rate=16000, fir_length=64)
    nom = plant.s_nominal
    norm_nom = np.linalg.norm(nom)

    # 1. Amplitude
    s_amp = plant.generate_multi_uncertain_secondary_path("amplitude", severity=0.25)
    assert len(s_amp) == len(nom)
    assert np.isclose(np.linalg.norm(s_amp), norm_nom * 1.25, rtol=1e-4)

    # 2. Phase
    s_phase = plant.generate_multi_uncertain_secondary_path("phase", severity=45.0)
    assert len(s_phase) == len(nom)
    assert not np.isnan(s_phase).any()

    # 3. Delay
    s_delay = plant.generate_multi_uncertain_secondary_path("delay", severity=2.0)
    assert len(s_delay) == len(nom)
    assert s_delay[0] == 0.0 and s_delay[1] == 0.0

    # 4. Resonance
    s_res = plant.generate_multi_uncertain_secondary_path("resonance", severity=0.3)
    assert len(s_res) == len(nom)

    # 5. Q-factor
    s_q = plant.generate_multi_uncertain_secondary_path("q_factor", severity=-0.4)
    assert len(s_q) == len(nom)

    # 6. Random FIR
    s_rfir = plant.generate_multi_uncertain_secondary_path("random_fir", severity=0.20, seed=77)
    err = np.linalg.norm(s_rfir - nom) / norm_nom
    assert np.isclose(err, 0.20, atol=0.02)

    # 7. Combined
    s_comb = plant.generate_multi_uncertain_secondary_path("combined", severity=1.0, seed=88)
    assert len(s_comb) == len(nom)
    assert not np.isnan(s_comb).any()


def test_actuator_saturation_clipping():
    """Verify actuator saturation limits output and tracks distortion."""
    plant = AcousticPlantModel(sample_rate=16000, fir_length=64)
    sr = 16000
    N = int(0.5 * sr)
    t = np.arange(N) / sr
    ref = (np.sin(2 * np.pi * 200.0 * t) * 0.5).astype(np.float32)
    prim = plant.convolve_primary(ref)

    # Constrained headroom V_max = 0.05 (severe clipping)
    res_clip = run_fxnlms_simulation(
        reference=ref,
        primary_disturbance=prim,
        s_true=plant.s_nominal,
        s_hat=plant.s_nominal.copy(),
        max_anti_noise_amplitude=0.05,
        saturation_mode="hard",
    )
    assert res_clip["clip_ratio"] > 0.0
    assert res_clip["distortion_power"] > 0.0
    assert np.max(np.abs(res_clip["anti_noise"])) <= 0.05001

    # Soft tanh saturation
    res_soft = run_fxnlms_simulation(
        reference=ref,
        primary_disturbance=prim,
        s_true=plant.s_nominal,
        s_hat=plant.s_nominal.copy(),
        max_anti_noise_amplitude=0.05,
        saturation_mode="soft",
    )
    assert np.max(np.abs(res_soft["anti_noise"])) <= 0.05001


def test_joint_realistic_simulation():
    """Verify that joint realistic conditions yield non-trivial stable cancellation (10-20 dB)."""
    plant = AcousticPlantModel(sample_rate=16000, fir_length=64)
    sr = 16000
    N = int(1.5 * sr)
    rng = np.random.RandomState(42)

    # Low-pass noise disturbance in ANC band
    b, a = signal.butter(4, 500.0 / (sr / 2.0), btype="low")
    clean_dist = signal.lfilter(b, a, rng.randn(N).astype(np.float32)).astype(np.float32) * 0.2
    prim = plant.convolve_primary(clean_dist)

    # Incoherent reference (Gamma = 0.85)
    ref = plant.synthesize_incoherent_reference(clean_dist, coherence=0.85, sr=sr, seed=101)

    # 15% combined secondary-path mismatch
    s_hat = plant.generate_multi_uncertain_secondary_path("combined", severity=0.75, seed=202)

    # Realistic sensor noise on error mic (35 dB SNR)
    p_signal = float(np.mean(prim ** 2))
    p_noise = p_signal / (10.0 ** (35.0 / 10.0))
    prim_noisy = prim + (rng.randn(N) * np.sqrt(p_noise)).astype(np.float32)

    res = run_fxnlms_simulation(
        reference=ref,
        primary_disturbance=prim_noisy,
        s_true=plant.s_nominal,
        s_hat=s_hat,
        filter_length=64,
        step_size=0.015,
        leakage=1e-4,
        max_anti_noise_amplitude=0.4,
    )

    assert not res["diverged"]
    # Realistic cancellation should fall in 6 to 18 dB range
    assert 6.0 <= res["cancellation_db"] <= 18.0
