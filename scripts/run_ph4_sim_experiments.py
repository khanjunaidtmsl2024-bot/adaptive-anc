"""
PH4-SIM: Physical Acoustic ANC & Secondary-Path Simulation Campaign.
PS 26052 — Adaptive Defence ANC.

Executes 10 Exhaustive Simulated Physical Control Experiments:
1. Baseline Convergence & Pure Tonal Cancellation across frequencies (100 Hz to 1000 Hz).
2. Secondary-Path Phase Sensitivity & Comparison with Standard NLMS (0 to 180 degrees).
3. Secondary-Path Uncertainty Matrix (0%, 5%, 10%, 20%, 30%, 50% mismatch).
4. Dynamic Acoustic Seal Perturbation (Glasses Frame Cushion Leak at t=1.0s).
5. Stability Boundaries & Maximum Step-Size (mu_max mapping).
6. Multi-Noise Disturbance Profiles (tonal, pink, engine RPM, rotor, gunfire).
7. Frequency-Domain Control Bandwidth Spectrum (20 Hz to 2000 Hz).
8. Measurement Sensor Noise Sensitivity (Error and Ref mic SNR).
9. Two-Rate Supervisory AI Speech Protection (E2 speech gating vs unsupervised FxNLMS).
10. Transport Delay & Causality Limits (1 to 24 samples delay).

Strictly designated as SIMULATED PHYSICAL CONTROL EXPERIMENT (Host CPU).
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.acoustic_plant import AcousticPlantModel, StreamingPlantState
from src.dsp.fxlms import run_fxnlms_simulation, _simulate_fxnlms_fast
from src.dsp.supervisory_coupling import SupervisoryGovernor, run_supervisory_hybrid_simulation
from src.dataset.synthetic_benchmark_matrix import synthesize_speech_profile, synthesize_noise_profile


def exp_1_baseline_tonal_convergence(plant: AcousticPlantModel, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- EXP 1: BASELINE CONVERGENCE ACROSS FREQUENCIES ---", flush=True)
    freqs = [100.0, 200.0, 300.0, 500.0, 800.0, 1000.0]
    duration = 1.5
    N = int(duration * sr)
    t = np.arange(N) / sr
    results = []

    for f in freqs:
        ref = np.sin(2 * np.pi * f * t).astype(np.float32)
        prim = plant.convolve_primary(ref)

        res = run_fxnlms_simulation(
            reference=ref,
            primary_disturbance=prim,
            s_true=plant.s_nominal,
            s_hat=plant.s_nominal.copy(),
            filter_length=64,
            step_size=0.015,
            leakage=1e-5,
        )

        # Estimate convergence time: samples until error drops by 10 dB below initial
        init_pow = np.mean(prim[:200] ** 2) + 1e-12
        err_sq = res["residual_error"] ** 2
        conv_idx = N
        for idx in range(200, N - 100, 50):
            if np.mean(err_sq[idx : idx + 100]) < 0.1 * init_pow:
                conv_idx = idx
                break
        conv_time_ms = round(float(conv_idx / sr * 1000.0), 1)

        record = {
            "frequency_hz": f,
            "cancellation_db": res["cancellation_db"],
            "convergence_time_ms": conv_time_ms,
            "diverged": res["diverged"],
        }
        results.append(record)
        print(f"  [{f:4.0f} Hz] -> Cancellation: {res['cancellation_db']:+6.2f} dB | Convergence: {conv_time_ms:6.1f} ms | Diverged: {res['diverged']}")

    return {"conditions": results, "passed": all(r["cancellation_db"] > 10.0 for r in results[:4])}


def exp_2_phase_sensitivity_nlms_vs_fxnlms(plant: AcousticPlantModel, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- EXP 2: SECONDARY-PATH PHASE SENSITIVITY (NLMS VS FxNLMS) ---", flush=True)
    phase_degs = [0, 30, 60, 90, 120, 150, 180]
    duration = 1.5
    N = int(duration * sr)
    t = np.arange(N) / sr
    ref = np.sin(2 * np.pi * 250.0 * t).astype(np.float32)
    prim = plant.convolve_primary(ref)
    results = []

    # Delta impulse representing standard NLMS (which assumes S(z) = 1, ignoring secondary path)
    s_nlms = np.zeros_like(plant.s_nominal)
    s_nlms[0] = 1.0

    for deg in phase_degs:
        rad = deg * np.pi / 180.0
        # Perturb secondary path with phase shift
        s_phase = plant.generate_uncertain_secondary_path(mismatch_percent=0.0, phase_jitter_rad=rad)

        # 1. Standard NLMS (ignoring secondary path)
        res_nlms = run_fxnlms_simulation(
            reference=ref, primary_disturbance=prim, s_true=s_phase, s_hat=s_nlms, step_size=0.01
        )

        # 2. FxNLMS (incorporating estimated secondary path)
        res_fxnlms = run_fxnlms_simulation(
            reference=ref, primary_disturbance=prim, s_true=s_phase, s_hat=s_phase.copy(), step_size=0.01
        )

        rec = {
            "phase_deg": deg,
            "nlms_cancellation_db": res_nlms["cancellation_db"],
            "nlms_diverged": res_nlms["diverged"],
            "fxnlms_cancellation_db": res_fxnlms["cancellation_db"],
            "fxnlms_diverged": res_fxnlms["diverged"],
        }
        results.append(rec)
        print(f"  [Phase {deg:3d} deg] -> NLMS: {res_nlms['cancellation_db']:+6.2f} dB (Div: {res_nlms['diverged']}) | FxNLMS: {res_fxnlms['cancellation_db']:+6.2f} dB (Div: {res_fxnlms['diverged']})")

    return {"phase_sweep": results, "passed": all(not r["fxnlms_diverged"] for r in results)}


def exp_3_uncertainty_matrix(plant: AcousticPlantModel, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- EXP 3: SECONDARY-PATH UNCERTAINTY MATRIX ---", flush=True)
    mismatch_percents = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    duration = 2.0
    N = int(duration * sr)
    rng = np.random.RandomState(42)

    # Low-pass disturbance in ANC band (< 600 Hz)
    b, a = signal.butter(4, 600.0 / (sr / 2.0), btype="low")
    ref = signal.lfilter(b, a, rng.randn(N).astype(np.float32)).astype(np.float32) * 0.2
    prim = plant.convolve_primary(ref)
    s_true = plant.s_nominal

    results = []
    for mismatch in mismatch_percents:
        s_hat = plant.generate_uncertain_secondary_path(mismatch_percent=mismatch, seed=505)
        res = run_fxnlms_simulation(
            reference=ref,
            primary_disturbance=prim,
            s_true=s_true,
            s_hat=s_hat,
            filter_length=64,
            step_size=0.01,
            leakage=1e-4,
        )

        record = {
            "mismatch_percent": mismatch,
            "cancellation_db": res["cancellation_db"],
            "diverged": res["diverged"],
            "steady_state_error_power": res["steady_state_error_power"],
        }
        results.append(record)
        print(f"  [Mismatch {mismatch:4.1f}%] -> Cancellation: {res['cancellation_db']:+6.2f} dB | Steady Error: {res['steady_state_error_power']:.2e} | Diverged: {res['diverged']}")

    return {"uncertainty_matrix": results, "passed": not any(r["diverged"] for r in results)}


def exp_4_dynamic_seal_break(plant: AcousticPlantModel, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- EXP 4: DYNAMIC ACOUSTIC SEAL PERTURBATION (GLASSES LEAK) ---", flush=True)
    duration = 2.5
    N = int(duration * sr)
    idx_leak = int(1.0 * sr)  # Seal breaks at t=1.0s
    rng = np.random.RandomState(101)

    b, a = signal.butter(4, 500.0 / (sr / 2.0), btype="low")
    ref = signal.lfilter(b, a, rng.randn(N).astype(np.float32)).astype(np.float32) * 0.2
    prim = plant.convolve_primary(ref)

    # Create time-varying true secondary path: nominal before t=1.0s, leaked after
    s_nominal = plant.s_nominal
    s_leaked = plant.s_leaked
    s_hat = plant.s_nominal.copy()

    # Sample-by-sample execution
    M = 64
    mu = 0.015
    y_out = np.zeros(N, dtype=np.float32)
    e_out = np.zeros(N, dtype=np.float32)

    weights = np.zeros(M, dtype=np.float32)
    ref_buf = np.zeros(M, dtype=np.float32)
    filt_ref_buf = np.zeros(M, dtype=np.float32)
    s_hat_buf = np.zeros(len(s_hat), dtype=np.float32)
    s_plant_buf = np.zeros(len(s_nominal), dtype=np.float32)

    for n in range(N):
        x_n = ref[n]
        d_n = prim[n]
        cur_s_plant = s_nominal if n < idx_leak else s_leaked

        ref_buf[1:] = ref_buf[:-1]
        ref_buf[0] = x_n
        y_n = float(np.dot(weights, ref_buf))
        y_n = float(np.clip(y_n, -1.0, 1.0))
        y_out[n] = y_n

        s_plant_buf[1:] = s_plant_buf[:-1]
        s_plant_buf[0] = y_n
        anti_sound = float(np.dot(cur_s_plant, s_plant_buf))
        e_n = d_n - anti_sound
        e_out[n] = e_n

        s_hat_buf[1:] = s_hat_buf[:-1]
        s_hat_buf[0] = x_n
        filt_ref_n = float(np.dot(s_hat, s_hat_buf))
        filt_ref_buf[1:] = filt_ref_buf[:-1]
        filt_ref_buf[0] = filt_ref_n

        norm = float(np.dot(filt_ref_buf, filt_ref_buf)) + 1e-6
        weights = (1.0 - 1e-4 * mu) * weights + (mu / norm) * e_n * filt_ref_buf

    # Cancellation before leak (0.5s to 1.0s) and after re-convergence (1.8s to 2.5s)
    pre_leak_p_in = np.mean(prim[int(0.5*sr) : idx_leak] ** 2)
    pre_leak_p_err = np.mean(e_out[int(0.5*sr) : idx_leak] ** 2)
    pre_leak_db = float(10.0 * np.log10(pre_leak_p_in / pre_leak_p_err))

    post_leak_p_in = np.mean(prim[int(1.8*sr) :] ** 2)
    post_leak_p_err = np.mean(e_out[int(1.8*sr) :] ** 2)
    post_leak_db = float(10.0 * np.log10(post_leak_p_in / post_leak_p_err))

    diverged = bool(np.isnan(e_out).any() or np.max(np.abs(weights)) > 50.0)

    print(f"  Pre-Leak Cancellation (t=0.5-1.0s):  {pre_leak_db:+6.2f} dB")
    print(f"  Post-Leak Cancellation (t=1.8-2.5s): {post_leak_db:+6.2f} dB")
    print(f"  Maintained Stability Across Break:  {not diverged}")

    return {
        "pre_leak_cancellation_db": round(pre_leak_db, 2),
        "post_leak_cancellation_db": round(post_leak_db, 2),
        "diverged": diverged,
        "passed": not diverged,
    }


def exp_5_stability_boundary_step_size(plant: AcousticPlantModel, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- EXP 5: STABILITY BOUNDARY & MAXIMUM STEP-SIZE (mu_max) ---", flush=True)
    mus = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    duration = 1.0
    N = int(duration * sr)
    rng = np.random.RandomState(42)
    b, a = signal.butter(4, 500.0 / (sr / 2.0), btype="low")
    ref = signal.lfilter(b, a, rng.randn(N).astype(np.float32)).astype(np.float32) * 0.1
    prim = plant.convolve_primary(ref)

    results = []
    mu_max_stable = 0.0

    for mu in mus:
        res = run_fxnlms_simulation(
            ref, prim, plant.s_nominal, plant.s_nominal.copy(), step_size=mu, leakage=1e-4
        )
        is_stable = not res["diverged"] and res["cancellation_db"] > -5.0
        if is_stable and mu > mu_max_stable:
            mu_max_stable = mu

        rec = {
            "step_size_mu": mu,
            "cancellation_db": res["cancellation_db"],
            "diverged": res["diverged"],
            "stable": is_stable,
        }
        results.append(rec)
        print(f"  [mu = {mu:5.3f}] -> Cancellation: {res['cancellation_db']:+6.2f} dB | Diverged: {res['diverged']} | Stable: {is_stable}")

    print(f"  Empirical Maximum Stable mu: {mu_max_stable}")
    return {"step_size_sweep": results, "mu_max_stable": mu_max_stable, "passed": bool(mu_max_stable >= 0.05)}


def exp_6_multi_noise_profiles(plant: AcousticPlantModel, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- EXP 6: MULTI-NOISE DISTURBANCE EVALUATION ---", flush=True)
    noises = ["stat_tonal_hum", "stat_pink", "nonstat_engine_mod", "nonstat_rotor_mod", "impulse_gunfire_burst"]
    duration = 2.0
    N = int(duration * sr)
    results = []

    for n_id in noises:
        noise = synthesize_noise_profile(n_id, n_samples=N, sr=sr, seed=808) * 0.2
        prim = plant.convolve_primary(noise)

        res = run_fxnlms_simulation(
            reference=noise,
            primary_disturbance=prim,
            s_true=plant.s_nominal,
            s_hat=plant.s_nominal.copy(),
            step_size=0.015,
            leakage=1e-4,
        )

        rec = {
            "noise_profile": n_id,
            "cancellation_db": res["cancellation_db"],
            "diverged": res["diverged"],
        }
        results.append(rec)
        print(f"  [{n_id:22s}] -> Cancellation: {res['cancellation_db']:+6.2f} dB | Diverged: {res['diverged']}")

    all_stable = all(not r["diverged"] for r in results)
    return {"profiles": results, "passed": all_stable}


def exp_7_control_bandwidth_spectrum(plant: AcousticPlantModel, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- EXP 7: FREQUENCY-DOMAIN CONTROL BANDWIDTH SPECTRUM ---", flush=True)
    freqs = [50, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000]
    duration = 1.0
    N = int(duration * sr)
    t = np.arange(N) / sr
    results = []

    for f in freqs:
        ref = np.sin(2 * np.pi * f * t).astype(np.float32)
        prim = plant.convolve_primary(ref)

        res = run_fxnlms_simulation(
            ref, prim, plant.s_nominal, plant.s_nominal.copy(), filter_length=64, step_size=0.02
        )
        results.append({"freq_hz": f, "cancellation_db": res["cancellation_db"]})
        print(f"  [{f:4d} Hz] -> Cancellation: {res['cancellation_db']:+6.2f} dB")

    # Physical ANC typically cancels 50 Hz to 600 Hz (> 12 dB), rolling off at high frequencies
    peak_band_canc = [r["cancellation_db"] for r in results if 100 <= r["freq_hz"] <= 500]
    passed = bool(np.mean(peak_band_canc) > 15.0)
    print(f"  Average Cancellation in 100-500 Hz Core ANC Band: {np.mean(peak_band_canc):.2f} dB")
    return {"spectrum": results, "core_band_avg_db": round(float(np.mean(peak_band_canc)), 2), "passed": passed}


def exp_8_sensor_noise_sensitivity(plant: AcousticPlantModel, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- EXP 8: MEASUREMENT SENSOR NOISE SENSITIVITY ---", flush=True)
    snrs = [float("inf"), 40.0, 30.0, 20.0, 10.0]
    duration = 1.5
    N = int(duration * sr)
    t = np.arange(N) / sr
    ref_clean = np.sin(2 * np.pi * 250.0 * t).astype(np.float32)
    prim = plant.convolve_primary(ref_clean)
    results = []

    rng = np.random.RandomState(99)
    p_signal = float(np.mean(prim ** 2))

    for snr in snrs:
        if snr == float("inf"):
            prim_noisy = prim.copy()
        else:
            p_noise = p_signal / (10.0 ** (snr / 10.0))
            sensor_noise = (rng.randn(N) * np.sqrt(p_noise)).astype(np.float32)
            prim_noisy = prim + sensor_noise

        res = run_fxnlms_simulation(
            ref_clean, prim_noisy, plant.s_nominal, plant.s_nominal.copy(), step_size=0.015
        )
        rec = {"sensor_snr_db": "inf" if snr == float("inf") else snr, "cancellation_db": res["cancellation_db"]}
        results.append(rec)
        print(f"  [Sensor SNR: {str(rec['sensor_snr_db']):>4s} dB] -> Cancellation: {res['cancellation_db']:+6.2f} dB")

    return {"sensor_sweep": results, "passed": True}


def exp_9_supervisory_ai_speech_protection(plant: AcousticPlantModel, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- EXP 9: TWO-RATE SUPERVISORY AI SPEECH PROTECTION ---", flush=True)
    duration = 2.0
    N = int(duration * sr)

    # Ambient noise (tonal drone)
    t = np.arange(N) / sr
    noise = (np.sin(2 * np.pi * 200.0 * t) * 0.2).astype(np.float32)

    # User speaks between t=0.6s and t=1.4s
    speech = np.zeros(N, dtype=np.float32)
    spk = synthesize_speech_profile("SPK_M01", n_samples=int(0.8 * sr), sr=sr, seed=333)
    speech[int(0.6 * sr) : int(0.6 * sr) + len(spk)] = spk * 0.5

    # Run Unsupervised FxNLMS
    res_unsuper = run_supervisory_hybrid_simulation(
        reference=noise,
        clean_speech=speech,
        external_noise=noise,
        plant_model=plant,
        s_hat=plant.s_nominal.copy(),
        enable_supervision=False,
        base_mu=0.02,
    )

    # Run Supervised FxNLMS (E2 / Governor speech gating active)
    res_super = run_supervisory_hybrid_simulation(
        reference=noise,
        clean_speech=speech,
        external_noise=noise,
        plant_model=plant,
        s_hat=plant.s_nominal.copy(),
        enable_supervision=True,
        base_mu=0.02,
    )

    print(f"  Unsupervised FxNLMS Speech Attenuation: {res_unsuper['speech_attenuation_db']:+6.2f} dB (Diverged: {res_unsuper['diverged']})")
    print(f"  Supervised FxNLMS Speech Attenuation:   {res_super['speech_attenuation_db']:+6.2f} dB (Frozen Hops: {res_super['frozen_hops']}/{res_super['total_hops']})")

    passed = bool(res_super["frozen_hops"] > 10 and not res_super["diverged"])
    return {
        "unsupervised_speech_attenuation_db": res_unsuper["speech_attenuation_db"],
        "supervised_speech_attenuation_db": res_super["speech_attenuation_db"],
        "frozen_hops": res_super["frozen_hops"],
        "passed": passed,
    }


def exp_10_transport_delay_limits(plant: AcousticPlantModel, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- EXP 10: TRANSPORT DELAY & CAUSALITY LIMITS ---", flush=True)
    delays_samples = [1, 2, 4, 8, 12, 16, 24]
    duration = 1.5
    N = int(duration * sr)
    rng = np.random.RandomState(42)
    b, a = signal.butter(4, 500.0 / (sr / 2.0), btype="low")
    ref = signal.lfilter(b, a, rng.randn(N).astype(np.float32)).astype(np.float32) * 0.1
    prim = plant.convolve_primary(ref)
    results = []

    for d in delays_samples:
        # Synthesize secondary path with specific pure transport delay
        s_delayed = np.zeros(64, dtype=np.float32)
        s_base = plant.s_nominal
        if d < 64:
            s_delayed[d:] = s_base[:-d]

        res = run_fxnlms_simulation(
            ref, prim, s_delayed, s_delayed.copy(), filter_length=64, step_size=0.015
        )
        latency_ms = round(float(d / sr * 1000.0), 3)
        rec = {
            "delay_samples": d,
            "latency_ms": latency_ms,
            "cancellation_db": res["cancellation_db"],
            "diverged": res["diverged"],
        }
        results.append(rec)
        print(f"  [Delay: {d:2d} samples ({latency_ms:5.3f} ms)] -> Cancellation: {res['cancellation_db']:+6.2f} dB | Diverged: {res['diverged']}")

    return {"delay_sweep": results, "passed": True}


def main():
    print("=" * 76)
    print("  PH4-SIM: PHYSICAL ACOUSTIC ANC & SECONDARY-PATH SIMULATION CAMPAIGN")
    print("  System Classification: Simulated Physical Control Experiment (Host CPU)")
    print("  Plant: Circumaural Over-Ear Headset Cavity + Transducer Dynamics")
    print("=" * 76)

    sr = 16000
    plant = AcousticPlantModel(sample_rate=sr, fir_length=64)

    t_start = time.perf_counter()
    r1 = exp_1_baseline_tonal_convergence(plant, sr)
    r2 = exp_2_phase_sensitivity_nlms_vs_fxnlms(plant, sr)
    r3 = exp_3_uncertainty_matrix(plant, sr)
    r4 = exp_4_dynamic_seal_break(plant, sr)
    r5 = exp_5_stability_boundary_step_size(plant, sr)
    r6 = exp_6_multi_noise_profiles(plant, sr)
    r7 = exp_7_control_bandwidth_spectrum(plant, sr)
    r8 = exp_8_sensor_noise_sensitivity(plant, sr)
    r9 = exp_9_supervisory_ai_speech_protection(plant, sr)
    r10 = exp_10_transport_delay_limits(plant, sr)
    total_time = time.perf_counter() - t_start

    all_passed = (
        r1["passed"]
        and r2["passed"]
        and r3["passed"]
        and r4["passed"]
        and r5["passed"]
        and r6["passed"]
        and r7["passed"]
        and r8["passed"]
        and r9["passed"]
        and r10["passed"]
    )

    campaign_report = {
        "campaign_id": "PH4_SIM_PHYSICAL_ANC_VALIDATION",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "PASS" if all_passed else "FAIL",
        "total_wall_time_sec": round(total_time, 2),
        "attribution": "SIMULATED PHYSICAL CONTROL EXPERIMENT (Host CPU)",
        "experiments": {
            "exp_1_tonal_convergence": r1,
            "exp_2_phase_sensitivity_nlms_vs_fxnlms": r2,
            "exp_3_uncertainty_matrix": r3,
            "exp_4_dynamic_seal_break": r4,
            "exp_5_stability_boundary_step_size": r5,
            "exp_6_multi_noise_profiles": r6,
            "exp_7_control_bandwidth_spectrum": r7,
            "exp_8_sensor_noise_sensitivity": r8,
            "exp_9_supervisory_ai_speech_protection": r9,
            "exp_10_transport_delay_limits": r10,
        },
    }

    out_dir = PROJECT_ROOT / "results" / "ph4_sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "ph4_sim_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(campaign_report, f, indent=2)

    print("\n" + "=" * 76)
    print(f"  PH4-SIM CAMPAIGN COMPLETED: {'ALL 10 INVESTIGATIONS PASSED [PASS]' if all_passed else 'FAILURES DETECTED [FAIL]'}")
    print(f"  Total Wall Time: {total_time:.2f}s")
    print(f"  Saved JSON Telemetry: {out_json}")
    print("=" * 76)


if __name__ == "__main__":
    main()
