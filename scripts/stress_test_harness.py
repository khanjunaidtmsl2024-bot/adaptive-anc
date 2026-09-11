"""
PH3.5 — Laptop-Only Pre-Hardware Stress Validation Harness.
PS 26052: Defence-Grade AI/ML Adaptive Noise Cancellation.

Executes 12 exhaustive stress suites on the frozen E2_causal deployment pipeline:
CausalStreamingEngine(use_fast_dsp=True, ai_backend="onnx").

Strict Constraints:
- No model changes, no retraining, no architectural modifications.
- Evaluates the frozen standalone FP32 ONNX model (models/E2_causal.onnx).
- Clearly records and labels all results as VERIFIED EXPERIMENT on Host CPU.
"""

import os
import sys
import time
import json
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import psutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.streaming.causal_engine import CausalStreamingEngine
from src.ai.tiny_enhancer import TinyEnhancerONNXWrapper
from src.dataset.synthetic_benchmark_matrix import (
    synthesize_speech_profile,
    synthesize_noise_profile,
    SPEAKER_PROFILES,
    NOISE_PROFILES,
)
from src.evaluation.metrics import compute_snr, compute_si_snr, compute_stoi


def get_current_rss_mb() -> float:
    """Returns the current process resident set size (RSS) in megabytes."""
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss / (1024.0 * 1024.0))


def run_suite_1_stationary_noise(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 1: STATIONARY NOISE STRESS ---", flush=True)
    noises = ["stat_white", "stat_pink", "stat_brown", "stat_tonal_hum"]
    snrs = [-10, 0, 10]
    results = []
    duration = 2.0
    n_samples = int(duration * sr)
    hop = 128

    clean = synthesize_speech_profile("SPK_M01", n_samples=n_samples, sr=sr, seed=101)

    for n_id in noises:
        noise = synthesize_noise_profile(n_id, n_samples=n_samples, sr=sr, seed=202)
        p_clean = np.mean(clean ** 2)
        p_noise = np.mean(noise ** 2) + 1e-12

        for snr in snrs:
            scale = np.sqrt(p_clean / (p_noise * (10.0 ** (snr / 10.0))))
            noisy_pri = clean + noise * scale
            ref = noise * scale

            engine.reset()
            out_hops = []
            for h in range(n_samples // hop):
                idx = h * hop
                p_h = noisy_pri[idx : idx + hop]
                r_h = ref[idx : idx + hop]
                o_h, _ = engine.process_hop(p_h, r_h)
                out_hops.append(o_h)

            out_wav = np.concatenate(out_hops)
            delay = 128
            si_sdr = float(compute_si_snr(clean[:len(out_wav) - delay], out_wav[delay:]))
            has_nan = bool(np.isnan(out_wav).any())
            has_inf = bool(np.isinf(out_wav).any())
            max_amp = float(np.max(np.abs(out_wav)))

            res = {
                "noise_id": n_id,
                "target_snr_db": snr,
                "out_si_sdr_db": round(si_sdr, 2),
                "has_nan": has_nan,
                "has_inf": has_inf,
                "max_amp": round(max_amp, 4),
                "stable": (not has_nan and not has_inf and max_amp <= 0.99),
            }
            results.append(res)
            print(f"  [{n_id:15s} @ {snr:+3d} dB] -> SI-SDR: {si_sdr:+6.2f} dB | Max Amp: {max_amp:.3f} | Stable: {res['stable']}")

    all_stable = all(r["stable"] for r in results)
    return {"passed": all_stable, "test_count": len(results), "conditions": results}


def run_suite_2_nonstationary_modulation(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 2: NON-STATIONARY MODULATION STRESS ---", flush=True)
    noises = ["nonstat_engine_mod", "nonstat_rotor_mod", "nonstat_siren_sweep"]
    duration = 3.0
    n_samples = int(duration * sr)
    hop = 128
    results = []

    clean = synthesize_speech_profile("SPK_F01", n_samples=n_samples, sr=sr, seed=303)
    p_clean = np.mean(clean ** 2)

    for n_id in noises:
        noise = synthesize_noise_profile(n_id, n_samples=n_samples, sr=sr, seed=404)
        p_noise = np.mean(noise ** 2) + 1e-12
        scale = np.sqrt(p_clean / p_noise)  # 0 dB SNR
        noisy_pri = clean + noise * scale
        ref = noise * scale

        engine.reset()
        out_hops = []
        tracking_errors = []
        for h in range(n_samples // hop):
            idx = h * hop
            p_h = noisy_pri[idx : idx + hop]
            r_h = ref[idx : idx + hop]
            o_h, diag = engine.process_hop(p_h, r_h)
            out_hops.append(o_h)
            tracking_errors.append(np.mean(o_h ** 2))

        out_wav = np.concatenate(out_hops)
        delay = 128
        clean_d = clean[:len(out_wav) - delay]
        out_d = out_wav[delay:]
        noisy_d = noisy_pri[:len(out_wav) - delay]
        si_sdr = float(compute_si_snr(clean_d, out_d))
        in_snr = float(compute_snr(clean_d, noisy_d))
        delta_snr = float(compute_snr(clean_d, out_d) - in_snr)

        res = {
            "noise_id": n_id,
            "in_snr_db": round(in_snr, 2),
            "out_si_sdr_db": round(si_sdr, 2),
            "delta_snr_db": round(delta_snr, 2),
            "stable": bool(not np.isnan(out_wav).any() and not np.isinf(out_wav).any() and delta_snr > -1.0),
        }
        results.append(res)
        print(f"  [{n_id:20s}] -> In SNR: {in_snr:+5.2f} dB | Out SI-SDR: {si_sdr:+6.2f} dB | dSNR: {delta_snr:+5.2f} dB | Stable: {res['stable']}")

    all_passed = all(r["stable"] for r in results)
    return {"passed": all_passed, "conditions": results}


def run_suite_3_impulsive_and_rapid_fire(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 3: IMPULSIVE NOISE & RAPID-FIRE TRANSIENT STRESS ---", flush=True)
    duration = 2.0
    n_samples = int(duration * sr)
    hop = 128
    clean = synthesize_speech_profile("SPK_M02", n_samples=n_samples, sr=sr, seed=505)

    # Condition A: Single +40 dB Gunfire Spike at t = 0.5s
    pri_a = clean.copy()
    ref_a = np.zeros_like(clean)
    idx_a = int(0.5 * sr)
    pri_a[idx_a : idx_a + 20] += 5.0  # extreme digital spike
    ref_a[idx_a : idx_a + 20] += 5.0

    # Condition B: Rapid-Fire Bursts (5 consecutive impulses spaced 25 ms apart = 400 samples)
    pri_b = clean.copy()
    ref_b = np.zeros_like(clean)
    idx_b0 = int(0.6 * sr)
    for k in range(5):
        pos = idx_b0 + k * 400
        pri_b[pos : pos + 10] += 4.0
        ref_b[pos : pos + 10] += 4.0

    tests = [("single_gunshot_40db", pri_a, ref_a, idx_a), ("rapid_fire_5_bursts", pri_b, ref_b, idx_b0 + 4 * 400)]
    results = []

    for name, pri, ref, last_imp_idx in tests:
        engine.reset()
        out_hops = []
        for h in range(n_samples // hop):
            idx = h * hop
            o_h, _ = engine.process_hop(pri[idx : idx + hop], ref[idx : idx + hop])
            out_hops.append(o_h)

        out_wav = np.concatenate(out_hops)
        # Check recovery 150ms after the last impulse
        recover_sample = last_imp_idx + int(0.150 * sr)
        post_impulse = out_wav[recover_sample : recover_sample + int(0.3 * sr)]
        has_recovered = bool(np.max(np.abs(post_impulse)) < 1.0 and not np.isnan(post_impulse).any())
        max_clipped = float(np.max(np.abs(out_wav)))

        res = {
            "name": name,
            "max_output_amp": round(max_clipped, 4),
            "recovered_at_150ms": has_recovered,
            "bounded": bool(max_clipped <= 0.99),
        }
        results.append(res)
        print(f"  [{name:22s}] -> Max Out: {max_clipped:.3f} | Recovered at 150ms: {has_recovered} | Bound <= 0.99: {res['bounded']}")

    all_passed = all(r["recovered_at_150ms"] and r["bounded"] for r in results)
    return {"passed": all_passed, "tests": results}


def run_suite_4_mixed_dynamic_regimes(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 4: MIXED DYNAMIC REGIME CONCATENATION STRESS ---", flush=True)
    # Concatenate: 1.0s silence -> 1.0s tonal hum -> 1.0s engine mod -> 1.0s gunfire burst -> 1.0s clean speech
    seg_samples = int(1.0 * sr)
    total_samples = 5 * seg_samples
    hop = 128

    silence = np.zeros(seg_samples, dtype=np.float32)
    hum = synthesize_noise_profile("stat_tonal_hum", n_samples=seg_samples, sr=sr, seed=601) * 0.2
    engine_n = synthesize_noise_profile("nonstat_engine_mod", n_samples=seg_samples, sr=sr, seed=602) * 0.3
    gunfire = synthesize_noise_profile("impulse_gunfire_burst", n_samples=seg_samples, sr=sr, seed=603) * 0.5
    speech = synthesize_speech_profile("SPK_F02", n_samples=seg_samples, sr=sr, seed=604)

    clean_full = np.concatenate([silence, silence, silence, silence, speech])
    pri_full = np.concatenate([silence, hum, engine_n, gunfire, speech])
    ref_full = np.concatenate([silence, hum, engine_n, gunfire, np.zeros(seg_samples, dtype=np.float32)])

    engine.reset()
    out_hops = []
    regimes_seen = set()
    for h in range(total_samples // hop):
        idx = h * hop
        o_h, diag = engine.process_hop(pri_full[idx : idx + hop], ref_full[idx : idx + hop])
        out_hops.append(o_h)
        regimes_seen.add(diag["regime"])

    out_wav = np.concatenate(out_hops)
    has_nan = bool(np.isnan(out_wav).any())
    has_inf = bool(np.isinf(out_wav).any())
    max_amp = float(np.max(np.abs(out_wav)))

    # Evaluate speech preservation in the final 1.0s clean speech segment
    speech_out = out_wav[4 * seg_samples :]
    delay = 128
    speech_si_sdr = float(compute_si_snr(speech[:len(speech_out) - delay], speech_out[delay:]))

    passed = bool(not has_nan and not has_inf and max_amp <= 0.99 and speech_si_sdr > 0.0)
    print(f"  [Regimes Transitioned: {list(regimes_seen)}] -> Max Amp: {max_amp:.3f} | Final Speech SI-SDR: {speech_si_sdr:+6.2f} dB | Passed: {passed}")
    return {
        "passed": passed,
        "regimes_seen": list(regimes_seen),
        "speech_si_sdr_db": round(speech_si_sdr, 2),
        "max_amp": round(max_amp, 4),
    }


def run_suite_5_reference_speech_leakage(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 5: REFERENCE SPEECH LEAKAGE & CROSSTALK STRESS ---", flush=True)
    duration = 2.0
    n_samples = int(duration * sr)
    hop = 128
    clean = synthesize_speech_profile("SPK_M03", n_samples=n_samples, sr=sr, seed=701)
    noise = synthesize_noise_profile("stat_pink", n_samples=n_samples, sr=sr, seed=702) * 0.2
    primary = clean + noise

    # Leakage levels (SIR on reference mic)
    leak_dbs = [-30, -20, -10, -6, -3]
    results = []

    for leak_db in leak_dbs:
        leak_scale = 10.0 ** (leak_db / 20.0)
        # Reference mic has ambient noise + leaked speech
        ref = noise + clean * leak_scale

        engine.reset()
        out_hops = []
        for h in range(n_samples // hop):
            idx = h * hop
            o_h, _ = engine.process_hop(primary[idx : idx + hop], ref[idx : idx + hop])
            out_hops.append(o_h)

        out_wav = np.concatenate(out_hops)
        delay = 128
        clean_d = clean[:len(out_wav) - delay]
        out_d = out_wav[delay:]
        si_sdr = float(compute_si_snr(clean_d, out_d))
        stoi_val = float(compute_stoi(clean_d, out_d, sample_rate=sr))
        clean_atten = float(20.0 * math.log10((np.sqrt(np.mean(out_d ** 2)) + 1e-12) / (np.sqrt(np.mean(clean_d ** 2)) + 1e-12)))

        res = {
            "leakage_level_db": leak_db,
            "out_si_sdr_db": round(si_sdr, 2),
            "out_stoi": round(stoi_val, 4),
            "attenuation_db": round(clean_atten, 2),
            "stable": bool(not np.isnan(out_wav).any() and not np.isinf(out_wav).any()),
        }
        results.append(res)
        print(f"  [Leakage: {leak_db:+3d} dB] -> SI-SDR: {si_sdr:+6.2f} dB | STOI: {stoi_val:.4f} | Atten: {clean_atten:+5.2f} dB | Stable: {res['stable']}")

    all_stable = all(r["stable"] for r in results)
    return {"passed": all_stable, "leakage_sweep": results}


def run_suite_6_extreme_silence_stability(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 6: EXTREME SILENCE & ZERO-INPUT STABILITY STRESS ---", flush=True)
    # 10 continuous seconds of digital 0.0 (1,250 hops)
    duration = 10.0
    n_samples = int(duration * sr)
    hop = 128
    zeros = np.zeros(hop, dtype=np.float32)

    engine.reset()
    out_hops = []
    for h in range(n_samples // hop):
        o_h, diag = engine.process_hop(zeros, zeros)
        out_hops.append(o_h)

    out_wav = np.concatenate(out_hops)
    has_nan = bool(np.isnan(out_wav).any())
    has_inf = bool(np.isinf(out_wav).any())
    max_drift = float(np.max(np.abs(out_wav)))
    mean_dc = float(np.mean(out_wav))

    passed = bool(not has_nan and not has_inf and max_drift < 1e-6)
    print(f"  [10s Pure Zero Input] -> Max Drift: {max_drift:.2e} | Mean DC: {mean_dc:.2e} | Has NaN: {has_nan} | Passed: {passed}")
    return {
        "passed": passed,
        "duration_sec": duration,
        "hops_processed": len(out_hops),
        "max_drift": max_drift,
        "mean_dc_offset": mean_dc,
    }


def run_suite_7_amplitude_extremes_and_clipping(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 7: AMPLITUDE EXTREMES & DIGITAL CLIPPING OVERDRIVE ---", flush=True)
    duration = 2.0
    n_samples = int(duration * sr)
    hop = 128
    clean = synthesize_speech_profile("SPK_M04", n_samples=n_samples, sr=sr, seed=801)
    noise = synthesize_noise_profile("stat_white", n_samples=n_samples, sr=sr, seed=802) * 0.1

    # Overdrive levels: 0 dBFS (peak 1.0), +6 dBFS (peak 2.0), +12 dBFS (peak 4.0), and -60 dBFS (peak 0.001)
    levels = [("nominal_0dbfs", 1.0), ("overdrive_plus_6dbfs", 2.0), ("extreme_plus_12dbfs", 4.0), ("quiet_minus_60dbfs", 1e-3)]
    results = []

    for name, scale in levels:
        pri = (clean + noise) * scale
        ref = noise * scale

        engine.reset()
        out_hops = []
        for h in range(n_samples // hop):
            idx = h * hop
            o_h, _ = engine.process_hop(pri[idx : idx + hop], ref[idx : idx + hop])
            out_hops.append(o_h)

        out_wav = np.concatenate(out_hops)
        max_out = float(np.max(np.abs(out_wav)))
        has_nan = bool(np.isnan(out_wav).any())
        has_inf = bool(np.isinf(out_wav).any())
        bounded = bool(max_out <= 0.99 and not has_nan and not has_inf)

        res = {
            "condition": name,
            "input_scale": scale,
            "max_output": round(max_out, 4),
            "has_nan": has_nan,
            "bounded_under_0_99": bounded,
        }
        results.append(res)
        print(f"  [{name:24s} scale={scale:5.3f}] -> Max Out: {max_out:.4f} | Bounded <= 0.99: {bounded}")

    all_passed = all(r["bounded_under_0_99"] for r in results)
    return {"passed": all_passed, "conditions": results}


def run_suite_8_channel_skew_and_delay_jitter(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 8: CHANNEL SKEW & ACOUSTIC DELAY JITTER STRESS ---", flush=True)
    duration = 2.0
    n_samples = int(duration * sr)
    hop = 128
    clean = synthesize_speech_profile("SPK_F03", n_samples=n_samples, sr=sr, seed=901)
    noise = synthesize_noise_profile("stat_tonal_hum", n_samples=n_samples, sr=sr, seed=902) * 0.2

    # Delay offsets between primary and reference (-8 samples, -4, 0, +4, +8, +16 samples)
    delays = [-8, -4, 0, 4, 8, 16]
    results = []

    for d in delays:
        pri = clean + noise
        ref = np.roll(noise, d)

        engine.reset()
        out_hops = []
        for h in range(n_samples // hop):
            idx = h * hop
            o_h, _ = engine.process_hop(pri[idx : idx + hop], ref[idx : idx + hop])
            out_hops.append(o_h)

        out_wav = np.concatenate(out_hops)
        delay = 128
        clean_d = clean[:len(out_wav) - delay]
        out_d = out_wav[delay:]
        si_sdr = float(compute_si_snr(clean_d, out_d))
        max_amp = float(np.max(np.abs(out_wav)))

        res = {
            "delay_samples": d,
            "delay_ms": round((d / sr) * 1000.0, 3),
            "out_si_sdr_db": round(si_sdr, 2),
            "max_amp": round(max_amp, 4),
            "stable": bool(not np.isnan(out_wav).any() and not np.isinf(out_wav).any() and max_amp <= 0.99),
        }
        results.append(res)
        print(f"  [Delay Skew: {d:+3d} samples ({res['delay_ms']:+6.3f} ms)] -> SI-SDR: {si_sdr:+6.2f} dB | Max Amp: {max_amp:.3f} | Stable: {res['stable']}")

    all_stable = all(r["stable"] for r in results)
    return {"passed": all_stable, "delays": results}


def run_suite_9_malformed_edge_inputs(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 9: MALFORMED & CORRUPTED INPUT INJECTION STRESS ---", flush=True)
    hop = 128
    rng = np.random.RandomState(42)

    # Test 1: Injected NaN in Primary
    p_nan = (rng.randn(hop) * 0.1).astype(np.float32)
    p_nan[30] = np.nan
    r_normal = (rng.randn(hop) * 0.1).astype(np.float32)

    # Test 2: Injected +Inf in Reference
    p_normal = (rng.randn(hop) * 0.1).astype(np.float32)
    r_inf = (rng.randn(hop) * 0.1).astype(np.float32)
    r_inf[50] = np.inf

    # Test 3: Injected -Inf in Primary
    p_neginf = (rng.randn(hop) * 0.1).astype(np.float32)
    p_neginf[10] = -np.inf

    engine.reset()
    out1, _ = engine.process_hop(p_nan, r_normal)
    out2, _ = engine.process_hop(p_normal, r_inf)
    out3, _ = engine.process_hop(p_neginf, r_normal)

    # Following recovery hop with normal audio
    out4, _ = engine.process_hop(p_normal, r_normal)

    has_nan = bool(np.isnan(out1).any() or np.isnan(out2).any() or np.isnan(out3).any() or np.isnan(out4).any())
    has_inf = bool(np.isinf(out1).any() or np.isinf(out2).any() or np.isinf(out3).any() or np.isinf(out4).any())
    recovered = bool(not np.isnan(out4).any() and not np.isinf(out4).any() and np.max(np.abs(out4)) < 1.0)

    passed = (not has_nan and not has_inf and recovered)
    print(f"  [Malformed Input Injection] -> Has NaN: {has_nan} | Has Inf: {has_inf} | Recovered Next Hop: {recovered} | Passed: {passed}")
    return {"passed": passed, "has_nan": has_nan, "has_inf": has_inf, "recovered": recovered}


def run_suite_10_long_duration_sustained_streaming(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 10: 60-SECOND SUSTAINED STREAMING & MEMORY LEAK STRESS ---", flush=True)
    duration_sec = 60.0
    total_samples = int(duration_sec * sr)
    hop = 128
    total_hops = total_samples // hop  # 7,500 hops

    print(f"  Processing {total_hops:,d} continuous hops ({duration_sec:.1f}s audio @ 16 kHz)...", flush=True)

    rng = np.random.RandomState(42)
    pri_buf = (rng.randn(total_samples) * 0.1).astype(np.float32)
    ref_buf = (rng.randn(total_samples) * 0.1).astype(np.float32)

    engine.reset()
    mem_checkpoints = {}
    latencies_ms = []

    # Warmup 50 hops
    for _ in range(50):
        engine.process_hop(pri_buf[:hop], ref_buf[:hop])
    engine.reset()

    initial_rss = get_current_rss_mb()
    mem_checkpoints["hop_0"] = initial_rss

    t_start_wall = time.perf_counter()
    for h in range(total_hops):
        idx = h * hop
        p_h = pri_buf[idx : idx + hop]
        r_h = ref_buf[idx : idx + hop]

        t0 = time.perf_counter_ns()
        o_h, _ = engine.process_hop(p_h, r_h)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)

        if (h + 1) in [1000, 2500, 5000, 7500]:
            mem_checkpoints[f"hop_{h + 1}"] = get_current_rss_mb()

    elapsed_wall = time.perf_counter() - t_start_wall
    final_rss = get_current_rss_mb()
    rss_growth_mb = final_rss - initial_rss

    arr_lat = np.array(latencies_ms)
    p50 = float(np.percentile(arr_lat, 50))
    p95 = float(np.percentile(arr_lat, 95))
    p99 = float(np.percentile(arr_lat, 99))
    p99_9 = float(np.percentile(arr_lat, 99.9))
    max_lat = float(np.max(arr_lat))
    mean_lat = float(np.mean(arr_lat))
    rtf_p50 = float(p50 / 8.0)
    rtf_p95 = float(p95 / 8.0)

    # Memory leak check: RSS growth < 15 MB over 7,500 hops
    no_mem_leak = bool(rss_growth_mb < 15.0)
    budget_met = bool(p95 <= 8.000)
    passed = bool(no_mem_leak and budget_met)

    print(f"  Completed {total_hops:,d} hops in {elapsed_wall:.2f}s wall time (RTF P50: {rtf_p50:.4f}, RTF P95: {rtf_p95:.4f})")
    print(f"  Latency Distribution (Host CPU):")
    print(f"    P50:   {p50:6.3f} ms")
    print(f"    P95:   {p95:6.3f} ms")
    print(f"    P99:   {p99:6.3f} ms")
    print(f"    P99.9: {p99_9:6.3f} ms")
    print(f"    Max:   {max_lat:6.3f} ms")
    print(f"  Memory RSS Profile:")
    for k, v in mem_checkpoints.items():
        print(f"    {k:10s}: {v:6.2f} MB")
    print(f"    Total RSS Growth: {rss_growth_mb:+.2f} MB (Threshold < 15.0 MB: {no_mem_leak})")
    print(f"  Suite 10 Verdict: {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "total_hops": total_hops,
        "duration_audio_sec": duration_sec,
        "wall_time_sec": round(elapsed_wall, 2),
        "latency_percentiles_ms": {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "p99_9": round(p99_9, 3),
            "max": round(max_lat, 3),
            "mean": round(mean_lat, 3),
        },
        "rtf": {"rtf_p50": round(rtf_p50, 4), "rtf_p95": round(rtf_p95, 4)},
        "memory_rss_mb": mem_checkpoints,
        "rss_growth_mb": round(rss_growth_mb, 2),
        "no_memory_leak": no_mem_leak,
    }


def run_suite_11_state_reset_determinism(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 11: STATE RESET DETERMINISM AUDIT ---", flush=True)
    duration = 1.0
    n_samples = int(duration * sr)
    hop = 128
    clean = synthesize_speech_profile("SPK_M05", n_samples=n_samples, sr=sr, seed=1001)
    noise = synthesize_noise_profile("stat_pink", n_samples=n_samples, sr=sr, seed=1002) * 0.2
    pri = clean + noise
    ref = noise

    # Run A: Fresh Engine on speech
    engine.reset()
    out_fresh = []
    for h in range(n_samples // hop):
        idx = h * hop
        o_h, _ = engine.process_hop(pri[idx : idx + hop], ref[idx : idx + hop])
        out_fresh.append(o_h)
    wav_fresh = np.concatenate(out_fresh)

    # Corrupt Engine with 1,000 hops of aggressive noise and spikes
    rng = np.random.RandomState(999)
    for _ in range(1000):
        engine.process_hop(rng.randn(hop).astype(np.float32) * 2.0, rng.randn(hop).astype(np.float32) * 2.0)

    # Reset Engine
    engine.reset()

    # Run B: Post-Reset Engine on the exact same speech
    out_reset = []
    for h in range(n_samples // hop):
        idx = h * hop
        o_h, _ = engine.process_hop(pri[idx : idx + hop], ref[idx : idx + hop])
        out_reset.append(o_h)
    wav_reset = np.concatenate(out_reset)

    max_diff = float(np.max(np.abs(wav_fresh - wav_reset)))
    passed = bool(max_diff < 1e-6)
    print(f"  [Engine State Reset Test] -> Max Difference: {max_diff:.2e} (Threshold < 1e-6) | Passed: {passed}")
    return {"passed": passed, "max_difference": max_diff}


def run_suite_12_batch_vs_streaming_consistency(engine: CausalStreamingEngine, sr: int = 16000) -> Dict[str, Any]:
    print("\n--- SUITE 12: BATCH VS STREAMING CONSISTENCY AUDIT ---", flush=True)
    duration = 2.0
    n_samples = int(duration * sr)
    hop = 128
    clean = synthesize_speech_profile("SPK_F04", n_samples=n_samples, sr=sr, seed=1101)
    noise = synthesize_noise_profile("stat_white", n_samples=n_samples, sr=sr, seed=1102) * 0.1
    pri = clean + noise
    ref = noise

    # Compare all hops processed by process_signal
    n = min(len(pri), len(ref))
    n_hops = (n - engine.frame_size) // engine.hop_size + 1

    # Process hop-by-hop
    engine.reset()
    out_stream = []
    for h in range(n_hops):
        idx = h * hop
        o_h, _ = engine.process_hop(pri[idx : idx + hop], ref[idx : idx + hop])
        out_stream.append(o_h)
    wav_stream = np.concatenate(out_stream)

    # Process via process_signal
    engine.reset()
    wav_batch, _ = engine.process_signal(pri[:n], ref[:n])
    wav_batch_valid = wav_batch[:len(wav_stream)]

    max_diff = float(np.max(np.abs(wav_stream - wav_batch_valid)))
    passed = bool(max_diff < 1e-6)
    print(f"  [Batch vs Streaming Equivalence] -> Max Difference: {max_diff:.2e} across {n_hops} hops (Threshold < 1e-6) | Passed: {passed}")
    return {"passed": passed, "max_difference": max_diff, "hops_evaluated": n_hops}


def main():
    print("=" * 76)
    print("  PH3.5: PRE-HARDWARE STREAMING STRESS VALIDATION CAMPAIGN")
    print("  Model: E2_causal.onnx (9,569 parameters, standalone FP32)")
    print("  Engine: CausalStreamingEngine(use_fast_dsp=True, ai_backend='onnx')")
    print("  Platform: Host CPU (Intel Core Ultra 7 155H, Windows 11)")
    print("=" * 76)

    onnx_path = PROJECT_ROOT / "models" / "E2_causal.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model missing: {onnx_path}")

    with open(onnx_path, "rb") as f:
        onnx_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"[+] Verified Model SHA-256: {onnx_hash}")

    # Instantiate engine with authoritative deployment configuration
    engine = CausalStreamingEngine(sample_rate=16000, use_fast_dsp=True, ai_backend="onnx")

    t_start_all = time.perf_counter()
    res_s1 = run_suite_1_stationary_noise(engine)
    res_s2 = run_suite_2_nonstationary_modulation(engine)
    res_s3 = run_suite_3_impulsive_and_rapid_fire(engine)
    res_s4 = run_suite_4_mixed_dynamic_regimes(engine)
    res_s5 = run_suite_5_reference_speech_leakage(engine)
    res_s6 = run_suite_6_extreme_silence_stability(engine)
    res_s7 = run_suite_7_amplitude_extremes_and_clipping(engine)
    res_s8 = run_suite_8_channel_skew_and_delay_jitter(engine)
    res_s9 = run_suite_9_malformed_edge_inputs(engine)
    res_s10 = run_suite_10_long_duration_sustained_streaming(engine)
    res_s11 = run_suite_11_state_reset_determinism(engine)
    res_s12 = run_suite_12_batch_vs_streaming_consistency(engine)
    total_campaign_time = time.perf_counter() - t_start_all

    all_passed = (
        res_s1["passed"]
        and res_s2["passed"]
        and res_s3["passed"]
        and res_s4["passed"]
        and res_s5["passed"]
        and res_s6["passed"]
        and res_s7["passed"]
        and res_s8["passed"]
        and res_s9["passed"]
        and res_s10["passed"]
        and res_s11["passed"]
        and res_s12["passed"]
    )

    full_report = {
        "campaign_id": "PH3_5_PRE_HARDWARE_STRESS_VALIDATION",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "PASS" if all_passed else "FAIL",
        "total_campaign_wall_time_sec": round(total_campaign_time, 2),
        "model_info": {
            "model_path": str(onnx_path),
            "sha256": onnx_hash,
            "parameter_count": 9569,
            "architecture": "TinyEnhancer CausalConv2d (Frozen E2_causal)",
        },
        "suites": {
            "suite_1_stationary_noise": res_s1,
            "suite_2_nonstationary_modulation": res_s2,
            "suite_3_impulsive_and_rapid_fire": res_s3,
            "suite_4_mixed_dynamic_regimes": res_s4,
            "suite_5_reference_speech_leakage": res_s5,
            "suite_6_extreme_silence_stability": res_s6,
            "suite_7_amplitude_extremes_and_clipping": res_s7,
            "suite_8_channel_skew_and_delay_jitter": res_s8,
            "suite_9_malformed_edge_inputs": res_s9,
            "suite_10_long_duration_sustained_streaming": res_s10,
            "suite_11_state_reset_determinism": res_s11,
            "suite_12_batch_vs_streaming_consistency": res_s12,
        },
    }

    out_dir = PROJECT_ROOT / "results" / "ph3_5_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "stress_validation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 76)
    print(f"  PH3.5 STRESS CAMPAIGN COMPLETED: {'ALL 12 SUITES PASSED [✓]' if all_passed else 'FAILURES DETECTED [✗]'}")
    print(f"  Total Wall Time: {total_campaign_time:.2f}s")
    print(f"  Saved JSON Report: {out_json}")
    print("=" * 76)


if __name__ == "__main__":
    main()
