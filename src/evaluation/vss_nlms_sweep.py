"""
Gate 2: VSS-NLMS Rigorous Parameter Characterization and Pareto Region Sweep.
PS 26052 — Adaptive Defence ANC.

Sweeps:
- mu_max: [0.01, 0.03, 0.05, 0.10, 0.20]
- filter_length: [32, 64, 128, 256]
- reference_delay: [0, 1, 2, 4, 8, 16]

Measures:
- SNR improvement (dB)
- SI-SDR improvement (dB)
- STOI (intelligibility)
- Speech distortion (dB)
- Convergence time (ms)
- Residual noise power
- Instability / coefficient explosion
- Pareto frontier ranking
"""

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import soundfile as sf
import pystoi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dsp.vss_nlms import VSSNLMSFilter


def calculate_snr(clean: np.ndarray, noisy_or_est: np.ndarray) -> float:
    """Compute classical SNR in dB."""
    noise = noisy_or_est - clean
    p_clean = np.mean(clean ** 2) + 1e-12
    p_noise = np.mean(noise ** 2) + 1e-12
    return float(10.0 * np.log10(p_clean / p_noise))


def calculate_si_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    ref = reference - np.mean(reference)
    est = estimated - np.mean(estimated)
    dot = np.sum(est * ref)
    s_target = (dot / (np.sum(ref ** 2) + 1e-12)) * ref
    e_noise = est - s_target
    p_target = np.sum(s_target ** 2) + 1e-12
    p_noise = np.sum(e_noise ** 2) + 1e-12
    return float(10.0 * np.log10(p_target / p_noise))


def measure_convergence_time(error: np.ndarray, sr: int = 16000, window_ms: int = 20) -> float:
    """
    Measure convergence time (ms): time until error envelope reaches within 3 dB
    of the steady-state average error power.
    """
    win_len = int(sr * window_ms / 1000.0)
    if len(error) < 2 * win_len:
        print(
            f"[!] Convergence time not measurable on {len(error)} samples "
            f"(needs >= {2 * win_len}); recording NaN, not 0.0 ms.",
            file=sys.stderr, flush=True,
        )
        return float("nan")
    
    # Compute moving RMS power
    n_windows = len(error) // win_len
    powers = np.zeros(n_windows)
    for i in range(n_windows):
        chunk = error[i * win_len : (i + 1) * win_len]
        powers[i] = np.mean(chunk ** 2)
    
    steady_state_power = np.mean(powers[-max(1, n_windows // 4):]) + 1e-12
    threshold = steady_state_power * 2.0  # +3 dB above steady state
    
    for i in range(n_windows):
        if powers[i] <= threshold:
            return float(i * window_ms)
    
    return float(n_windows * window_ms)


def run_vss_nlms_sweep(
    output_csv: str = "results/csv/vss_nlms_param_sweep.csv",
    sr: int = 16000,
    duration_s: float = 0.5,
) -> List[Dict[str, Any]]:
    """Executes the full parameter sweep and writes results to CSV."""
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # 1. Prepare representative speech signal
    clean_path = Path("data/v4/clean/SPK_001_clean.wav")
    if clean_path.exists():
        wav, _ = sf.read(str(clean_path))
        clean_speech = wav[:n_samples].astype(np.float32)
        if len(clean_speech) < n_samples:
            clean_speech = np.pad(clean_speech, (0, n_samples - len(clean_speech)))
    else:
        clean_speech = (
            0.4 * np.sin(2 * np.pi * 300 * t) * np.sin(2 * np.pi * 5 * t) +
            0.3 * np.sin(2 * np.pi * 1200 * t) * np.cos(2 * np.pi * 3 * t) +
            0.2 * np.sin(2 * np.pi * 2500 * t)
        ).astype(np.float32)

    # Scale speech to nominal -26 dBFS (~0.05 RMS)
    speech_rms = np.sqrt(np.mean(clean_speech ** 2)) + 1e-12
    clean_speech = clean_speech * (0.05 / speech_rms)

    # 2. Prepare synthetic correlated military noise (engine hum + pink noise)
    np.random.seed(42)
    pink_filter = np.array([0.0499, 0.0905, 0.0805, 0.0632, 0.0469, 0.0336, 0.0240])
    raw_noise = np.random.randn(n_samples + 32).astype(np.float32)
    pink = np.convolve(raw_noise, pink_filter, mode="same")[:n_samples]
    engine_hum = (0.2 * np.sin(2 * np.pi * 120 * t) + 0.15 * np.sin(2 * np.pi * 240 * t)).astype(np.float32)
    base_noise = (pink + engine_hum).astype(np.float32)
    noise_rms = np.sqrt(np.mean(base_noise ** 2)) + 1e-12
    # Set to 0 dB input SNR
    base_noise = base_noise * (0.05 / noise_rms)

    # Acoustic transmission channel for primary mic: FIR lowpass transfer function
    h_primary = np.array([0.8, 0.4, 0.2, 0.1, -0.05], dtype=np.float32)
    h_primary /= np.sum(np.abs(h_primary))
    primary_noise = np.convolve(base_noise, h_primary, mode="same")
    primary_mic = clean_speech + primary_noise

    in_snr = calculate_snr(clean_speech, primary_mic)
    in_sisdr = calculate_si_sdr(clean_speech, primary_mic)

    # Sweep parameter grids
    mu_max_list = [0.01, 0.03, 0.05, 0.10, 0.20]
    filter_length_list = [32, 64, 128, 256]
    delay_list = [0, 1, 2, 4, 8, 16]

    results = []
    total_runs = len(mu_max_list) * len(filter_length_list) * len(delay_list)
    print(f"[GATE 2] Starting VSS-NLMS Parameter Sweep: {total_runs} combinations...", flush=True)
    print(f"Input SNR: {in_snr:.2f} dB, Input SI-SDR: {in_sisdr:.2f} dB\n", flush=True)

    run_idx = 0
    for mu_max in mu_max_list:
        for flen in filter_length_list:
            for delay in delay_list:
                run_idx += 1
                # Reference signal with delay
                if delay > 0:
                    ref_mic = np.pad(base_noise, (delay, 0))[:n_samples]
                else:
                    ref_mic = base_noise.copy()

                # Instantiate VSS-NLMS
                nlms = VSSNLMSFilter(
                    filter_length=flen,
                    mu_init=min(0.05, mu_max),
                    mu_min=0.001,
                    mu_max=mu_max,
                    eps=1e-6
                )

                t0 = time.perf_counter()
                e_out, y_est, mu_hist = nlms.filter_block(primary_mic, ref_mic)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                # Check numerical instability / coefficient explosion
                has_nan = bool(np.isnan(e_out).any() or np.isnan(nlms.weights).any())
                has_inf = bool(np.isinf(e_out).any() or np.isinf(nlms.weights).any())
                max_coeff = float(np.max(np.abs(nlms.weights)))
                exploded = bool(max_coeff >= 4.95 or has_nan or has_inf)

                # Compute Metrics
                out_snr = calculate_snr(clean_speech, e_out)
                delta_snr = out_snr - in_snr

                out_sisdr = calculate_si_sdr(clean_speech, e_out)
                delta_sisdr = out_sisdr - in_sisdr

                try:
                    stoi_score = float(pystoi.stoi(clean_speech, e_out, sr, extended=False))
                except Exception as ex:
                    print(f"[!] STOI failed (run {run_idx}): {ex}; recording NaN, not 0.0.", file=sys.stderr, flush=True)
                    stoi_score = float("nan")

                # Speech distortion: residual vs clean speech during speech intervals
                p_dist = np.mean((clean_speech - e_out) ** 2) + 1e-12
                p_clean = np.mean(clean_speech ** 2) + 1e-12
                speech_distortion_db = float(10.0 * np.log10(p_dist / p_clean))

                # Convergence time on noise-only
                conv_time_ms = measure_convergence_time(e_out - clean_speech, sr=sr)

                # Residual noise power in steady-state (last 25% of audio)
                residual_noise_power = float(np.mean((e_out[int(0.75 * n_samples):] - clean_speech[int(0.75 * n_samples):]) ** 2))

                record = {
                    "run_id": run_idx,
                    "mu_max": mu_max,
                    "filter_length": flen,
                    "delay_samples": delay,
                    "delta_snr_db": round(delta_snr, 3),
                    "delta_sisdr_db": round(delta_sisdr, 3),
                    "stoi": round(stoi_score, 4),
                    "speech_distortion_db": round(speech_distortion_db, 2),
                    "convergence_time_ms": round(conv_time_ms, 1),
                    "residual_noise_power": round(residual_noise_power, 6),
                    "max_coefficient": round(max_coeff, 4),
                    "is_stable": not exploded,
                    "compute_time_ms": round(elapsed_ms, 2),
                }
                results.append(record)

                if run_idx % 20 == 0 or run_idx == total_runs:
                    print(f"  [Progress] Evaluated {run_idx}/{total_runs} configs...", flush=True)

    # 3. Identify Pareto Frontier
    # Criteria: maximize delta_snr_db, minimize convergence_time_ms, minimize speech_distortion_db
    for i, r1 in enumerate(results):
        dominated = False
        if not r1["is_stable"]:
            r1["is_pareto"] = False
            continue

        for j, r2 in enumerate(results):
            if i != j and r2["is_stable"]:
                # r2 dominates r1 if it is better or equal in all 3, and strictly better in at least one
                better_snr = r2["delta_snr_db"] >= r1["delta_snr_db"]
                better_conv = r2["convergence_time_ms"] <= r1["convergence_time_ms"]
                better_dist = r2["speech_distortion_db"] <= r1["speech_distortion_db"]

                strictly_better = (
                    r2["delta_snr_db"] > r1["delta_snr_db"] or
                    r2["convergence_time_ms"] < r1["convergence_time_ms"] or
                    r2["speech_distortion_db"] < r1["speech_distortion_db"]
                )

                if better_snr and better_conv and better_dist and strictly_better:
                    dominated = True
                    break
        r1["is_pareto"] = not dominated

    # 4. Write to CSV
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[GATE 2] Parameter sweep complete! Saved to {out_path}")
    pareto_configs = [r for r in results if r["is_pareto"]]
    print(f"Total Configurations: {len(results)}")
    print(f"Pareto Optimal Configurations: {len(pareto_configs)}")

    print("\n--- TOP PARETO REGION CANDIDATES ---")
    sorted_pareto = sorted(pareto_configs, key=lambda x: x["delta_snr_db"], reverse=True)
    for p in sorted_pareto[:5]:
        print(
            f"mu_max={p['mu_max']:.2f}, L={p['filter_length']:3d}, tau={p['delay_samples']:2d} samples | "
            f"Delta_SNR: +{p['delta_snr_db']:.2f} dB, Delta_SISDR: +{p['delta_sisdr_db']:.2f} dB, "
            f"STOI: {p['stoi']:.3f}, Conv: {p['convergence_time_ms']} ms"
        )

    return results


if __name__ == "__main__":
    run_vss_nlms_sweep()
