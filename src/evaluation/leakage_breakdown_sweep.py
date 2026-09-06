"""
Gate 3: Reference-Microphone Speech Leakage Attack and Breakdown Sweep.
PS 26052 — Adaptive Defence ANC.

Attacks the acoustic vulnerability:
    x[n] = n_r[n] + alpha * s[n]

Sweeps:
    alpha in [0.00, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

Compares 3 Architectures:
    1. Unprotected VSS-NLMS (No speech leakage guard)
    2. Protected VSS-NLMS (With SpeechLeakageDetector dynamic gating)
    3. Full Hybrid Chain (Protected VSS-NLMS + AI Stage 2)

Outputs:
    results/csv/leakage_breakdown_sweep.csv
    Identifies the exact critical threshold alpha_crit where speech cancellation
    becomes intolerable (speech distortion > 3.0 dB or STOI drop > 0.10).
"""

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import soundfile as sf
import pystoi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.leakage_detector import SpeechLeakageDetector
from src.pipeline.hybrid_chain import HybridEnhancementPipeline


def calculate_snr(clean: np.ndarray, noisy_or_est: np.ndarray) -> float:
    noise = noisy_or_est - clean
    p_clean = np.mean(clean ** 2) + 1e-12
    p_noise = np.mean(noise ** 2) + 1e-12
    return float(10.0 * np.log10(p_clean / p_noise))


def calculate_si_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    ref = reference - np.mean(reference)
    est = estimated - np.mean(estimated)
    dot = np.sum(est * ref)
    s_target = (dot / (np.sum(ref ** 2) + 1e-12)) * ref
    e_noise = est - s_target
    p_target = np.sum(s_target ** 2) + 1e-12
    p_noise = np.sum(e_noise ** 2) + 1e-12
    return float(10.0 * np.log10(p_target / p_noise))


def run_leakage_breakdown_sweep(
    output_csv: str = "results/csv/leakage_breakdown_sweep.csv",
    sr: int = 16000,
    duration_s: float = 0.5,
) -> List[Dict[str, Any]]:
    """Runs the reference-mic leakage attack sweep."""
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # 1. Load clean speech
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

    speech_rms = np.sqrt(np.mean(clean_speech ** 2)) + 1e-12
    clean_speech = clean_speech * (0.05 / speech_rms)

    # 2. Correlated noise
    np.random.seed(42)
    pink_filter = np.array([0.0499, 0.0905, 0.0805, 0.0632, 0.0469, 0.0336, 0.0240])
    raw_noise = np.random.randn(n_samples + 32).astype(np.float32)
    pink = np.convolve(raw_noise, pink_filter, mode="same")[:n_samples]
    engine_hum = (0.2 * np.sin(2 * np.pi * 120 * t) + 0.15 * np.sin(2 * np.pi * 240 * t)).astype(np.float32)
    base_noise = (pink + engine_hum).astype(np.float32)
    noise_rms = np.sqrt(np.mean(base_noise ** 2)) + 1e-12
    base_noise = base_noise * (0.05 / noise_rms)  # 0 dB SNR

    # Primary mic: speech + correlated noise
    h_primary = np.array([0.8, 0.4, 0.2, 0.1, -0.05], dtype=np.float32)
    h_primary /= np.sum(np.abs(h_primary))
    primary_noise = np.convolve(base_noise, h_primary, mode="same")
    primary_mic = clean_speech + primary_noise

    in_snr = calculate_snr(clean_speech, primary_mic)
    in_sisdr = calculate_si_sdr(clean_speech, primary_mic)
    in_stoi = float(pystoi.stoi(clean_speech, primary_mic, sr, extended=False))

    alpha_values = [0.00, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    modes = ["Unprotected_NLMS", "Protected_VSS_NLMS", "Hybrid_Chain"]

    print(f"[GATE 3] Reference-Mic Leakage Breakdown Sweep ({len(alpha_values)} alphas x 3 modes)...", flush=True)
    print(f"Input SNR: {in_snr:.2f} dB, SI-SDR: {in_sisdr:.2f} dB, STOI: {in_stoi:.4f}\n", flush=True)

    results = []
    run_id = 0

    for alpha in alpha_values:
        # Reference microphone with deliberate alpha speech bleed
        ref_mic = (base_noise + alpha * clean_speech).astype(np.float32)

        for mode in modes:
            run_id += 1

            if mode == "Unprotected_NLMS":
                # Standard VSS-NLMS without leakage detector
                nlms = VSSNLMSFilter(filter_length=64, mu_init=0.05, mu_max=0.05)
                e_out, _, _ = nlms.filter_block(primary_mic, ref_mic)
                g_leak_avg = 1.0

            elif mode == "Protected_VSS_NLMS":
                # VSS-NLMS with SpeechLeakageDetector
                leak_det = SpeechLeakageDetector(sample_rate=sr)
                g_leak, _ = leak_det.compute_frame_gating(primary_mic, ref_mic)
                g_leak_avg = float(g_leak)

                # Gated step-size
                nlms = VSSNLMSFilter(
                    filter_length=64,
                    mu_init=0.05 * g_leak,
                    mu_max=0.05 * g_leak
                )
                e_out, _, _ = nlms.filter_block(primary_mic, ref_mic)

            elif mode == "Hybrid_Chain":
                # Full 2-stage hybrid pipeline with leakage detector and AI
                pipeline = HybridEnhancementPipeline(
                    filter_length=64,
                    step_size=0.05,
                    use_vss=True,
                    enable_leakage_protection=True,
                    enable_impulse_protection=True,
                    sample_rate=sr,
                    frame_size=512,
                    hop_size=128
                )
                e_out, diag = pipeline.process_signals(primary_mic, ref_mic)
                # Cut padding
                e_out = e_out[:n_samples]
                g_leak_avg = float(diag.get("leakage_gating_factor", 1.0))

            # Metrics
            out_snr = calculate_snr(clean_speech, e_out)
            delta_snr = out_snr - in_snr
            out_sisdr = calculate_si_sdr(clean_speech, e_out)
            delta_sisdr = out_sisdr - in_sisdr

            try:
                out_stoi = float(pystoi.stoi(clean_speech, e_out, sr, extended=False))
            except Exception as ex:
                print(f"[!] STOI failed (alpha={alpha}, mode={mode}): {ex}; recording NaN, not 0.0.",
                      file=sys.stderr, flush=True)
                out_stoi = float("nan")

            # Speech Distortion: degradation relative to clean speech
            p_clean = np.mean(clean_speech ** 2) + 1e-12
            p_err = np.mean((clean_speech - e_out) ** 2) + 1e-12
            speech_distortion_db = float(10.0 * np.log10(p_err / p_clean))

            # Speech Self-Cancellation Flag: STOI degradation > 0.10 from nominal alpha=0.
            # A NaN out_stoi (failed measurement) must NOT silently read as
            # "not degraded" -- Python NaN comparisons are False, which would
            # corrupt the alpha_crit "robust" conclusion downstream.
            stoi_failed = bool(np.isnan(out_stoi))
            stoi_drop = (in_stoi - out_stoi) if not stoi_failed else 1.0
            record = {
                "run_id": run_id,
                "alpha_leakage": alpha,
                "mode": mode,
                "g_leak": round(g_leak_avg, 3),
                "delta_snr_db": round(delta_snr, 3),
                "delta_sisdr_db": round(delta_sisdr, 3),
                "stoi": round(out_stoi, 4),
                "stoi_error": stoi_failed,
                "speech_distortion_db": round(speech_distortion_db, 2),
                "is_degraded": bool(stoi_failed or speech_distortion_db > 3.0 or stoi_drop > 0.10),
            }
            results.append(record)

        print(f"  alpha={alpha:.2f} evaluated across all 3 modes", flush=True)

    # Write to CSV
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[GATE 3] Sweep complete! Saved to {out_path}", flush=True)

    # Identify critical breakdown threshold alpha_crit for each mode
    print("\n--- CRITICAL LEAKAGE BREAKDOWN THRESHOLDS (alpha_crit) ---")
    for mode in modes:
        mode_records = [r for r in results if r["mode"] == mode]
        alpha_crit = None
        for r in mode_records:
            if r["is_degraded"]:
                alpha_crit = r["alpha_leakage"]
                break
        if alpha_crit is not None:
            print(f"  {mode:<24}: alpha_crit = {alpha_crit:.2f} (Degradation exceeded tolerance)")
        else:
            print(f"  {mode:<24}: Robust across all tested alphas up to {alpha_values[-1]:.2f}")

    return results


if __name__ == "__main__":
    run_leakage_breakdown_sweep()
