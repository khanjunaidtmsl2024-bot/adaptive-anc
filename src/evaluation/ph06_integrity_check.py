"""
PH0.6 -- Evaluation Integrity Check.
PS 26052 -- Adaptive Defence ANC.

PURPOSE: Explain the ~50 dB SI-SDR discrepancy between earlier PH0/laptop
metrics (SI-SDR ~ +11 dB) and PH0.5 PESQ ablation (SI-SDR ~ -38 dB).

This script runs ONE canonical clip through 4 separate processing paths,
saves every intermediate signal, and computes metrics 3 ways to isolate
the measurement bug.

Evidence Tier: OFFLINE EXPERIMENTALLY MEASURED (DIAGNOSTIC)
"""

import sys
import time
import csv
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.impulse_protection import ImpulseProtectionController
from src.ai.tiny_enhancer import TinyEnhancerWrapper
from src.streaming.causal_engine import CausalStreamingEngine
from src.pipeline.hybrid_chain import HybridEnhancementPipeline

try:
    from pesq import pesq as pesq_eval
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi as stoi_eval
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


SR = 16000
DURATION = 3.0
SNR_DB = 0.0
SEED = 42

# Canonical STFT configs
ABLATION_FRAME = 256
ABLATION_HOP = 128
HYBRID_FRAME = 512
HYBRID_HOP = 256
FILTER_LEN = 64


def generate_canonical_clip(
    seed: int = SEED, duration_s: float = DURATION, snr_db: float = SNR_DB
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single canonical test clip with known seed.

    Returns: (clean, noisy, reference)
    """
    rng = np.random.RandomState(seed)
    n = int(duration_s * SR)
    t = np.arange(n) / SR

    # Multi-harmonic speech-like signal
    f0 = 150.0
    envelope = np.sin(np.pi * t / duration_s) ** 2
    clean = np.zeros(n, dtype=np.float64)
    for h in [1, 2, 3, 4, 5]:
        clean += (0.3 / h) * np.sin(2 * np.pi * f0 * h * t)
    clean = clean * envelope
    clean = clean / (np.max(np.abs(clean)) + 1e-12) * 0.5
    clean = clean.astype(np.float32)

    # Noise: band-limited 200-4000 Hz
    from scipy.signal import butter, filtfilt
    noise_raw = rng.randn(n).astype(np.float64)
    b, a = butter(4, [200 / (SR / 2), 4000 / (SR / 2)], btype="band")
    noise_filt = filtfilt(b, a, noise_raw).astype(np.float32)

    # Scale to target SNR
    clean_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean(noise_filt ** 2))
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scaled = noise_filt * np.sqrt(target_noise_power / (noise_power + 1e-12))

    noisy = (clean + noise_scaled).astype(np.float32)
    reference = (noise_scaled * 0.9 + rng.randn(n).astype(np.float32) * 0.01).astype(np.float32)

    return clean, noisy, reference


# =====================================================================
# Signal Integrity Inspection
# =====================================================================
def signal_stats(sig: np.ndarray, name: str) -> Dict[str, Any]:
    """Compute signal integrity statistics."""
    return {
        "name": name,
        "length": len(sig),
        "dtype": str(sig.dtype),
        "rms": float(np.sqrt(np.mean(sig ** 2))),
        "peak": float(np.max(np.abs(sig))),
        "min": float(np.min(sig)),
        "max": float(np.max(sig)),
        "mean": float(np.mean(sig)),
        "nan_count": int(np.sum(np.isnan(sig))),
        "inf_count": int(np.sum(np.isinf(sig))),
        "clip_count_098": int(np.sum(np.abs(sig) > 0.98)),
        "zero_count": int(np.sum(sig == 0.0)),
        "energy": float(np.sum(sig ** 2)),
    }


# =====================================================================
# Metric Computation (3 modes)
# =====================================================================
def compute_si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Standard SI-SDR with mean removal only."""
    n = min(len(ref), len(est))
    r = ref[:n].astype(np.float64) - np.mean(ref[:n].astype(np.float64))
    e = est[:n].astype(np.float64) - np.mean(est[:n].astype(np.float64))
    s_target = r * np.dot(e, r) / (np.dot(r, r) + 1e-12)
    noise = e - s_target
    return float(10 * np.log10(np.sum(s_target ** 2) / (np.sum(noise ** 2) + 1e-12)))


def compute_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Output SNR."""
    n = min(len(clean), len(enhanced))
    c = clean[:n].astype(np.float64)
    noise = enhanced[:n].astype(np.float64) - c
    return float(10 * np.log10(np.sum(c ** 2) / (np.sum(noise ** 2) + 1e-12)))


def compute_delta_snr(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray) -> float:
    """Delta-SNR = SNR_out - SNR_in."""
    snr_in = compute_snr(clean, noisy)
    snr_out = compute_snr(clean, enhanced)
    return snr_out - snr_in


def compute_metrics_raw(clean: np.ndarray, enhanced: np.ndarray, noisy: np.ndarray) -> Dict[str, float]:
    """Metrics on raw signals (no normalization)."""
    n = min(len(clean), len(enhanced), len(noisy))
    return {
        "si_sdr_db": round(compute_si_sdr(clean[:n], enhanced[:n]), 2),
        "snr_out_db": round(compute_snr(clean[:n], enhanced[:n]), 2),
        "delta_snr_db": round(compute_delta_snr(clean[:n], noisy[:n], enhanced[:n]), 2),
        "stoi": round(float(stoi_eval(clean[:n], enhanced[:n], SR, extended=False)), 4) if STOI_AVAILABLE else float("nan"),
        "pesq_wb": round(float(pesq_eval(SR, clean[:n], enhanced[:n], "wb")), 4) if PESQ_AVAILABLE else float("nan"),
        "norm_mode": "RAW",
    }


def compute_metrics_independent_norm(clean: np.ndarray, enhanced: np.ndarray, noisy: np.ndarray) -> Dict[str, float]:
    """Metrics using the ablation's independent peak-normalization (the suspected bug)."""
    n = min(len(clean), len(enhanced), len(noisy))
    c = clean[:n].copy()
    y = noisy[:n].copy()
    e = enhanced[:n].copy()

    c = c / (np.max(np.abs(c)) + 1e-12) * 0.7
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.7
    e = e / (np.max(np.abs(e)) + 1e-12) * 0.7

    return {
        "si_sdr_db": round(compute_si_sdr(c, e), 2),
        "snr_out_db": round(compute_snr(c, e), 2),
        "delta_snr_db": round(compute_delta_snr(c, y, e), 2),
        "stoi": round(float(stoi_eval(c, e, SR, extended=False)), 4) if STOI_AVAILABLE else float("nan"),
        "pesq_wb": round(float(pesq_eval(SR, c, e, "wb")), 4) if PESQ_AVAILABLE else float("nan"),
        "norm_mode": "INDEPENDENT_PEAK_NORM",
    }


def compute_metrics_joint_norm(clean: np.ndarray, enhanced: np.ndarray, noisy: np.ndarray) -> Dict[str, float]:
    """Metrics using joint peak-normalization (scale all by the same factor)."""
    n = min(len(clean), len(enhanced), len(noisy))
    c = clean[:n].copy()
    y = noisy[:n].copy()
    e = enhanced[:n].copy()

    # Joint normalization: scale by the max across all three signals
    global_peak = max(np.max(np.abs(c)), np.max(np.abs(y)), np.max(np.abs(e))) + 1e-12
    scale = 0.7 / global_peak
    c = c * scale
    y = y * scale
    e = e * scale

    return {
        "si_sdr_db": round(compute_si_sdr(c, e), 2),
        "snr_out_db": round(compute_snr(c, e), 2),
        "delta_snr_db": round(compute_delta_snr(c, y, e), 2),
        "stoi": round(float(stoi_eval(c, e, SR, extended=False)), 4) if STOI_AVAILABLE else float("nan"),
        "pesq_wb": round(float(pesq_eval(SR, c, e, "wb")), 4) if PESQ_AVAILABLE else float("nan"),
        "norm_mode": "JOINT_PEAK_NORM",
    }


# =====================================================================
# Processing Paths
# =====================================================================
def path_a_hybrid_pipeline(noisy: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Path A: HybridEnhancementPipeline (frame=512, hop=256) — earlier evaluation."""
    ai = TinyEnhancerWrapper()
    pipeline = HybridEnhancementPipeline(
        config_mode="A",
        filter_length=FILTER_LEN,
        step_size=0.05,
        use_vss=True,
        enable_leakage_protection=True,
        enable_impulse_protection=True,
        ai_backend=ai,
        sample_rate=SR,
        frame_size=HYBRID_FRAME,
        hop_size=HYBRID_HOP,
    )
    output, diag = pipeline.process_signals(noisy, reference)
    return output, {"path": "A_HybridPipeline", "frame": HYBRID_FRAME, "hop": HYBRID_HOP, "dsp_residual": diag.get("dsp_output", None)}


def path_b_causal_engine(noisy: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Path B: CausalStreamingEngine (frame=256, hop=128) — the deployed streaming engine."""
    engine = CausalStreamingEngine(
        frame_size=ABLATION_FRAME,
        hop_size=ABLATION_HOP,
        sample_rate=SR,
        filter_length=FILTER_LEN,
        step_size=0.05,
        use_fast_dsp=False,  # Use Python DSP for fair comparison
    )
    output, diag = engine.process_signal(noisy, reference)
    return output, {"path": "B_CausalEngine", "frame": ABLATION_FRAME, "hop": ABLATION_HOP}


def path_c_ablation_baseline(noisy: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Path C: Ablation-style processing (reproducing ph05_pesq_ablation.py ABL-0)."""
    n = min(len(noisy), len(reference))
    frame_size = ABLATION_FRAME
    hop_size = ABLATION_HOP

    nlms = VSSNLMSFilter(filter_length=FILTER_LEN, mu_init=0.05)
    imp_ctl = ImpulseProtectionController(sample_rate=SR)
    ai_model = TinyEnhancerWrapper()

    window = np.hanning(frame_size).astype(np.float32)
    buf_primary = np.zeros(frame_size, dtype=np.float32)
    buf_reference = np.zeros(frame_size, dtype=np.float32)
    overlap_buf = np.zeros(frame_size, dtype=np.float32)
    output = np.zeros(n, dtype=np.float32)

    n_hops = (n - frame_size) // hop_size + 1

    # Collect intermediates for diagnostics
    ai_masks = []
    dsp_frame_energies = []
    ai_input_mags = []

    for i in range(n_hops):
        start = i * hop_size
        p_hop = noisy[start:start + hop_size]
        r_hop = reference[start:start + hop_size]

        if len(p_hop) < hop_size:
            p_hop = np.pad(p_hop, (0, hop_size - len(p_hop)))
            r_hop = np.pad(r_hop, (0, hop_size - len(r_hop)))

        buf_primary[:-hop_size] = buf_primary[hop_size:]
        buf_primary[-hop_size:] = p_hop
        buf_reference[:-hop_size] = buf_reference[hop_size:]
        buf_reference[-hop_size:] = r_hop

        dsp_hop, _, _ = nlms.filter_block(p_hop, r_hop)
        dsp_hop, _, _ = imp_ctl.filter_block_protection(dsp_hop)

        # THE BUG: Frame is half-noisy, half-filtered
        dsp_frame = buf_primary.copy()
        dsp_frame[-hop_size:] = dsp_hop

        windowed = dsp_frame * window
        stft_frame = np.fft.rfft(windowed)
        freq_bins = len(stft_frame)
        mag = np.abs(stft_frame).reshape(-1, 1)
        phase_orig = np.angle(stft_frame).reshape(-1, 1)

        enh_mag, enh_phase = ai_model.enhance_spectrogram(mag, phase_orig)
        enh_mag_1d = enh_mag.flatten()[:freq_bins]
        enh_phase_1d = enh_phase.flatten()[:freq_bins]
        enh_stft = enh_mag_1d * np.exp(1j * enh_phase_1d)
        recon = np.fft.irfft(enh_stft, n=frame_size) * window

        overlap_buf += recon
        out_hop = overlap_buf[:hop_size].copy()
        overlap_buf[:-hop_size] = overlap_buf[hop_size:]
        overlap_buf[-hop_size:] = 0.0

        output[start:start + hop_size] = out_hop

        # Save mask stats
        mask_vals = (enh_mag.flatten()[:freq_bins] / (mag.flatten()[:freq_bins] + 1e-12))
        ai_masks.append(float(np.mean(mask_vals)))
        dsp_frame_energies.append(float(np.sum(dsp_frame ** 2)))
        ai_input_mags.append(float(np.mean(mag.flatten())))

    peak = np.max(np.abs(output)) + 1e-12
    if peak > 0.98:
        output = output * 0.98 / peak

    diag = {
        "path": "C_AblationBaseline",
        "frame": frame_size,
        "hop": hop_size,
        "avg_mask": float(np.mean(ai_masks)) if ai_masks else 0.0,
        "avg_dsp_frame_energy": float(np.mean(dsp_frame_energies)) if dsp_frame_energies else 0.0,
        "avg_ai_input_mag": float(np.mean(ai_input_mags)) if ai_input_mags else 0.0,
    }
    return output, diag


def path_d_nlms_only(noisy: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Path D: NLMS-only (no AI stage)."""
    nlms = VSSNLMSFilter(filter_length=FILTER_LEN, mu_init=0.05)
    imp_ctl = ImpulseProtectionController(sample_rate=SR)
    residual, _, _ = nlms.filter_block(noisy, reference)
    residual, _, _ = imp_ctl.filter_block_protection(residual)
    return residual, {"path": "D_NLMS_Only"}


# =====================================================================
# Main Execution
# =====================================================================
def run_integrity_check():
    """Execute the PH0.6 evaluation integrity check."""
    print("=" * 76)
    print("PH0.6 -- EVALUATION INTEGRITY CHECK")
    print("Diagnosing ~50 dB SI-SDR discrepancy between PH0 and PH0.5 ablation")
    print("Evidence Tier: OFFLINE EXPERIMENTALLY MEASURED (DIAGNOSTIC)")
    print("=" * 76)

    out_dir = Path("results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    sig_dir = Path("results/ph06_intermediates")
    sig_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate canonical clip
    print("\n[1/4] Generating canonical clip (seed=%d, duration=%.1fs, SNR=%.0f dB)..." % (SEED, DURATION, SNR_DB))
    clean, noisy, reference = generate_canonical_clip()

    for name, sig in [("clean", clean), ("noisy", noisy), ("reference", reference)]:
        np.save(str(sig_dir / f"{name}.npy"), sig)
        stats = signal_stats(sig, name)
        print(f"  {name}: RMS={stats['rms']:.6f}  peak={stats['peak']:.6f}  "
              f"len={stats['length']}  NaN={stats['nan_count']}  Inf={stats['inf_count']}")

    # Step 2: Process through all 4 paths
    print("\n[2/4] Processing through 4 evaluation paths...")
    paths = [
        ("A_HybridPipeline", path_a_hybrid_pipeline),
        ("B_CausalEngine", path_b_causal_engine),
        ("C_AblationBaseline", path_c_ablation_baseline),
        ("D_NLMS_Only", path_d_nlms_only),
    ]

    path_outputs = {}
    path_diags = {}

    for path_name, path_fn in paths:
        t0 = time.perf_counter()
        output, diag = path_fn(noisy.copy(), reference.copy())
        elapsed = (time.perf_counter() - t0) * 1000
        path_outputs[path_name] = output
        path_diags[path_name] = diag

        np.save(str(sig_dir / f"{path_name}_output.npy"), output)
        stats = signal_stats(output, path_name)
        print(f"  {path_name}: RMS={stats['rms']:.6f}  peak={stats['peak']:.6f}  "
              f"len={stats['length']}  clip098={stats['clip_count_098']}  "
              f"time={elapsed:.1f}ms")
        if "avg_mask" in diag:
            print(f"    AI mask avg={diag['avg_mask']:.4f}  "
                  f"DSP frame energy avg={diag['avg_dsp_frame_energy']:.6f}  "
                  f"AI input mag avg={diag['avg_ai_input_mag']:.6f}")

    # Step 3: Compute metrics 3 ways
    print("\n[3/4] Computing metrics (3 normalization modes x 4 paths = 12 evaluations)...")

    all_results = []
    metric_fns = [
        ("RAW", compute_metrics_raw),
        ("INDEP_NORM", compute_metrics_independent_norm),
        ("JOINT_NORM", compute_metrics_joint_norm),
    ]

    for path_name, output in path_outputs.items():
        for norm_name, metric_fn in metric_fns:
            try:
                m = metric_fn(clean, output, noisy)
                m["path"] = path_name
                all_results.append(m)
            except Exception as ex:
                print(f"  WARNING: {path_name}/{norm_name} metric failed: {ex}")
                all_results.append({
                    "path": path_name,
                    "norm_mode": norm_name,
                    "si_sdr_db": float("nan"),
                    "snr_out_db": float("nan"),
                    "delta_snr_db": float("nan"),
                    "stoi": float("nan"),
                    "pesq_wb": float("nan"),
                })

    # Save CSV
    csv_path = out_dir / "ph06_integrity_check.csv"
    fieldnames = ["path", "norm_mode", "si_sdr_db", "snr_out_db", "delta_snr_db", "stoi", "pesq_wb"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"  Saved: {csv_path}")

    # Step 4: Print diagnostic comparison table
    print("\n[4/4] DISCREPANCY DIAGNOSIS TABLE")
    print("=" * 100)
    print(f"{'Path':<22} | {'Norm Mode':<18} | {'SI-SDR':>8} | {'SNR_out':>8} | {'Delta-SNR':>10} | {'STOI':>7} | {'PESQ':>6}")
    print("-" * 100)

    for r in all_results:
        si = r.get("si_sdr_db", float("nan"))
        snr = r.get("snr_out_db", float("nan"))
        dsnr = r.get("delta_snr_db", float("nan"))
        stoi_v = r.get("stoi", float("nan"))
        pesq_v = r.get("pesq_wb", float("nan"))
        si_s = f"{si:+8.2f}" if not np.isnan(si) else "     NaN"
        snr_s = f"{snr:+8.2f}" if not np.isnan(snr) else "     NaN"
        dsnr_s = f"{dsnr:+10.2f}" if not np.isnan(dsnr) else "       NaN"
        stoi_s = f"{stoi_v:7.4f}" if not np.isnan(stoi_v) else "    NaN"
        pesq_s = f"{pesq_v:6.3f}" if not np.isnan(pesq_v) else "   NaN"
        print(f"{r['path']:<22} | {r['norm_mode']:<18} | {si_s} | {snr_s} | {dsnr_s} | {stoi_s} | {pesq_s}")

    print("-" * 100)

    # Diagnosis
    print("\n" + "=" * 76)
    print("DISCREPANCY ROOT CAUSE ANALYSIS")
    print("=" * 76)

    # Compare Path A RAW vs Path C INDEP_NORM
    path_a_raw = [r for r in all_results if r["path"] == "A_HybridPipeline" and r["norm_mode"] == "RAW"]
    path_c_indep = [r for r in all_results if r["path"] == "C_AblationBaseline" and r["norm_mode"] == "INDEP_NORM"]
    path_c_raw = [r for r in all_results if r["path"] == "C_AblationBaseline" and r["norm_mode"] == "RAW"]
    path_b_raw = [r for r in all_results if r["path"] == "B_CausalEngine" and r["norm_mode"] == "RAW"]

    if path_a_raw and path_c_indep:
        a_si = path_a_raw[0]["si_sdr_db"]
        c_indep_si = path_c_indep[0]["si_sdr_db"]
        print(f"\n  1. Path A (HybridPipeline, RAW):      SI-SDR = {a_si:+.2f} dB")
        print(f"  2. Path C (Ablation, INDEP_NORM):     SI-SDR = {c_indep_si:+.2f} dB")
        print(f"     DELTA = {c_indep_si - a_si:+.2f} dB")

    if path_c_raw and path_c_indep:
        c_raw_si = path_c_raw[0]["si_sdr_db"]
        c_indep_si = path_c_indep[0]["si_sdr_db"]
        print(f"\n  3. NORMALIZATION BUG IMPACT:")
        print(f"     Path C RAW SI-SDR:        {c_raw_si:+.2f} dB")
        print(f"     Path C INDEP_NORM SI-SDR: {c_indep_si:+.2f} dB")
        norm_delta = c_indep_si - c_raw_si
        print(f"     Normalization bug delta:   {norm_delta:+.2f} dB")
        if abs(norm_delta) > 5.0:
            print(f"     >> SIGNIFICANT: Independent normalization changes SI-SDR by {abs(norm_delta):.1f} dB")
        else:
            print(f"     >> MINOR: Normalization contributes only {abs(norm_delta):.1f} dB")

    if path_a_raw and path_c_raw:
        a_raw_si = path_a_raw[0]["si_sdr_db"]
        c_raw_si = path_c_raw[0]["si_sdr_db"]
        pipeline_delta = c_raw_si - a_raw_si
        print(f"\n  4. PROCESSING PATH IMPACT (both RAW):")
        print(f"     Path A (frame=512, full-signal NLMS): SI-SDR = {a_raw_si:+.2f} dB")
        print(f"     Path C (frame=256, hop-by-hop NLMS):  SI-SDR = {c_raw_si:+.2f} dB")
        print(f"     Processing path delta:                {pipeline_delta:+.2f} dB")
        if abs(pipeline_delta) > 5.0:
            print(f"     >> SIGNIFICANT: Frame size / NLMS granularity / DSP frame construction changes SI-SDR by {abs(pipeline_delta):.1f} dB")

    if path_b_raw and path_c_raw:
        b_raw_si = path_b_raw[0]["si_sdr_db"]
        c_raw_si = path_c_raw[0]["si_sdr_db"]
        engine_delta = b_raw_si - c_raw_si
        print(f"\n  5. ENGINE vs ABLATION (both frame=256, RAW):")
        print(f"     Path B (CausalEngine):     SI-SDR = {b_raw_si:+.2f} dB")
        print(f"     Path C (Ablation baseline): SI-SDR = {c_raw_si:+.2f} dB")
        print(f"     Engine implementation delta: {engine_delta:+.2f} dB")
        if abs(engine_delta) > 3.0:
            print(f"     >> SIGNIFICANT: DSP frame construction differs between CausalEngine and ablation")

    path_d_raw = [r for r in all_results if r["path"] == "D_NLMS_Only" and r["norm_mode"] == "RAW"]
    if path_d_raw:
        d_raw_si = path_d_raw[0]["si_sdr_db"]
        print(f"\n  6. NLMS-ONLY SANITY CHECK (RAW):")
        print(f"     Path D (NLMS-only, no AI): SI-SDR = {d_raw_si:+.2f} dB")
        print(f"     >> If positive, classical DSP works correctly. AI is the divergent factor.")

    print("\n" + "=" * 76)
    print("END PH0.6 INTEGRITY CHECK")
    print("=" * 76)

    return all_results


if __name__ == "__main__":
    run_integrity_check()
