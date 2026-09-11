"""
Cross-Repo Merge: Generic Noise Grid Evaluation.
PS 26052 -- Adaptive Defence ANC.

Adopts the evaluation protocol of the collaborator baseline repository
(https://github.com/ichigo137/anc, scripts/generate_dataset.py + evaluate.py):

  - Noise types : hum, pink, white, impulsive
  - SNR grid    : -5, 0, 5, 10, 15, 20 dB
  - Reporting   : noise-type x SNR cross-breakdown of SNR / SI-SDR / STOI / PESQ

but runs it through OUR authoritative deployment chain instead of the
collaborator's offline batch model:

  - VSS-NLMS adaptive cancellation (primary vs delayed reference)
  - E2_causal ONNX spectral-mask enhancement
  - CausalStreamingEngine (256/128, center=False, strictly causal, hop-by-hop)

Three variants are evaluated per cell so the reader can see each stage's
contribution, mirroring the collaborator's raw-vs-AI-vs-hybrid framing:

  - NOISY      : unprocessed primary mic
  - NLMS_ONLY  : VSS-NLMS residual only (no AI)
  - FULL       : VSS-NLMS + E2_causal ONNX (the deployed System A path)

Noise synthesis recipes are ported 1:1 from the collaborator repo's
generate_dataset.py so numbers are comparable to its published outputs.

Usage:
    python scripts/evaluate_generic_noise_grid.py
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.streaming.causal_engine import CausalStreamingEngine
from src.evaluation.metrics import (
    evaluate_all_metrics,
    get_metric_provenance,
    compute_snr,
    compute_si_snr,
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SR = 16000
NOISE_TYPES = ["hum", "pink", "white", "impulsive"]
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]
REFERENCE_DELAY_SAMPLES = 4          # NLMS needs a correlated reference; delay only, no gain
VARIANTS = ["NOISY", "NLMS_ONLY", "FULL"]

CLEAN_FILES = [
    "data/v4/clean/SPK_009_clean.wav",   # TEST_A_UNSEEN_SPEAKER
    "data/v4/clean/SPK_010_clean.wav",   # TEST_A_UNSEEN_SPEAKER
]
OUT_DIR = Path("results/generic_noise_grid")


# ------------------------------------------------------------------
# Noise synthesis (ported 1:1 from ichigo137/anc generate_dataset.py)
# ------------------------------------------------------------------
def generate_synthetic_noise(length: int, noise_type: str) -> np.ndarray:
    """Same recipes as the collaborator repo: hum/pink/white/impulsive."""
    if noise_type == "white":
        noise = np.random.normal(0, 1, length)
    elif noise_type == "pink":
        white = np.random.normal(0, 1, length)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1
        spectrum /= np.sqrt(freqs)
        noise = np.fft.irfft(spectrum, n=length)
    elif noise_type == "hum":
        t = np.arange(length) / SR
        noise = (
            0.7 * np.sin(2 * np.pi * 50 * t)
            + 0.3 * np.sin(2 * np.pi * 100 * t)
            + 0.15 * np.sin(2 * np.pi * 150 * t)
        )
    elif noise_type == "impulsive":
        noise = np.random.normal(0, 0.03, length)
        number_of_impulses = max(1, length // (SR // 2))
        positions = np.random.randint(0, length, number_of_impulses)
        for position in positions:
            width = min(80, length - position)
            noise[position:position + width] += (
                np.random.uniform(0.5, 1.0) * np.exp(-np.arange(width) / 15)
            )
    else:
        noise = np.random.normal(0, 1, length)

    noise = noise.astype(np.float32)
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise /= peak
    return noise


def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise to the requested SNR, same as collaborator add_noise()."""
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(clean)]

    clean_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    desired_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(desired_noise_power / noise_power)
    noisy = clean + noise * scale

    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy = noisy / peak * 0.99
    return noisy.astype(np.float32)


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------
def run_variant(primary: np.ndarray, reference: np.ndarray, variant: str) -> np.ndarray:
    if variant == "NOISY":
        return primary.astype(np.float32)

    nlms_only = variant == "NLMS_ONLY"

    if nlms_only:
        # Replicate the DSP stage of CausalStreamingEngine without the AI stage.
        from src.dsp.vss_nlms import VSSNLMSFilter
        filt = VSSNLMSFilter(filter_length=64, mu_init=0.05, mu_max=0.05)
        residual, _, _ = filt.filter_block(primary, reference)
        return np.clip(np.asarray(residual, dtype=np.float32), -0.98, 0.98)

    engine = CausalStreamingEngine(
        frame_size=256,
        hop_size=128,
        sample_rate=SR,
        filter_length=64,
        step_size=0.05,
        ai_backend="onnx",          # E2_causal.onnx -- deployed System A path
        enable_regime_adaptation=True,
        use_fast_dsp=True,
    )
    enhanced, _ = engine.process_signal(primary, reference)
    return enhanced.astype(np.float32)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260909)
    np.random.seed(20260909)

    print("=" * 72)
    print("CROSS-REPO MERGE: GENERIC NOISE GRID (ichigo137/anc protocol)")
    print("Engine: VSS-NLMS + E2_causal ONNX (CausalStreamingEngine, 256/128 causal)")
    print(f"Noise types: {NOISE_TYPES} | SNRs: {SNR_LEVELS} dB")
    print(f"Clean clips: {[Path(f).name for f in CLEAN_FILES]}")
    print("=" * 72)

    rows: List[Dict] = []
    t_start = time.perf_counter()

    for clean_path in CLEAN_FILES:
        clean, _ = sf.read(clean_path, dtype="float32")
        clean = clean[: SR * 3]  # 3 s clips

        for noise_type in NOISE_TYPES:
            for snr in SNR_LEVELS:
                # Deterministic per-cell seed so runs are reproducible
                cell_seed = int(hash((Path(clean_path).stem, noise_type, snr)) % (2**31))
                np.random.seed(cell_seed)
                noise = generate_synthetic_noise(len(clean), noise_type)
                primary = add_noise(clean, noise, snr)

                # NLMS reference: delayed copy of the noise (no gain mismatch)
                reference = np.zeros_like(noise)
                reference[REFERENCE_DELAY_SAMPLES:] = noise[:-REFERENCE_DELAY_SAMPLES]

                for variant in VARIANTS:
                    enhanced = run_variant(primary, reference, variant)
                    m = evaluate_all_metrics(clean, enhanced, sample_rate=SR, strict=True)

                    # Noise-type rows report means over the 3-s clip
                    row = {
                        "clip": Path(clean_path).stem,
                        "noise_type": noise_type,
                        "target_snr_db": snr,
                        "variant": variant,
                        "output_snr_db": m["snr_db"],
                        "delta_snr_db": round(m["snr_db"] - compute_snr(clean, primary), 2),
                        "si_sdr_db": m["si_snr_db"],
                        "stoi": m["stoi"],
                        "pesq": m["pesq"],
                        "stoi_source": m["stoi_source"],
                        "pesq_source": m["pesq_source"],
                    }
                    rows.append(row)
                    print(
                        f"{Path(clean_path).stem:9s} {noise_type:9s} snr{snr:>+3d} "
                        f"{variant:9s} | SNR {m['snr_db']:+.2f} dB | "
                        f"SI-SDR {m['si_snr_db']:+.2f} | STOI {m['stoi']:.3f} | PESQ {m['pesq']:.2f}",
                        flush=True,
                    )

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {elapsed:.1f}s for {len(rows)} evaluations")

    # ------------------------------------------------------------
    # Persist per-cell matrix + summary
    # ------------------------------------------------------------
    csv_path = OUT_DIR / "evaluation_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def cell_mean(variant: str, noise_type: str, snr: int, key: str) -> float:
        vals = [r[key] for r in rows
                if r["variant"] == variant and r["noise_type"] == noise_type
                and r["target_snr_db"] == snr]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "protocol_source": "https://github.com/ichigo137/anc (generate_dataset.py + evaluate.py)",
        "engine": "CausalStreamingEngine VSS-NLMS + E2_causal.onnx (256/128, center=False, strict causal)",
        "clean_clips": CLEAN_FILES,
        "reference_delay_samples": REFERENCE_DELAY_SAMPLES,
        "noise_types": NOISE_TYPES,
        "snr_levels": SNR_LEVELS,
        "variants": VARIANTS,
        "metric_provenance": get_metric_provenance(),
        "noise_type_breakdown": {},
        "snr_breakdown": {},
        "cells_evaluated": len(rows),
        "total_time_s": round(elapsed, 1),
    }

    # Noise-type x variant breakdown (mean over SNR levels and clips)
    for noise_type in NOISE_TYPES:
        summary["noise_type_breakdown"][noise_type] = {}
        for variant in VARIANTS:
            subset = [r for r in rows if r["noise_type"] == noise_type and r["variant"] == variant]
            summary["noise_type_breakdown"][noise_type][variant] = {
                "mean_delta_snr_db": round(float(np.mean([r["delta_snr_db"] for r in subset])), 2) if subset else None,
                "mean_si_sdr_db": round(float(np.mean([r["si_sdr_db"] for r in subset])), 2) if subset else None,
                "mean_stoi": round(float(np.mean([r["stoi"] for r in subset])), 3) if subset else None,
                "mean_pesq": round(float(np.mean([r["pesq"] for r in subset])), 2) if subset else None,
            }

    # SNR-level x variant breakdown (mean over noise types and clips)
    for snr in SNR_LEVELS:
        summary["snr_breakdown"][f"{snr:+d}"] = {}
        for variant in VARIANTS:
            subset = [r for r in rows if r["target_snr_db"] == snr and r["variant"] == variant]
            summary["snr_breakdown"][f"{snr:+d}"][variant] = {
                "mean_delta_snr_db": round(float(np.mean([r["delta_snr_db"] for r in subset])), 2) if subset else None,
                "mean_si_sdr_db": round(float(np.mean([r["si_sdr_db"] for r in subset])), 2) if subset else None,
                "mean_stoi": round(float(np.mean([r["stoi"] for r in subset])), 3) if subset else None,
                "mean_pesq": round(float(np.mean([r["pesq"] for r in subset])), 2) if subset else None,
            }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------
    # Markdown report with the external repo's two breakdown tables
    # ------------------------------------------------------------
    lines = []
    lines.append("# Generic Noise Grid Evaluation (Cross-Repo Merge)")
    lines.append("")
    lines.append(f"Protocol: **{summary['protocol_source']}**")
    lines.append("")
    lines.append(f"Engine: **{summary['engine']}**")
    lines.append("")
    lines.append(f"Clean clips (unseen speakers): {', '.join(CLEAN_FILES)}")
    lines.append(f"Grid: {len(NOISE_TYPES)} noise types x {len(SNR_LEVELS)} SNRs x {len(VARIANTS)} variants = {len(rows)} evaluations")
    lines.append("")
    lines.append("## Noise Type Breakdown (mean over SNR levels)")
    lines.append("")
    lines.append("| Noise | Variant | SNR Δ (dB) | SI-SDR (dB) | STOI | PESQ |")
    lines.append("|---|---|---|---|---|---|")
    for noise_type in NOISE_TYPES:
        for variant in VARIANTS:
            b = summary["noise_type_breakdown"][noise_type][variant]
            lines.append(
                f"| {noise_type} | {variant} | {b['mean_delta_snr_db']:+.2f} | "
                f"{b['mean_si_sdr_db']:+.2f} | {b['mean_stoi']:.3f} | {b['mean_pesq']:.2f} |"
            )
    lines.append("")
    lines.append("## SNR Level Breakdown (mean over noise types)")
    lines.append("")
    lines.append("| Target SNR (dB) | Variant | SNR Δ (dB) | SI-SDR (dB) | STOI | PESQ |")
    lines.append("|---|---|---|---|---|---|")
    for snr in SNR_LEVELS:
        for variant in VARIANTS:
            b = summary["snr_breakdown"][f"{snr:+d}"][variant]
            lines.append(
                f"| {snr:+d} | {variant} | {b['mean_delta_snr_db']:+.2f} | "
                f"{b['mean_si_sdr_db']:+.2f} | {b['mean_stoi']:.3f} | {b['mean_pesq']:.2f} |"
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Metrics: {summary['metric_provenance']}")
    lines.append("- FULL = deployed System A path (VSS-NLMS + E2_causal ONNX, strictly causal, 8 ms hop).")
    lines.append("- Impulsive noise exercises the regime detector's IMPULSIVE mode and impulse protection.")
    lines.append("")
    with open(OUT_DIR / "GENERIC_GRID_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {OUT_DIR / 'summary.json'}")
    print(f"Saved: {OUT_DIR / 'GENERIC_GRID_REPORT.md'}")


if __name__ == "__main__":
    main()