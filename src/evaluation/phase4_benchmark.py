"""
Phase 4: AI Model Benchmarking & Pareto Analysis.
PS 26052 — Adaptive Defence ANC.

Compares three AI backends within the hybrid pipeline:
  1. TinyEnhancer V3  (4-layer Conv2D mask, ~9.6K params)
  2. DTLN             (Dual-stage LSTM, ~90K params)
  3. CRN-Micro        (Conv-Recurrent U-Net, ~55K params)

Also benchmarks:
  - Noise Regime Detection accuracy on the tactical noise taxonomy
  - Hybrid pipeline quality: DSP-only vs DSP+AI for each backend

Output:
  results/csv/phase4_model_benchmark.csv
  results/csv/phase4_regime_detection.csv
  results/csv/phase4_pareto_summary.csv
"""

import os
import sys
import csv
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

import soundfile as sf

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dsp.spectral_subtraction import SpectralSubtraction
from src.dsp.wiener import WienerFilter
from src.dsp.nlms import NLMSFilter
from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.noise_regime_detector import NoiseRegimeDetector, NoiseRegime
from src.ai.tiny_enhancer import TinyEnhancerWrapper
from src.ai.dtln import DTLNWrapper
from src.ai.crn import CRNWrapper
from src.pipeline.hybrid_chain import HybridEnhancementPipeline
from src.evaluation.metrics import evaluate_all_metrics


RESULTS_FIELDS = [
    "sample_id", "split", "speaker_id", "noise_category", "noise_source",
    "system_config", "ai_backend", "input_snr_db",
    "output_snr_db", "delta_snr_db", "si_sdr_db", "stoi", "pesq",
    "processing_time_ms", "rtf", "model_params", "ram_mb",
]

REGIME_FIELDS = [
    "sample_id", "noise_category", "noise_source",
    "predicted_regime", "kurtosis", "spectral_flatness", "energy_db",
    "regime_distribution",
]

PARETO_FIELDS = [
    "ai_backend", "system_config", "mean_stoi", "mean_pesq",
    "mean_si_sdr_db", "mean_delta_snr_db",
    "mean_rtf", "p95_latency_ms", "model_params",
    "pareto_optimal",
]


def build_hybrid(ai_backend, use_vss: bool = True, protected: bool = True, sr: int = 16000):
    """Factory: build a HybridEnhancementPipeline with the given AI backend."""
    return HybridEnhancementPipeline(
        config_mode="A",
        filter_length=64,
        step_size=0.08,
        use_vss=use_vss,
        enable_leakage_protection=protected,
        enable_impulse_protection=protected,
        ai_backend=ai_backend,
        sample_rate=sr,
    )


def count_params(wrapper) -> int:
    """Extract parameter count from a model wrapper."""
    if hasattr(wrapper, 'net') and wrapper.net is not None and hasattr(wrapper.net, 'count_parameters'):
        return wrapper.net.count_parameters()
    elif hasattr(wrapper, 'net') and wrapper.net is not None and hasattr(wrapper.net, 'parameters'):
        import torch
        return sum(p.numel() for p in wrapper.net.parameters())
    return 0


def run_phase4_benchmark(
    dataset_metadata_csv: str = "data/v4/metadata/metadata_v4.csv",
    output_dir: str = "results/csv",
    samples_per_split: int = 3,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Execute Phase 4 model benchmark with Pareto analysis."""

    meta_path = Path(dataset_metadata_csv)
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = meta_path.parent.parent

    # Read records
    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split not in records_by_split:
                records_by_split[split] = []
            records_by_split[split].append(row)

    eval_records = []
    for split in sorted(records_by_split.keys()):
        rows = records_by_split[split]
        eval_records.extend(rows[:samples_per_split])

    print("\n" + "=" * 80, flush=True)
    print("  PHASE 4: AI MODEL BENCHMARK & PARETO ANALYSIS", flush=True)
    print("  PS 26052 — Adaptive Defence ANC", flush=True)
    print("=" * 80, flush=True)
    print(f"[*] Evaluation clips: {len(eval_records)} across {len(records_by_split)} splits", flush=True)

    # ────────────────────────────────────────────────────────────
    # Initialize AI backends (loading trained checkpoints if available)
    # ────────────────────────────────────────────────────────────
    tiny_ckpt = "checkpoints/tiny_enhancer_v3.pt" if Path("checkpoints/tiny_enhancer_v3.pt").exists() else None
    crn_ckpt = "checkpoints/crn_micro.pt" if Path("checkpoints/crn_micro.pt").exists() else None
    if tiny_ckpt:
        print(f"[*] Loaded trained TinyEnhancer checkpoint: {tiny_ckpt}", flush=True)
    if crn_ckpt:
        print(f"[*] Loaded trained CRN-Micro checkpoint: {crn_ckpt}", flush=True)

    # Legacy offline path: models/checkpoints from the phase-4 campaign were
    # trained/evaluated at frame 512 (freq_bins 257). Pass the legacy geometry
    # explicitly; the adapter defaults are the frozen PH1 contract geometry
    # (frame 256 / freq_bins 129). See src/ai/crn.py and src/ai/dtln.py.
    ai_backends = {
        "TinyEnhancer_V3": TinyEnhancerWrapper(checkpoint_path=tiny_ckpt),
        "DTLN": DTLNWrapper(frame_size=512, hop_size=128, hidden_size=128, encoder_size=256),
        "CRN_Micro": CRNWrapper(checkpoint_path=crn_ckpt, freq_bins=257,
                                hidden_size=128, channels=(8, 16, 32, 64, 128)),
    }

    backend_params = {name: count_params(wrapper) for name, wrapper in ai_backends.items()}

    print("\n[*] AI Backend Parameter Counts:", flush=True)
    for name, params in backend_params.items():
        print(f"    {name:20s} -> {params:>8,d} params", flush=True)

    # Build pipelines
    pipelines = {}
    for name, backend in ai_backends.items():
        pipelines[f"HYBRID_PROTECTED_{name}"] = {
            "pipeline": build_hybrid(backend, use_vss=True, protected=True),
            "ai_name": name,
        }
        pipelines[f"HYBRID_VANILLA_{name}"] = {
            "pipeline": build_hybrid(backend, use_vss=False, protected=False),
            "ai_name": name,
        }

    # Also include pure DSP baselines
    spec_sub = SpectralSubtraction(sample_rate=sample_rate)
    wiener = WienerFilter(sample_rate=sample_rate)
    vss_nlms = VSSNLMSFilter(filter_length=64, mu_init=0.08)

    # Noise regime detector
    regime_detector = NoiseRegimeDetector(sample_rate=sample_rate)
    process = psutil.Process() if psutil else None

    # ────────────────────────────────────────────────────────────
    # Open CSV writers
    # ────────────────────────────────────────────────────────────
    bench_csv_path = out_dir / "phase4_model_benchmark.csv"
    regime_csv_path = out_dir / "phase4_regime_detection.csv"

    bench_file = open(bench_csv_path, "w", newline="", encoding="utf-8")
    bench_writer = csv.DictWriter(bench_file, fieldnames=RESULTS_FIELDS)
    bench_writer.writeheader()

    regime_file = open(regime_csv_path, "w", newline="", encoding="utf-8")
    regime_writer = csv.DictWriter(regime_file, fieldnames=REGIME_FIELDS)
    regime_writer.writeheader()

    all_results = []

    print("\n[*] Running benchmark matrix...", flush=True)

    for idx, row in enumerate(eval_records, 1):
        sample_id = row["sample_id"]
        split = row["split"]
        spk = row["speaker_id"]
        cat = row["noise_category"]
        noise_src = row["noise_id"]
        input_snr = float(row["actual_snr_db"])

        clean_file = data_root / row["clean_path"]
        prim_file = data_root / row["primary_path"]
        ref_file = data_root / row["reference_path"]

        clean_audio, _ = sf.read(str(clean_file), dtype="float32")
        prim_audio, _ = sf.read(str(prim_file), dtype="float32")
        ref_audio, _ = sf.read(str(ref_file), dtype="float32")
        duration = len(clean_audio) / sample_rate

        # ── Noise Regime Detection ──
        regime_detector.reset()
        regime, regime_diag = regime_detector.classify_signal(prim_audio)
        regime_writer.writerow({
            "sample_id": sample_id,
            "noise_category": cat,
            "noise_source": noise_src,
            "predicted_regime": regime.value,
            "kurtosis": regime_diag.get("dominant_regime", ""),
            "spectral_flatness": str(regime_diag.get("regime_distribution", {})),
            "energy_db": "",
            "regime_distribution": str(regime_diag.get("regime_distribution", {})),
        })
        regime_file.flush()

        # ── DSP Baselines ──
        dsp_configs = [
            ("CONFIG_0_NOISY", "none", lambda: prim_audio.copy()),
            ("CONFIG_1_SPECTRAL_SUB", "none", lambda: spec_sub.process(prim_audio)),
            ("CONFIG_2_WIENER", "none", lambda: wiener.process(prim_audio)),
            ("CONFIG_4_VSS_NLMS_ONLY", "none", lambda: vss_nlms.filter_block(prim_audio, ref_audio)[0]),
        ]

        for cfg_name, ai_name, proc_fn in dsp_configs:
            t0 = time.perf_counter()
            out_audio = proc_fn()
            t1 = time.perf_counter()
            proc_ms = (t1 - t0) * 1000.0
            rtf = (proc_ms / 1000.0) / duration
            mem = (process.memory_info().rss / (1024 * 1024)) if process else 0

            m = evaluate_all_metrics(clean_audio, out_audio, sample_rate=sample_rate)
            res = {
                "sample_id": sample_id, "split": split, "speaker_id": spk,
                "noise_category": cat, "noise_source": noise_src,
                "system_config": cfg_name, "ai_backend": ai_name,
                "input_snr_db": input_snr,
                "output_snr_db": m["snr_db"],
                "delta_snr_db": round(m["snr_db"] - input_snr, 2),
                "si_sdr_db": m["si_snr_db"], "stoi": m["stoi"], "pesq": m["pesq"],
                "processing_time_ms": round(proc_ms, 2), "rtf": round(rtf, 4),
                "model_params": 0, "ram_mb": round(mem, 1),
            }
            all_results.append(res)
            bench_writer.writerow(res)

        # ── AI+Hybrid Pipeline Configs ──
        for cfg_name, cfg_info in pipelines.items():
            pipeline = cfg_info["pipeline"]
            ai_name = cfg_info["ai_name"]
            pipeline.reset()

            t0 = time.perf_counter()
            out_audio, _ = pipeline.process_signals(prim_audio, ref_audio)
            t1 = time.perf_counter()
            proc_ms = (t1 - t0) * 1000.0
            rtf = (proc_ms / 1000.0) / duration
            mem = (process.memory_info().rss / (1024 * 1024)) if process else 0

            m = evaluate_all_metrics(clean_audio, out_audio, sample_rate=sample_rate)
            res = {
                "sample_id": sample_id, "split": split, "speaker_id": spk,
                "noise_category": cat, "noise_source": noise_src,
                "system_config": cfg_name, "ai_backend": ai_name,
                "input_snr_db": input_snr,
                "output_snr_db": m["snr_db"],
                "delta_snr_db": round(m["snr_db"] - input_snr, 2),
                "si_sdr_db": m["si_snr_db"], "stoi": m["stoi"], "pesq": m["pesq"],
                "processing_time_ms": round(proc_ms, 2), "rtf": round(rtf, 4),
                "model_params": backend_params[ai_name], "ram_mb": round(mem, 1),
            }
            all_results.append(res)
            bench_writer.writerow(res)

        bench_file.flush()
        print(f"    [{idx:02d}/{len(eval_records):02d}] [OK] {sample_id} ({split}, {cat})", flush=True)

    bench_file.close()
    regime_file.close()

    # ────────────────────────────────────────────────────────────
    # Pareto Analysis
    # ────────────────────────────────────────────────────────────
    print("\n[*] Computing Pareto frontier...", flush=True)

    summary_by_config = {}
    for r in all_results:
        key = (r["system_config"], r["ai_backend"])
        if key not in summary_by_config:
            summary_by_config[key] = {
                "delta_snr": [], "si_sdr": [], "stoi": [], "pesq": [],
                "rtf": [], "latency_ms": [], "params": r["model_params"],
            }
        summary_by_config[key]["delta_snr"].append(r["delta_snr_db"])
        summary_by_config[key]["si_sdr"].append(r["si_sdr_db"])
        summary_by_config[key]["stoi"].append(r["stoi"])
        summary_by_config[key]["pesq"].append(r["pesq"])
        summary_by_config[key]["rtf"].append(r["rtf"])
        summary_by_config[key]["latency_ms"].append(r["processing_time_ms"])

    pareto_rows = []
    for (cfg, ai), vals in summary_by_config.items():
        pareto_rows.append({
            "ai_backend": ai,
            "system_config": cfg,
            "mean_stoi": round(float(np.mean(vals["stoi"])), 4),
            "mean_pesq": round(float(np.mean(vals["pesq"])), 2),
            "mean_si_sdr_db": round(float(np.mean(vals["si_sdr"])), 2),
            "mean_delta_snr_db": round(float(np.mean(vals["delta_snr"])), 2),
            "mean_rtf": round(float(np.mean(vals["rtf"])), 4),
            "p95_latency_ms": round(float(np.percentile(vals["latency_ms"], 95)), 1),
            "model_params": vals["params"],
            "pareto_optimal": False,
        })

    # Pareto dominance: point A dominates B if A.stoi >= B.stoi AND A.rtf <= B.rtf
    # and at least one is strictly better
    for i, pi in enumerate(pareto_rows):
        dominated = False
        for j, pj in enumerate(pareto_rows):
            if i == j:
                continue
            if (pj["mean_stoi"] >= pi["mean_stoi"] and pj["mean_rtf"] <= pi["mean_rtf"]
                    and (pj["mean_stoi"] > pi["mean_stoi"] or pj["mean_rtf"] < pi["mean_rtf"])):
                dominated = True
                break
        if not dominated:
            pareto_rows[i]["pareto_optimal"] = True

    pareto_csv_path = out_dir / "phase4_pareto_summary.csv"
    with open(pareto_csv_path, "w", newline="", encoding="utf-8") as pf:
        pw = csv.DictWriter(pf, fieldnames=PARETO_FIELDS)
        pw.writeheader()
        for row in sorted(pareto_rows, key=lambda x: -x["mean_stoi"]):
            pw.writerow(row)

    # Print Summary Table
    print("\n" + "=" * 110, flush=True)
    print(f"{'CONFIG':<38} | {'AI BACKEND':<18} | {'dSNR':>6} | {'SI-SDR':>7} | {'STOI':>6} | {'PESQ':>5} | {'RTF':>6} | {'P95-LAT':>7} | {'PARAMS':>8} | {'P*'}", flush=True)
    print("-" * 110, flush=True)
    for row in sorted(pareto_rows, key=lambda x: -x["mean_stoi"]):
        marker = " *" if row["pareto_optimal"] else ""
        print(
            f"{row['system_config']:<38} | {row['ai_backend']:<18} | "
            f"{row['mean_delta_snr_db']:+5.1f}  | {row['mean_si_sdr_db']:+6.1f}  | "
            f"{row['mean_stoi']:5.3f} | {row['mean_pesq']:4.1f}  | "
            f"{row['mean_rtf']:5.3f} | {row['p95_latency_ms']:5.1f}ms | "
            f"{row['model_params']:>8,d} |{marker}",
            flush=True,
        )
    print("=" * 110, flush=True)
    print(f"\n[+] Phase 4 results saved:", flush=True)
    print(f"    Benchmark:  {bench_csv_path.resolve()}", flush=True)
    print(f"    Regimes:    {regime_csv_path.resolve()}", flush=True)
    print(f"    Pareto:     {pareto_csv_path.resolve()}", flush=True)

    return {
        "status": "COMPLETED",
        "total_configs": len(pareto_rows),
        "total_rows": len(all_results),
        "pareto_optimal": [r for r in pareto_rows if r["pareto_optimal"]],
    }


if __name__ == "__main__":
    run_phase4_benchmark()
