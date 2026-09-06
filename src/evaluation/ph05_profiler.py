"""
PH0.5 — Isolated & Integrated Performance Profiler.
PS 26052 — Adaptive Defence ANC.

Benchmarks each DSP/AI stage INDEPENDENTLY to determine per-component
compute cost, then benchmarks the FULL INTEGRATED pipeline to get the
true per-hop compute time (which includes all overheads).

Evidence Tier: All measurements labeled OFFLINE EXPERIMENTALLY MEASURED
(measurements taken on host development PC, not target hardware).

Output CSVs:
  - results/csv/ph05_isolated_profile.csv
  - results/csv/ph05_integrated_profile.csv
  - results/csv/ph05_speedup_summary.csv
"""

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.impulse_protection import ImpulseProtectionController
from src.dsp.noise_regime_detector import NoiseRegimeDetector
from src.ai.tiny_enhancer import TinyEnhancerWrapper

# Try import fast backends
try:
    from src.dsp.vss_nlms_fast import VSSNLMSFilterFast, NUMBA_AVAILABLE as NLMS_NUMBA
except ImportError:
    VSSNLMSFilterFast = None
    NLMS_NUMBA = False

try:
    from src.dsp.impulse_protection_fast import ImpulseProtectionControllerFast, NUMBA_AVAILABLE as IMP_NUMBA
except ImportError:
    ImpulseProtectionControllerFast = None
    IMP_NUMBA = False


# Constants
SR = 16000
FRAME_SIZE = 256    # CausalStreamingEngine frame size
HOP_SIZE = 128      # Hop size (8 ms budget)
FILTER_LENGTH = 64
N_WARMUP = 50
N_BENCH_ISOLATED = 500
N_BENCH_INTEGRATED = 350
BUDGET_MS = HOP_SIZE / SR * 1000.0  # 8.000 ms


def _stats(times_us: List[float]) -> Dict[str, float]:
    """Compute timing statistics from microsecond measurements."""
    arr_ms = np.array(times_us) / 1000.0
    return {
        "mean_ms": round(float(np.mean(arr_ms)), 4),
        "p50_ms": round(float(np.percentile(arr_ms, 50)), 4),
        "p95_ms": round(float(np.percentile(arr_ms, 95)), 4),
        "p99_ms": round(float(np.percentile(arr_ms, 99)), 4),
        "max_ms": round(float(np.max(arr_ms)), 4),
        "std_ms": round(float(np.std(arr_ms)), 4),
    }


def bench_nlms_python(rng: np.random.RandomState) -> Dict[str, float]:
    """ISO-1: Benchmark Python VSS-NLMS on hop_size samples."""
    nlms = VSSNLMSFilter(filter_length=FILTER_LENGTH, mu_init=0.05)

    # Warmup
    for _ in range(N_WARMUP):
        p = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        r = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        nlms.filter_block(p, r)

    nlms.reset()
    times = []
    for _ in range(N_BENCH_ISOLATED):
        p = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        r = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        t0 = time.perf_counter()
        nlms.filter_block(p, r)
        times.append((time.perf_counter() - t0) * 1e6)

    return _stats(times)


def bench_nlms_numba(rng: np.random.RandomState) -> Dict[str, float]:
    """ISO-2: Benchmark Numba VSS-NLMS on hop_size samples."""
    if VSSNLMSFilterFast is None or not NLMS_NUMBA:
        return {"mean_ms": -1, "p50_ms": -1, "p95_ms": -1, "p99_ms": -1, "max_ms": -1, "std_ms": -1}

    nlms = VSSNLMSFilterFast(filter_length=FILTER_LENGTH, mu_init=0.05)

    # Warmup (includes JIT compilation)
    for _ in range(N_WARMUP + 5):
        p = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        r = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        nlms.filter_block(p, r)

    nlms.reset()
    times = []
    for _ in range(N_BENCH_ISOLATED):
        p = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        r = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        t0 = time.perf_counter()
        nlms.filter_block(p, r)
        times.append((time.perf_counter() - t0) * 1e6)

    return _stats(times)


def bench_impulse_python(rng: np.random.RandomState) -> Dict[str, float]:
    """ISO-3: Benchmark Python Impulse Controller."""
    ctl = ImpulseProtectionController(sample_rate=SR)

    for _ in range(N_WARMUP):
        sig = rng.randn(HOP_SIZE).astype(np.float32) * 0.05
        ctl.filter_block_protection(sig)

    ctl.reset()
    times = []
    for _ in range(N_BENCH_ISOLATED):
        sig = rng.randn(HOP_SIZE).astype(np.float32) * 0.05
        t0 = time.perf_counter()
        ctl.filter_block_protection(sig)
        times.append((time.perf_counter() - t0) * 1e6)

    return _stats(times)


def bench_impulse_numba(rng: np.random.RandomState) -> Dict[str, float]:
    """ISO-4: Benchmark Numba Impulse Controller."""
    if ImpulseProtectionControllerFast is None or not IMP_NUMBA:
        return {"mean_ms": -1, "p50_ms": -1, "p95_ms": -1, "p99_ms": -1, "max_ms": -1, "std_ms": -1}

    ctl = ImpulseProtectionControllerFast(sample_rate=SR)

    for _ in range(N_WARMUP + 5):
        sig = rng.randn(HOP_SIZE).astype(np.float32) * 0.05
        ctl.filter_block_protection(sig)

    ctl.reset()
    times = []
    for _ in range(N_BENCH_ISOLATED):
        sig = rng.randn(HOP_SIZE).astype(np.float32) * 0.05
        t0 = time.perf_counter()
        ctl.filter_block_protection(sig)
        times.append((time.perf_counter() - t0) * 1e6)

    return _stats(times)


def bench_regime_detector(rng: np.random.RandomState) -> Dict[str, float]:
    """ISO-5: Benchmark Noise Regime Detector."""
    det = NoiseRegimeDetector(sample_rate=SR, frame_size=FRAME_SIZE)

    for _ in range(N_WARMUP):
        frame = rng.randn(FRAME_SIZE).astype(np.float32) * 0.1
        det.classify_frame(frame)

    det.reset()
    times = []
    for _ in range(N_BENCH_ISOLATED):
        frame = rng.randn(FRAME_SIZE).astype(np.float32) * 0.1
        t0 = time.perf_counter()
        det.classify_frame(frame)
        times.append((time.perf_counter() - t0) * 1e6)

    return _stats(times)


def bench_stft_analysis(rng: np.random.RandomState) -> Dict[str, float]:
    """ISO-6: Benchmark STFT windowing + rfft."""
    window = np.hanning(FRAME_SIZE).astype(np.float32)

    for _ in range(N_WARMUP):
        frame = rng.randn(FRAME_SIZE).astype(np.float32) * 0.1
        _ = np.fft.rfft(frame * window)

    times = []
    for _ in range(N_BENCH_ISOLATED):
        frame = rng.randn(FRAME_SIZE).astype(np.float32) * 0.1
        t0 = time.perf_counter()
        windowed = frame * window
        stft = np.fft.rfft(windowed)
        mag = np.abs(stft).reshape(-1, 1)
        phase = np.angle(stft).reshape(-1, 1)
        times.append((time.perf_counter() - t0) * 1e6)

    return _stats(times)


def bench_ai_forward(rng: np.random.RandomState) -> Dict[str, float]:
    """ISO-7: Benchmark TinyEnhancer forward pass."""
    ai = TinyEnhancerWrapper()
    freq_bins = FRAME_SIZE // 2 + 1  # 129 for frame_size=256

    for _ in range(N_WARMUP):
        mag = np.abs(rng.randn(freq_bins, 1).astype(np.float32)) * 0.1
        phase = rng.randn(freq_bins, 1).astype(np.float32)
        ai.enhance_spectrogram(mag, phase)

    times = []
    for _ in range(N_BENCH_ISOLATED):
        mag = np.abs(rng.randn(freq_bins, 1).astype(np.float32)) * 0.1
        phase = rng.randn(freq_bins, 1).astype(np.float32)
        t0 = time.perf_counter()
        ai.enhance_spectrogram(mag, phase)
        times.append((time.perf_counter() - t0) * 1e6)

    return _stats(times)


def bench_istft_synthesis(rng: np.random.RandomState) -> Dict[str, float]:
    """ISO-8: Benchmark iSTFT + windowing."""
    freq_bins = FRAME_SIZE // 2 + 1
    window = np.hanning(FRAME_SIZE).astype(np.float32)

    for _ in range(N_WARMUP):
        enh_stft = rng.randn(freq_bins).astype(np.complex64)
        _ = np.fft.irfft(enh_stft, n=FRAME_SIZE)

    times = []
    for _ in range(N_BENCH_ISOLATED):
        mag = np.abs(rng.randn(freq_bins).astype(np.float32)) * 0.1
        phase = rng.randn(freq_bins).astype(np.float32)
        t0 = time.perf_counter()
        enh_stft = mag * np.exp(1j * phase)
        recon = np.fft.irfft(enh_stft, n=FRAME_SIZE) * window
        times.append((time.perf_counter() - t0) * 1e6)

    return _stats(times)


def bench_integrated_pipeline(rng: np.random.RandomState, use_numba: bool) -> Dict[str, Any]:
    """INT-1/INT-2: Benchmark the FULL integrated pipeline per hop using CausalStreamingEngine."""
    from src.streaming.causal_engine import CausalStreamingEngine

    engine = CausalStreamingEngine(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        sample_rate=SR,
        filter_length=FILTER_LENGTH,
        use_fast_dsp=use_numba,
    )

    # Full warm-up: run N_WARMUP complete hops to ensure JIT compilation,
    # PyTorch cache allocation, and OS thread pool warming are finished.
    for _ in range(N_WARMUP):
        p = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        r = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        engine.process_hop(p, r)

    # Discard warmup completely and reset state
    engine.reset()

    times_total = []
    times_dsp = []
    times_ai = []
    times_regime = []

    # Run steady-state benchmark for at least 300 hops
    for _ in range(N_BENCH_INTEGRATED):
        p_chunk = rng.randn(HOP_SIZE).astype(np.float32) * 0.1
        r_chunk = rng.randn(HOP_SIZE).astype(np.float32) * 0.1

        t0 = time.perf_counter()
        _, diag = engine.process_hop(p_chunk, r_chunk)
        t_total = (time.perf_counter() - t0) * 1e6

        times_total.append(t_total)
        times_dsp.append(diag["time_dsp_ms"] * 1000.0)
        times_ai.append(diag["time_ai_ms"] * 1000.0)
        times_regime.append(diag["time_regime_ms"] * 1000.0)

    total_stats = _stats(times_total)
    dsp_stats = _stats(times_dsp)
    ai_stats = _stats(times_ai)
    regime_stats = _stats(times_regime)

    return {
        "total": total_stats,
        "dsp": dsp_stats,
        "ai": ai_stats,
        "regime": regime_stats,
        "budget_pass": total_stats["p95_ms"] <= BUDGET_MS,
    }


def run_ph05_profiling():
    """Execute the full PH0.5 profiling suite."""
    rng = np.random.RandomState(42)
    out_dir = Path("results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PH0.5 — ISOLATED & INTEGRATED PERFORMANCE PROFILER")
    print(f"Hop budget: {BUDGET_MS:.3f} ms ({HOP_SIZE} samples @ {SR} Hz)")
    print("Evidence Tier: OFFLINE EXPERIMENTALLY MEASURED (host PC)")
    print("=" * 72)

    # ─── Isolated Benchmarks ───
    print("\n[PHASE 1] ISOLATED COMPONENT BENCHMARKS")
    print("-" * 60)

    iso_results = {}

    print("  ISO-1: VSS-NLMS (Python)...", end=" ", flush=True)
    iso_results["ISO-1 NLMS Python"] = bench_nlms_python(np.random.RandomState(42))
    print(f"P50={iso_results['ISO-1 NLMS Python']['p50_ms']:.4f} ms")

    print("  ISO-2: VSS-NLMS (Numba)...", end=" ", flush=True)
    iso_results["ISO-2 NLMS Numba"] = bench_nlms_numba(np.random.RandomState(42))
    print(f"P50={iso_results['ISO-2 NLMS Numba']['p50_ms']:.4f} ms")

    print("  ISO-3: Impulse Controller (Python)...", end=" ", flush=True)
    iso_results["ISO-3 Impulse Python"] = bench_impulse_python(np.random.RandomState(42))
    print(f"P50={iso_results['ISO-3 Impulse Python']['p50_ms']:.4f} ms")

    print("  ISO-4: Impulse Controller (Numba)...", end=" ", flush=True)
    iso_results["ISO-4 Impulse Numba"] = bench_impulse_numba(np.random.RandomState(42))
    print(f"P50={iso_results['ISO-4 Impulse Numba']['p50_ms']:.4f} ms")

    print("  ISO-5: Regime Detector...", end=" ", flush=True)
    iso_results["ISO-5 Regime Detector"] = bench_regime_detector(np.random.RandomState(42))
    print(f"P50={iso_results['ISO-5 Regime Detector']['p50_ms']:.4f} ms")

    print("  ISO-6: STFT Analysis...", end=" ", flush=True)
    iso_results["ISO-6 STFT Analysis"] = bench_stft_analysis(np.random.RandomState(42))
    print(f"P50={iso_results['ISO-6 STFT Analysis']['p50_ms']:.4f} ms")

    print("  ISO-7: AI TinyEnhancer...", end=" ", flush=True)
    iso_results["ISO-7 AI Forward"] = bench_ai_forward(np.random.RandomState(42))
    print(f"P50={iso_results['ISO-7 AI Forward']['p50_ms']:.4f} ms")

    print("  ISO-8: iSTFT Synthesis...", end=" ", flush=True)
    iso_results["ISO-8 iSTFT Synthesis"] = bench_istft_synthesis(np.random.RandomState(42))
    print(f"P50={iso_results['ISO-8 iSTFT Synthesis']['p50_ms']:.4f} ms")

    # Save isolated CSV
    iso_csv = out_dir / "ph05_isolated_profile.csv"
    with open(iso_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark_id", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "std_ms", "evidence_tier"])
        for name, stats in iso_results.items():
            writer.writerow([
                name,
                stats["mean_ms"], stats["p50_ms"], stats["p95_ms"],
                stats["p99_ms"], stats["max_ms"], stats["std_ms"],
                "OFFLINE EXPERIMENTALLY MEASURED (host PC)"
            ])
    print(f"\n  Saved: {iso_csv}")

    # ─── Speedup Summary ───
    print("\n[PHASE 2] SPEEDUP ANALYSIS")
    print("-" * 60)

    speedup_csv = out_dir / "ph05_speedup_summary.csv"
    with open(speedup_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "python_p50_ms", "numba_p50_ms", "speedup_factor", "evidence_tier"])

        nlms_py = iso_results["ISO-1 NLMS Python"]["p50_ms"]
        nlms_nb = iso_results["ISO-2 NLMS Numba"]["p50_ms"]
        nlms_speedup = nlms_py / nlms_nb if nlms_nb > 0 else float("nan")
        writer.writerow(["VSS-NLMS", nlms_py, nlms_nb, round(nlms_speedup, 2),
                         "OFFLINE EXPERIMENTALLY MEASURED (host PC)"])
        print(f"  NLMS: Python={nlms_py:.4f} ms -> Numba={nlms_nb:.4f} ms ({nlms_speedup:.1f}x speedup)")

        imp_py = iso_results["ISO-3 Impulse Python"]["p50_ms"]
        imp_nb = iso_results["ISO-4 Impulse Numba"]["p50_ms"]
        imp_speedup = imp_py / imp_nb if imp_nb > 0 else float("nan")
        writer.writerow(["Impulse Controller", imp_py, imp_nb, round(imp_speedup, 2),
                         "OFFLINE EXPERIMENTALLY MEASURED (host PC)"])
        print(f"  Impulse: Python={imp_py:.4f} ms -> Numba={imp_nb:.4f} ms ({imp_speedup:.1f}x speedup)")

    print(f"\n  Saved: {speedup_csv}")

    # ─── Integrated Pipeline ───
    print("\n[PHASE 3] INTEGRATED PIPELINE BENCHMARK")
    print("-" * 60)

    print("  INT-1: Full Pipeline (Python backends)...", flush=True)
    int_py = bench_integrated_pipeline(np.random.RandomState(42), use_numba=False)
    print(f"    Total P50={int_py['total']['p50_ms']:.4f} ms | P95={int_py['total']['p95_ms']:.4f} ms | "
          f"Budget={'PASS' if int_py['budget_pass'] else 'FAIL'}")

    print("  INT-2: Full Pipeline (Numba backends)...", flush=True)
    int_nb = bench_integrated_pipeline(np.random.RandomState(42), use_numba=True)
    print(f"    Total P50={int_nb['total']['p50_ms']:.4f} ms | P95={int_nb['total']['p95_ms']:.4f} ms | "
          f"Budget={'PASS' if int_nb['budget_pass'] else 'FAIL'}")

    int_speedup = int_py["total"]["p50_ms"] / int_nb["total"]["p50_ms"] if int_nb["total"]["p50_ms"] > 0 else float("nan")
    print(f"\n  Integrated speedup: {int_speedup:.2f}x")

    # Save integrated CSV
    int_csv = out_dir / "ph05_integrated_profile.csv"
    with open(int_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "stage", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms",
                         "budget_8ms_pass", "evidence_tier"])

        for config_name, result in [("Python", int_py), ("Numba", int_nb)]:
            for stage_name in ["total", "dsp", "ai", "regime"]:
                s = result[stage_name]
                writer.writerow([
                    config_name, stage_name,
                    s["mean_ms"], s["p50_ms"], s["p95_ms"], s["p99_ms"], s["max_ms"],
                    result["budget_pass"] if stage_name == "total" else "",
                    "OFFLINE EXPERIMENTALLY MEASURED (host PC)"
                ])

    print(f"  Saved: {int_csv}")

    # ─── Final Summary ───
    print("\n" + "=" * 72)
    print("PH0.5 PROFILING COMPLETE")
    print("=" * 72)
    print(f"\n{'Component':<30} | {'Python P50':>12} | {'Numba P50':>12} | {'Speedup':>8}")
    print("-" * 72)
    print(f"{'VSS-NLMS (isolated)':<30} | {nlms_py:>10.4f} ms | {nlms_nb:>10.4f} ms | {nlms_speedup:>6.1f}x")
    print(f"{'Impulse Ctl (isolated)':<30} | {imp_py:>10.4f} ms | {imp_nb:>10.4f} ms | {imp_speedup:>6.1f}x")
    print(f"{'Full Pipeline (integrated)':<30} | {int_py['total']['p50_ms']:>10.4f} ms | {int_nb['total']['p50_ms']:>10.4f} ms | {int_speedup:>6.1f}x")
    print("-" * 72)
    print(f"8 ms Hop Budget (Python): {'PASS [OK]' if int_py['budget_pass'] else 'FAIL [X]'}")
    print(f"8 ms Hop Budget (Numba):  {'PASS [OK]' if int_nb['budget_pass'] else 'FAIL [X]'}")

    return {
        "isolated": iso_results,
        "integrated_python": int_py,
        "integrated_numba": int_nb,
        "nlms_speedup": nlms_speedup,
        "impulse_speedup": imp_speedup,
        "integrated_speedup": int_speedup,
    }


if __name__ == "__main__":
    run_ph05_profiling()
