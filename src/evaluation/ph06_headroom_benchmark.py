"""
PH0.6 -- Extended Headroom Benchmark.
PS 26052 -- Adaptive Defence ANC.

PURPOSE: Rigorous latency characterization with 1000 steady-state hops
to validate the tight 8.00 ms pass (PH0.5 P95 = 7.41 ms, headroom = 0.59 ms).

Measures:
  - P50/P95/P99/P99.9/Max/Mean/StdDev over 1000 steady-state hops
  - Worst contiguous latency streak (consecutive hops > threshold)
  - CPU utilization ratio (process time / wall time)
  - Per-hop timing breakdown (DSP / AI / regime / overhead)

Evidence Tier: OFFLINE EXPERIMENTALLY MEASURED (host PC)
Output: results/csv/ph06_headroom_profile.csv
"""

import sys
import time
import csv
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.streaming.causal_engine import CausalStreamingEngine

# Backend provenance: prove which DSP backend actually ran (audit item 1).
# If numba is unavailable or the fast modules import fails, the engine silently
# falls back to the pure-Python reference path and timings are NOT numba ones.
try:
    from src.dsp.vss_nlms_fast import NUMBA_AVAILABLE as NLMS_NUMBA_AVAIL
except ImportError:
    NLMS_NUMBA_AVAIL = False
try:
    from src.dsp.impulse_protection_fast import NUMBA_AVAILABLE as IMP_NUMBA_AVAIL
except ImportError:
    IMP_NUMBA_AVAIL = False
try:
    import numpy as _np
    NUMPY_VERSION = _np.__version__
except Exception:
    NUMPY_VERSION = "unknown"
try:
    import numba as _nb
    NUMBA_VERSION = _nb.__version__
except Exception:
    NUMBA_VERSION = "unavailable"

SR = 16000
FRAME_SIZE = 256
HOP_SIZE = 128
FILTER_LEN = 64
WARMUP_HOPS = 100
STEADY_HOPS = 1000
BUDGET_MS = 8.0
STREAK_THRESHOLD_MS = 7.0  # Track consecutive hops above this


def generate_benchmark_signal(duration_s: float = 10.0, seed: int = 42):
    """Generate a long synthetic signal for benchmarking."""
    rng = np.random.RandomState(seed)
    n = int(duration_s * SR)
    t = np.arange(n) / SR

    # Multi-harmonic clean signal
    f0 = 150.0
    envelope = np.sin(np.pi * t / duration_s) ** 2
    clean = np.zeros(n, dtype=np.float32)
    for h in [1, 2, 3, 4, 5]:
        clean += (0.3 / h) * np.sin(2 * np.pi * f0 * h * t).astype(np.float32)
    clean = (clean * envelope).astype(np.float32)
    clean = clean / (np.max(np.abs(clean)) + 1e-12) * 0.5

    noise = rng.randn(n).astype(np.float32) * 0.3
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [200 / (SR / 2), 4000 / (SR / 2)], btype="band")
    noise = filtfilt(b, a, noise).astype(np.float32)

    noisy = (clean + noise).astype(np.float32)
    reference = (noise * 0.9 + rng.randn(n).astype(np.float32) * 0.01).astype(np.float32)

    return noisy, reference


def run_headroom_benchmark():
    """Execute the PH0.6 extended headroom benchmark."""
    print("=" * 76)
    print("PH0.6 -- EXTENDED HEADROOM BENCHMARK")
    print(f"Warm-up: {WARMUP_HOPS} hops (discarded)  |  Steady-state: {STEADY_HOPS} hops")
    print(f"Budget: {BUDGET_MS:.3f} ms  |  Streak threshold: {STREAK_THRESHOLD_MS:.1f} ms")
    print("Evidence Tier: OFFLINE EXPERIMENTALLY MEASURED (host PC)")
    print("=" * 76)

    out_dir = Path("results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_hops_needed = WARMUP_HOPS + STEADY_HOPS + 10
    min_duration = (total_hops_needed * HOP_SIZE + FRAME_SIZE) / SR + 1.0
    print(f"\n[1/3] Generating benchmark signal ({min_duration:.1f}s)...")
    noisy, reference = generate_benchmark_signal(duration_s=min_duration)

    # Numba-compiled engine
    print("[2/3] Running benchmark...")
    print(f"       Backend probe: vss_nlms NUMBA_AVAILABLE={NLMS_NUMBA_AVAIL}, "
          f"impulse NUMBA_AVAILABLE={IMP_NUMBA_AVAIL}, numba={NUMBA_VERSION}, "
          f"numpy={NUMPY_VERSION}")
    engine = CausalStreamingEngine(
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        sample_rate=SR,
        filter_length=FILTER_LEN,
        step_size=0.05,
        use_fast_dsp=True,
    )
    engine.reset()

    # Record the backend class actually instantiated by the engine
    engine_nlms_cls = type(engine.nlms).__name__
    engine_imp_cls = type(engine.impulse_controller).__name__
    nlms_backend = "numba" if engine_nlms_cls.endswith("Fast") and NLMS_NUMBA_AVAIL else "python_fallback"
    imp_backend = "numba" if engine_imp_cls.endswith("Fast") and IMP_NUMBA_AVAIL else "python_fallback"
    print(f"       Engine backend: nlms={nlms_backend} ({engine_nlms_cls}), "
          f"impulse={imp_backend} ({engine_imp_cls})")

    n = min(len(noisy), len(reference))
    n_hops = (n - FRAME_SIZE) // HOP_SIZE + 1

    if n_hops < WARMUP_HOPS + STEADY_HOPS:
        print(f"  ERROR: Only {n_hops} hops available, need {WARMUP_HOPS + STEADY_HOPS}")
        return

    # Warm-up phase
    print(f"  Warm-up ({WARMUP_HOPS} hops)...", end="", flush=True)
    for i in range(WARMUP_HOPS):
        start = i * HOP_SIZE
        p_hop = noisy[start:start + HOP_SIZE]
        r_hop = reference[start:start + HOP_SIZE]
        _, _ = engine.process_hop(p_hop, r_hop)
    print(" done.")

    # Steady-state measurement
    print(f"  Steady-state ({STEADY_HOPS} hops)...", end="", flush=True)
    hop_times = []
    dsp_times = []
    ai_times = []
    regime_times = []

    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    for i in range(STEADY_HOPS):
        idx = WARMUP_HOPS + i
        start = idx * HOP_SIZE
        p_hop = noisy[start:start + HOP_SIZE]
        r_hop = reference[start:start + HOP_SIZE]

        _, diag = engine.process_hop(p_hop, r_hop)
        hop_times.append(diag["time_total_ms"])
        dsp_times.append(diag["time_dsp_ms"])
        ai_times.append(diag["time_ai_ms"])
        regime_times.append(diag["time_regime_ms"])

    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    print(" done.")

    hop_arr = np.array(hop_times)
    dsp_arr = np.array(dsp_times)
    ai_arr = np.array(ai_times)
    regime_arr = np.array(regime_times)

    # Compute statistics
    stats = {
        # Backend provenance (audit item 1): these prove the measured timings
        # are genuinely Numba-backed, not Python-fallback timings.
        "numba_version": NUMBA_VERSION,
        "numpy_version": NUMPY_VERSION,
        "nlms_backend": nlms_backend,
        "impulse_backend": imp_backend,
        "engine_nlms_class": engine_nlms_cls,
        "engine_impulse_class": engine_imp_cls,
        "total_hops": STEADY_HOPS,
        "warmup_hops": WARMUP_HOPS,
        "budget_ms": BUDGET_MS,
        "p50_ms": round(float(np.percentile(hop_arr, 50)), 3),
        "p95_ms": round(float(np.percentile(hop_arr, 95)), 3),
        "p99_ms": round(float(np.percentile(hop_arr, 99)), 3),
        "p999_ms": round(float(np.percentile(hop_arr, 99.9)), 3),
        "max_ms": round(float(np.max(hop_arr)), 3),
        "min_ms": round(float(np.min(hop_arr)), 3),
        "mean_ms": round(float(np.mean(hop_arr)), 3),
        "std_ms": round(float(np.std(hop_arr)), 3),
        "budget_pass": bool(np.percentile(hop_arr, 95) <= BUDGET_MS),
        "headroom_ms": round(BUDGET_MS - float(np.percentile(hop_arr, 95)), 3),
        # Component breakdown
        "dsp_p50_ms": round(float(np.percentile(dsp_arr, 50)), 3),
        "dsp_p95_ms": round(float(np.percentile(dsp_arr, 95)), 3),
        "ai_p50_ms": round(float(np.percentile(ai_arr, 50)), 3),
        "ai_p95_ms": round(float(np.percentile(ai_arr, 95)), 3),
        "regime_p50_ms": round(float(np.percentile(regime_arr, 50)), 3),
        "regime_p95_ms": round(float(np.percentile(regime_arr, 95)), 3),
        # CPU utilization
        "wall_time_s": round(wall_elapsed, 3),
        "cpu_time_s": round(cpu_elapsed, 3),
        "cpu_utilization_pct": round(cpu_elapsed / (wall_elapsed + 1e-12) * 100, 1),
    }

    # Worst contiguous latency streak
    streak_current = 0
    streak_max = 0
    streak_start_idx = 0
    worst_streak_start = 0
    for i, t in enumerate(hop_times):
        if t > STREAK_THRESHOLD_MS:
            if streak_current == 0:
                streak_start_idx = i
            streak_current += 1
            if streak_current > streak_max:
                streak_max = streak_current
                worst_streak_start = streak_start_idx
        else:
            streak_current = 0

    stats["worst_streak_count"] = streak_max
    stats["worst_streak_start_hop"] = worst_streak_start + WARMUP_HOPS
    stats["worst_streak_duration_ms"] = round(streak_max * (HOP_SIZE / SR * 1000), 1)

    # Budget violation counts
    stats["hops_over_7ms"] = int(np.sum(hop_arr > 7.0))
    stats["hops_over_8ms"] = int(np.sum(hop_arr > 8.0))
    stats["hops_over_9ms"] = int(np.sum(hop_arr > 9.0))
    stats["hops_over_10ms"] = int(np.sum(hop_arr > 10.0))

    # Save CSV
    csv_path = out_dir / "ph06_headroom_profile.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        writer.writeheader()
        writer.writerow(stats)
    print(f"\n  Saved: {csv_path}")

    # Save per-hop timing
    per_hop_path = out_dir / "ph06_per_hop_timing.csv"
    with open(per_hop_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hop_index", "total_ms", "dsp_ms", "ai_ms", "regime_ms"])
        for i in range(STEADY_HOPS):
            writer.writerow([WARMUP_HOPS + i, hop_times[i], dsp_times[i], ai_times[i], regime_times[i]])
    print(f"  Saved: {per_hop_path}")

    # Print results
    print("\n[3/3] HEADROOM BENCHMARK RESULTS")
    print("=" * 76)
    verdict = "PASS" if stats["budget_pass"] else "FAIL"
    headroom_quality = "COMFORTABLE" if stats["headroom_ms"] > 1.5 else ("ADEQUATE" if stats["headroom_ms"] > 0.5 else "TIGHT")

    print(f"  Budget:      {BUDGET_MS:.3f} ms (128 samples @ 16 kHz)")
    print(f"  Verdict:     {verdict} ({headroom_quality} headroom)")
    print(f"  Headroom:    {stats['headroom_ms']:+.3f} ms")
    print()
    print(f"  {'Metric':<16} | {'Total':>8} | {'DSP':>8} | {'AI':>8} | {'Regime':>8}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    print(f"  {'P50 (ms)':<16} | {stats['p50_ms']:>8.3f} | {stats['dsp_p50_ms']:>8.3f} | {stats['ai_p50_ms']:>8.3f} | {stats['regime_p50_ms']:>8.3f}")
    print(f"  {'P95 (ms)':<16} | {stats['p95_ms']:>8.3f} | {stats['dsp_p95_ms']:>8.3f} | {stats['ai_p95_ms']:>8.3f} | {stats['regime_p95_ms']:>8.3f}")
    print(f"  {'P99 (ms)':<16} | {stats['p99_ms']:>8.3f} |")
    print(f"  {'P99.9 (ms)':<16} | {stats['p999_ms']:>8.3f} |")
    print(f"  {'Max (ms)':<16} | {stats['max_ms']:>8.3f} |")
    print(f"  {'Mean (ms)':<16} | {stats['mean_ms']:>8.3f} |")
    print(f"  {'StdDev (ms)':<16} | {stats['std_ms']:>8.3f} |")
    print()
    print(f"  Hops > 7.0 ms:  {stats['hops_over_7ms']:>5} / {STEADY_HOPS}  ({stats['hops_over_7ms']/STEADY_HOPS*100:.1f}%)")
    print(f"  Hops > 8.0 ms:  {stats['hops_over_8ms']:>5} / {STEADY_HOPS}  ({stats['hops_over_8ms']/STEADY_HOPS*100:.1f}%)")
    print(f"  Hops > 9.0 ms:  {stats['hops_over_9ms']:>5} / {STEADY_HOPS}  ({stats['hops_over_9ms']/STEADY_HOPS*100:.1f}%)")
    print(f"  Hops > 10.0 ms: {stats['hops_over_10ms']:>5} / {STEADY_HOPS}  ({stats['hops_over_10ms']/STEADY_HOPS*100:.1f}%)")
    print()
    print(f"  Worst streak (>{STREAK_THRESHOLD_MS:.0f}ms): {stats['worst_streak_count']} consecutive hops "
          f"({stats['worst_streak_duration_ms']:.1f} ms audio)")
    print(f"  CPU utilization: {stats['cpu_utilization_pct']:.1f}%")
    print(f"  Wall time: {stats['wall_time_s']:.1f}s for {STEADY_HOPS} hops")
    print("=" * 76)

    return stats


if __name__ == "__main__":
    run_headroom_benchmark()
