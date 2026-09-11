#!/usr/bin/env python3
"""
PS 26052: Embedded Latency Benchmark for Raspberry Pi 4 / 5.
Executes 1,000 steady-state iterations on the target CPU using ONLY
lightweight embedded runtimes: NumPy + ONNX Runtime (+ optional Numba).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import onnxruntime as ort
except ImportError:
    print("[!] Error: onnxruntime is not installed. Run: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)


def benchmark_embedded(
    onnx_model_path: str,
    n_iters: int = 1000,
    threads: int = 4,
    use_numba_if_available: bool = True,
) -> Dict[str, Any]:
    model_file = Path(onnx_model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print("=" * 72)
    print("  PS 26052: EMBEDDED REAL-TIME INFERENCE BENCHMARK")
    print(f"  Target Model: {model_file.name}")
    print(f"  Iterations:   {n_iters}")
    print(f"  ORT Threads:  {threads}")
    print("=" * 72)

    # 1. Setup VSS-NLMS DSP Filter (Numba if available, else pure NumPy)
    numba_available = False
    dsp_fn = None
    filter_length = 64
    w = np.zeros(filter_length, dtype=np.float32)
    buf = np.zeros(filter_length, dtype=np.float32)
    mu_val = 0.05
    p_cor = 0.0

    if use_numba_if_available:
        try:
            from numba import njit
            @njit(cache=True)
            def _numba_step(pri, ref, w, buf, mu_v, pc):
                n = len(pri)
                err = np.zeros(n, dtype=np.float32)
                for i in range(n):
                    buf[1:] = buf[:-1]
                    buf[0] = ref[i]
                    y = 0.0
                    for k in range(len(w)):
                        y += w[k] * buf[k]
                    e = pri[i] - y
                    err[i] = e
                    pc = 0.95 * pc + 0.05 * (e * y)
                    mu_v = min(max(0.98 * mu_v + 0.02 * abs(pc), 1e-4), 0.5)
                    norm = 1e-6
                    for k in range(len(buf)):
                        norm += buf[k] * buf[k]
                    step = (mu_v / norm) * e
                    for k in range(len(w)):
                        w[k] = (w[k] + step * buf[k]) * (1.0 - 1e-4 * mu_v)
                return err, mu_v, pc

            numba_available = True
            print("[+] Numba available. Using native compiled DSP.")
            def dsp_block_numba(pri, ref):
                nonlocal w, buf, mu_val, p_cor
                err, mu_val, p_cor = _numba_step(pri, ref, w, buf, mu_val, p_cor)
                return err
            dsp_fn = dsp_block_numba
        except ImportError:
            print("[-] Numba not found. Using pure NumPy vectorised DSP fallback.")

    if dsp_fn is None:
        def dsp_block_numpy(primary: np.ndarray, ref: np.ndarray) -> np.ndarray:
            nonlocal w, buf, mu_val, p_cor
            n = len(primary)
            err = np.zeros(n, dtype=np.float32)
            for i in range(n):
                buf[1:] = buf[:-1]
                buf[0] = ref[i]
                y = float(np.dot(w, buf))
                e = primary[i] - y
                err[i] = e
                p_cor = 0.95 * p_cor + 0.05 * (e * y)
                mu_val = np.clip(0.98 * mu_val + 0.02 * abs(p_cor), 1e-4, 0.5)
                norm = float(np.dot(buf, buf)) + 1e-6
                w += (mu_val / norm) * e * buf
                w *= (1.0 - 1e-4 * mu_val)
            return err
        dsp_fn = dsp_block_numpy

    # 2. Setup ONNX Runtime Session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_file), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"[+] ONNX Runtime session active. Input: '{input_name}', Provider: CPUExecutionProvider")

    # Pre-generate inputs
    rng = np.random.RandomState(42)
    frames_t1 = [rng.randn(1, 1, 129, 1).astype(np.float32) for _ in range(n_iters)]
    frames_t9 = [rng.randn(1, 1, 129, 9).astype(np.float32) for _ in range(n_iters)]
    pri_hops = [rng.randn(128).astype(np.float32) * 0.1 for _ in range(n_iters)]
    ref_hops = [rng.randn(128).astype(np.float32) * 0.1 for _ in range(n_iters)]

    # Warmup
    print("[+] Warming up execution pipelines (50 iterations)...")
    for i in range(50):
        session.run(None, {input_name: frames_t1[i % len(frames_t1)]})
        session.run(None, {input_name: frames_t9[i % len(frames_t9)]})
        dsp_fn(pri_hops[i % len(pri_hops)], ref_hops[i % len(ref_hops)])

    # Benchmark AI T=1 (Single frame)
    print(f"[+] Benchmarking ONNX FP32 T=1 inference ({n_iters} iterations)...")
    times_ai_t1 = []
    for i in range(n_iters):
        t0 = time.perf_counter_ns()
        session.run(None, {input_name: frames_t1[i]})
        t1 = time.perf_counter_ns()
        times_ai_t1.append((t1 - t0) / 1e6)

    # Benchmark AI T=9 (Receptive field window)
    print(f"[+] Benchmarking ONNX FP32 T=9 inference ({n_iters} iterations)...")
    times_ai_t9 = []
    for i in range(n_iters):
        t0 = time.perf_counter_ns()
        session.run(None, {input_name: frames_t9[i]})
        t1 = time.perf_counter_ns()
        times_ai_t9.append((t1 - t0) / 1e6)

    # Benchmark DSP
    print(f"[+] Benchmarking DSP 128-sample filter_block ({n_iters} iterations)...")
    times_dsp = []
    for i in range(n_iters):
        t0 = time.perf_counter_ns()
        dsp_fn(pri_hops[i], ref_hops[i])
        t1 = time.perf_counter_ns()
        times_dsp.append((t1 - t0) / 1e6)

    # Combined pipeline
    times_total = [ai + dsp for ai, dsp in zip(times_ai_t1, times_dsp)]

    hop_budget_ms = 8.0

    def calc_stats(times: List[float]) -> Dict[str, float]:
        arr = np.array(times)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "p99_9_ms": float(np.percentile(arr, 99.9)),
            "max_ms": float(np.max(arr)),
            "mean_ms": float(np.mean(arr)),
            "rtf_p50": float(np.percentile(arr, 50) / hop_budget_ms),
            "rtf_p95": float(np.percentile(arr, 95) / hop_budget_ms),
        }

    results = {
        "n_iters": n_iters,
        "hop_budget_ms": hop_budget_ms,
        "ai_fp32_t1": calc_stats(times_ai_t1),
        "ai_fp32_t9": calc_stats(times_ai_t9),
        "dsp_128": calc_stats(times_dsp),
        "total_pipeline": calc_stats(times_total),
    }

    print("\n" + "=" * 72)
    print("  MEASURED BENCHMARK RESULTS")
    print("=" * 72)
    print(f"  AI FP32 (T=1):     P50 = {results['ai_fp32_t1']['p50_ms']:6.3f} ms | P95 = {results['ai_fp32_t1']['p95_ms']:6.3f} ms | Max = {results['ai_fp32_t1']['max_ms']:6.3f} ms")
    print(f"  AI FP32 (T=9):     P50 = {results['ai_fp32_t9']['p50_ms']:6.3f} ms | P95 = {results['ai_fp32_t9']['p95_ms']:6.3f} ms | Max = {results['ai_fp32_t9']['max_ms']:6.3f} ms")
    print(f"  DSP 128-sample:    P50 = {results['dsp_128']['p50_ms']:6.3f} ms | P95 = {results['dsp_128']['p95_ms']:6.3f} ms | Max = {results['dsp_128']['max_ms']:6.3f} ms")
    print(f"  Combined Pipeline: P50 = {results['total_pipeline']['p50_ms']:6.3f} ms | P95 = {results['total_pipeline']['p95_ms']:6.3f} ms | Max = {results['total_pipeline']['max_ms']:6.3f} ms")
    print(f"  Real-Time Factor:  RTF (P50) = {results['total_pipeline']['rtf_p50']:.4f} | RTF (P95) = {results['total_pipeline']['rtf_p95']:.4f}")
    if results['total_pipeline']['p95_ms'] <= hop_budget_ms:
        print(f"  [PASS] HARD REAL-TIME BUDGET MET (P95 {results['total_pipeline']['p95_ms']:.3f} ms <= {hop_budget_ms:.1f} ms)")
    else:
        print(f"  [FAIL] HARD REAL-TIME BUDGET EXCEEDED (P95 {results['total_pipeline']['p95_ms']:.3f} ms > {hop_budget_ms:.1f} ms)")
    print("=" * 72)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedded Latency Benchmark for PS 26052.")
    parser.add_argument("--model", type=str, default="models/E2_causal.onnx", help="Path to E2_causal.onnx")
    parser.add_argument("--iters", type=int, default=1000, help="Number of benchmark iterations")
    parser.add_argument("--threads", type=int, default=4, help="Number of intra-op threads")
    parser.add_argument("--out", type=str, default="embedded_benchmark_results.json", help="Path to output JSON")
    args = parser.parse_args()

    res = benchmark_embedded(args.model, n_iters=args.iters, threads=args.threads)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[+] Results saved to {args.out}")
