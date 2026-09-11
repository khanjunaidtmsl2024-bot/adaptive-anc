"""
Latency and Model Profiler for PH2A Causal Models.
PS 26052 — Adaptive Defence ANC.

Benchmarks 1,000 steady-state streaming iterations (8.0 ms hops = 128 samples @ 16 kHz):
- DSP (VSS-NLMS)
- AI Forward Pass (Causal TinyEnhancer stateful streaming T=1)
- Total Hop Latency
Measures P50, P95, P99, P99.9, Max, Mean, RTF.
"""

import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.vss_nlms import VSSNLMSFilter
from src.ai.tiny_enhancer import TinyEnhancerNet


def profile_latency(e2_ckpt_path: str, n_iters: int = 1000):
    device = "cpu"
    torch.set_num_threads(4)

    # Initialize models
    nlms = VSSNLMSFilter(filter_length=64)
    model = TinyEnhancerNet().to(device)
    model.load_state_dict(torch.load(e2_ckpt_path, map_location=device))
    model.eval()

    # Pre-generate inputs
    rng = np.random.RandomState(42)
    primary_hops = [rng.randn(128).astype(np.float32) * 0.1 for _ in range(n_iters)]
    ref_hops = [rng.randn(128).astype(np.float32) * 0.1 for _ in range(n_iters)]
    stft_frames = [torch.randn(1, 1, 129, 1).to(device) for _ in range(n_iters)]

    # Warmup
    model.reset_state()
    for _ in range(50):
        nlms.filter_block(primary_hops[0], ref_hops[0])
        with torch.no_grad():
            model(stft_frames[0], stateful=True)

    # Benchmark DSP
    times_dsp = []
    for i in range(n_iters):
        t0 = time.perf_counter_ns()
        nlms.filter_block(primary_hops[i], ref_hops[i])
        t1 = time.perf_counter_ns()
        times_dsp.append((t1 - t0) / 1e6)

    # Benchmark AI Causal Streaming (T=1)
    model.reset_state()
    times_ai = []
    with torch.no_grad():
        for i in range(n_iters):
            t0 = time.perf_counter_ns()
            model(stft_frames[i], stateful=True)
            t1 = time.perf_counter_ns()
            times_ai.append((t1 - t0) / 1e6)

    # Combined Total
    times_total = [d + a for d, a in zip(times_dsp, times_ai)]

    hop_budget_ms = 8.0

    def calc_stats(times):
        arr = np.array(times)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "p99_9": float(np.percentile(arr, 99.9)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "rtf_p50": float(np.percentile(arr, 50) / hop_budget_ms),
            "rtf_p95": float(np.percentile(arr, 95) / hop_budget_ms),
        }

    stats = {
        "dsp_vss_nlms": calc_stats(times_dsp),
        "ai_causal_tinyenhancer": calc_stats(times_ai),
        "total_causal_pipeline": calc_stats(times_total),
    }

    # Model parameters & size
    param_count = sum(p.numel() for p in model.parameters())
    ckpt_file = Path(e2_ckpt_path)
    ckpt_bytes = ckpt_file.stat().st_size
    h = hashlib.sha256()
    with open(ckpt_file, "rb") as f:
        h.update(f.read())
    ckpt_sha256 = h.hexdigest()

    stats["model_info"] = {
        "parameter_count": param_count,
        "checkpoint_bytes": ckpt_bytes,
        "checkpoint_sha256": ckpt_sha256,
        "checkpoint_path": str(ckpt_file.resolve()),
    }

    out_json = Path("results/ph2a_causal/latency_benchmark.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n--- PH2A LATENCY BENCHMARK (1000 iterations, 8.0 ms budget) ---")
    print(f"DSP (VSS-NLMS):        P50 = {stats['dsp_vss_nlms']['p50']:.3f} ms | P95 = {stats['dsp_vss_nlms']['p95']:.3f} ms | Max = {stats['dsp_vss_nlms']['max']:.3f} ms")
    print(f"AI (E2_causal):        P50 = {stats['ai_causal_tinyenhancer']['p50']:.3f} ms | P95 = {stats['ai_causal_tinyenhancer']['p95']:.3f} ms | Max = {stats['ai_causal_tinyenhancer']['max']:.3f} ms")
    print(f"TOTAL CAUSAL PIPELINE: P50 = {stats['total_causal_pipeline']['p50']:.3f} ms | P95 = {stats['total_causal_pipeline']['p95']:.3f} ms | Max = {stats['total_causal_pipeline']['max']:.3f} ms")
    print(f"RTF (P50):             {stats['total_causal_pipeline']['rtf_p50']:.4f} (budget: 8.0 ms)")
    print(f"Parameters:            {param_count:,d}")
    print(f"Checkpoint Size:       {ckpt_bytes:,d} bytes")
    print(f"Checkpoint SHA-256:    {ckpt_sha256}")
    return stats


if __name__ == "__main__":
    profile_latency("checkpoints/E2_causal.pt")
