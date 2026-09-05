"""
Gate 7: Hop-by-Hop Computational Budget Profiler for Raspberry Pi 4.
PS 26052 — Adaptive Defence ANC.

Mathematical Framework:
    - Hop Size: 128 samples at 16 kHz = 8.000 ms hop period.
    - Available Processing Time Budget: 8.000 ms per hop.
    - Algorithmic Pipeline Latency: (512 - 128) / 16000 = 24.000 ms.
    - Target Processing Headroom: >= 35% margin to prevent ALSA buffer underruns (XRUNs).

Instruments every individual stage of the streaming loop:
1. Buffer Management (FIFO shifts)
2. Noise Regime Detection
3. Stage 1 VSS-NLMS Filtering (64 taps)
4. Impulse Transient Controller
5. STFT Windowing & RFFT (512 -> 257 bins)
6. Stage 2 AI TinyEnhancer Forward Inference
7. iSTFT Synthesis & Windowing
8. Overlap-Add & Output Buffer Management

Outputs:
- results/csv/hop_budget_profile.csv
- Terminal diagnostic budget table
"""

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.streaming.causal_engine import CausalStreamingEngine
from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.impulse_protection import ImpulseProtectionController
from src.dsp.noise_regime_detector import NoiseRegimeDetector
from src.ai.tiny_enhancer import TinyEnhancerWrapper


def profile_pipeline_budget(
    n_hops: int = 150,
    sr: int = 16000,
    frame_size: int = 512,
    hop_size: int = 128,
    output_csv: str = "results/csv/hop_budget_profile.csv"
) -> Dict[str, Any]:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    available_budget_ms = hop_size / sr * 1000.0  # 8.000 ms
    algorithmic_latency_ms = (frame_size - hop_size) / sr * 1000.0  # 24.000 ms

    print(f"[GATE 7] Profiling Hop-by-Hop Computational Budget across {n_hops} hops...", flush=True)
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Hop Size: {hop_size} samples ({available_budget_ms:.3f} ms available budget)")
    print(f"  Frame Size: {frame_size} samples ({frame_size / sr * 1000.0:.1f} ms window)")
    print(f"  Algorithmic Pipeline Latency: {algorithmic_latency_ms:.1f} ms\n", flush=True)

    # Initialize components
    nlms = VSSNLMSFilter(filter_length=64, mu_init=0.05)
    impulse_ctl = ImpulseProtectionController(sample_rate=sr)
    regime_det = NoiseRegimeDetector(sample_rate=sr, frame_size=frame_size)
    ai_model = TinyEnhancerWrapper()
    window = np.hanning(frame_size).astype(np.float32)

    # State buffers
    buf_primary = np.zeros(frame_size, dtype=np.float32)
    buf_reference = np.zeros(frame_size, dtype=np.float32)
    overlap_buf = np.zeros(frame_size, dtype=np.float32)

    rng = np.random.RandomState(42)

    # Stage timing lists (in microseconds)
    times_buffer_shift = []
    times_regime = []
    times_nlms = []
    times_impulse = []
    times_stft = []
    times_ai = []
    times_istft = []
    times_ola = []
    times_total = []

    # Warmup
    for _ in range(10):
        dummy_p = rng.randn(hop_size).astype(np.float32) * 0.1
        dummy_r = rng.randn(hop_size).astype(np.float32) * 0.1
        buf_primary[:-hop_size] = buf_primary[hop_size:]
        buf_primary[-hop_size:] = dummy_p
        buf_reference[:-hop_size] = buf_reference[hop_size:]
        buf_reference[-hop_size:] = dummy_r
        d_out, _, _ = nlms.filter_block(buf_primary[-hop_size:], buf_reference[-hop_size:])
        w = d_out * window[:hop_size]

    # Benchmark profiling loop
    for hop_idx in range(n_hops):
        p_chunk = rng.randn(hop_size).astype(np.float32) * 0.1
        r_chunk = rng.randn(hop_size).astype(np.float32) * 0.1

        t_total_0 = time.perf_counter()

        # 1. Input buffer shift
        t0 = time.perf_counter()
        buf_primary[:-hop_size] = buf_primary[hop_size:]
        buf_primary[-hop_size:] = p_chunk
        buf_reference[:-hop_size] = buf_reference[hop_size:]
        buf_reference[-hop_size:] = r_chunk
        t_buf = (time.perf_counter() - t0) * 1e6

        # 2. Noise Regime Detection
        t0 = time.perf_counter()
        regime, _ = regime_det.classify_frame(buf_primary)
        t_reg = (time.perf_counter() - t0) * 1e6

        # 3. Stage 1 VSS-NLMS (processes the new 128 incoming samples)
        t0 = time.perf_counter()
        dsp_hop, _, _ = nlms.filter_block(p_chunk, r_chunk)
        t_dsp = (time.perf_counter() - t0) * 1e6

        # 4. Impulse transient protection
        t0 = time.perf_counter()
        dsp_hop, _, _ = impulse_ctl.filter_block_protection(dsp_hop)
        t_imp = (time.perf_counter() - t0) * 1e6

        # 5. STFT Analysis
        t0 = time.perf_counter()
        windowed = buf_primary * window
        stft_frame = np.fft.rfft(windowed)
        mag = np.abs(stft_frame).reshape(-1, 1)
        phase = np.angle(stft_frame).reshape(-1, 1)
        t_stft = (time.perf_counter() - t0) * 1e6

        # 6. Stage 2 AI Inference
        t0 = time.perf_counter()
        enh_mag, enh_phase = ai_model.enhance_spectrogram(mag, phase)
        t_ai = (time.perf_counter() - t0) * 1e6

        # 7. iSTFT Synthesis
        t0 = time.perf_counter()
        enh_stft = enh_mag.flatten() * np.exp(1j * enh_phase.flatten())
        recon = np.fft.irfft(enh_stft, n=frame_size) * window
        t_istft = (time.perf_counter() - t0) * 1e6

        # 8. Overlap-Add & Output Buffer Shift
        t0 = time.perf_counter()
        overlap_buf += recon
        out_chunk = overlap_buf[:hop_size].copy() / 1.5
        overlap_buf[:-hop_size] = overlap_buf[hop_size:]
        overlap_buf[-hop_size:] = 0.0
        t_ola = (time.perf_counter() - t0) * 1e6

        t_total = (time.perf_counter() - t_total_0) * 1000.0

        times_buffer_shift.append(t_buf)
        times_regime.append(t_reg)
        times_nlms.append(t_dsp)
        times_impulse.append(t_imp)
        times_stft.append(t_stft)
        times_ai.append(t_ai)
        times_istft.append(t_istft)
        times_ola.append(t_ola)
        times_total.append(t_total)

    # Compute statistics
    def calc_stats(arr_us):
        arr_ms = np.array(arr_us) / 1000.0
        return {
            "mean_ms": round(float(np.mean(arr_ms)), 3),
            "p95_ms": round(float(np.percentile(arr_ms, 95)), 3),
            "max_ms": round(float(np.max(arr_ms)), 3),
        }

    stages = [
        ("Buffer Shift (FIFO)", times_buffer_shift),
        ("Regime Detection", times_regime),
        ("VSS-NLMS (64 taps)", times_nlms),
        ("Impulse Controller", times_impulse),
        ("STFT Analysis (512)", times_stft),
        ("AI TinyEnhancer", times_ai),
        ("iSTFT Synthesis", times_istft),
        ("Overlap-Add & Output", times_ola),
    ]

    records = []
    for name, t_list in stages:
        st = calc_stats(t_list)
        pct = (st["mean_ms"] / (np.mean(times_total) + 1e-12)) * 100.0
        records.append({
            "stage": name,
            "mean_time_ms": st["mean_ms"],
            "p95_time_ms": st["p95_ms"],
            "max_time_ms": st["max_ms"],
            "pct_of_total": round(pct, 1),
        })

    total_mean_ms = float(np.mean(times_total))
    total_p95_ms = float(np.percentile(times_total, 95))
    total_max_ms = float(np.max(times_total))
    headroom_pct = ((available_budget_ms - total_mean_ms) / available_budget_ms) * 100.0
    budget_satisfied = bool(total_max_ms < available_budget_ms)

    # Save CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("--- HOP-BY-HOP COMPUTATIONAL BUDGET BREAKDOWN ---")
    print(f"{'Pipeline Processing Stage':<26} | {'Mean (ms)':<9} | {'P95 (ms)':<9} | {'Worst (ms)':<10} | {'Budget %':<8}")
    print("-" * 72)
    for r in records:
        print(f"{r['stage']:<26} | {r['mean_time_ms']:7.3f} ms | {r['p95_time_ms']:7.3f} ms | {r['max_time_ms']:8.3f} ms | {r['pct_of_total']:6.1f}%")
    print("-" * 72)
    print(f"{'TOTAL SYSTEM COMPUTE':<26} | {total_mean_ms:7.3f} ms | {total_p95_ms:7.3f} ms | {total_max_ms:8.3f} ms | 100.0%")
    print(f"{'AVAILABLE BUDGET PER HOP':<26} | {available_budget_ms:7.3f} ms | {available_budget_ms:7.3f} ms | {available_budget_ms:8.3f} ms |")
    print(f"HEADROOM MARGIN: {headroom_pct:.1f}% ({available_budget_ms - total_mean_ms:.3f} ms remaining per hop)")
    print(f"BUDGET SATISFIED (Worst-Case < 8.0 ms): {'YES (BUDGET CLEARED)' if budget_satisfied else 'NO (RISK OF XRUN)'}\n")

    return {
        "records": records,
        "total_mean_ms": total_mean_ms,
        "total_p95_ms": total_p95_ms,
        "total_max_ms": total_max_ms,
        "budget_satisfied": budget_satisfied,
    }


if __name__ == "__main__":
    profile_pipeline_budget()
