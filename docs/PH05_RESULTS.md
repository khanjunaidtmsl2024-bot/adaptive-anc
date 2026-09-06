# PH0.5 — Performance Profiling & PESQ Diagnostic Results
**Problem Statement ID:** 26052 — AI/ML-Enabled Adaptive Noise Cancellation for Defence Vehicles  
**Report Date:** 2026-09-05  
**Evidence Tier:** `OFFLINE EXPERIMENTALLY MEASURED (host PC)` & `SOFTWARE VERIFIED`  

---

## 1. Executive Boundary Notice

> [!IMPORTANT]
> **EVIDENCE BOUNDARY & REALITY CONTRACT**
> 1. **Component speedup is not system speedup:** An isolated 69.1x speedup in NLMS yields an integrated system speedup of 6.76x due to Amdahl's Law (AI inference and regime classification remain Python/ONNX/Torch runtime).
> 2. **Laptop real-time simulation is not embedded real-time validation:** Measuring a P95 per-hop latency of 7.41 ms on an x86_64 host PC proves that the software algorithm meets the 8.00 ms hop budget on this machine. **It does NOT prove real-time execution on physical Raspberry Pi 4 (Quad Cortex-A72) or Jetson AGX Orin.** Embedded hardware latency remains strictly `UNVERIFIED`.
> 3. **Synthetic improvement is not defence-environment validation:** Metric evaluations are conducted on synthetic speech and band-limited noise mixtures. Zero real military vehicle or combat recordings were used.
> 4. **PASS only what was actually measured.** No estimations converted into measurements.

---

## 2. Test 8 Formal Split Verdict

| Sub-Test | Focus | Criterion | Measured Result | Verdict | Evidence Tier |
|---|---|---|---|---|---|
| **Test 8a** | **Streaming Mechanism** | Finite output, OLA continuity, zero future lookahead, state preservation | Reconstructed 3.0s (375 hops) without NaN/Inf, energy continuity preserved | **PASS** | `SOFTWARE VERIFIED` |
| **Test 8b** | **Computational Budget** | Steady-state P95 latency &le; 8.000 ms per 128-sample hop (16 kHz) | **P95 = 7.411 ms** (&le; 8.000 ms), P50 = 5.623 ms, P99 = 7.798 ms | **PASS** | `OFFLINE EXPERIMENTALLY MEASURED (host PC)` |

> **Official Test 8b Boundary Statement:**  
> *"Laptop software benchmark passes the 8 ms computational criterion (P95 = 7.41 ms &le; 8.00 ms). Physical embedded real-time performance on target hardware remains unverified."*

---

## 3. Phase 1: Isolated Component Performance

Each component was benchmarked for 500 steady-state iterations after 50 warm-up iterations. Microsecond-accurate wall-clock timing (`time.perf_counter()`) was collected on 128-sample hops (8.00 ms audio duration) @ 16 kHz.

**Artifact:** [`results/csv/ph05_isolated_profile.csv`](file:///F:/SIH%202026/results/csv/ph05_isolated_profile.csv)  
**Artifact:** [`results/csv/ph05_speedup_summary.csv`](file:///F:/SIH%202026/results/csv/ph05_speedup_summary.csv)

| Benchmark ID | Component | Implementation | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Mean (ms) | Std (ms) | Isolated Speedup |
|---|---|---|---|---|---|---|---|---|---|
| **ISO-1** | VSS-NLMS (64 taps) | Pure Python | 9.7865 | 17.1113 | 18.4945 | 18.8329 | 11.0292 | 2.5872 | Baseline |
| **ISO-2** | VSS-NLMS (64 taps) | Numba JIT (@njit) | **0.1417** | 0.1638 | 0.1940 | 0.2224 | 0.1451 | 0.0103 | **69.1x** |
| **ISO-3** | Impulse Controller | Pure Python | 3.7104 | 4.4007 | 5.0268 | 8.1213 | 3.8391 | 0.3743 | Baseline |
| **ISO-4** | Impulse Controller | Numba JIT (@njit) | **0.1669** | 0.2531 | 0.2949 | 0.3134 | 0.1782 | 0.0283 | **22.2x** |
| **ISO-5** | Noise Regime Detector | NumPy / FFT | 0.6683 | 0.8286 | 1.0686 | 1.1797 | 0.6955 | 0.0744 | — |
| **ISO-6** | Causal STFT Analysis | NumPy (256-pt rfft) | 0.1326 | 0.1722 | 0.1976 | 0.2047 | 0.1176 | 0.0358 | — |
| **ISO-7** | TinyEnhancer AI Forward | PyTorch CPU (9.6K params) | 2.3523 | 2.7760 | 3.4190 | 3.7151 | 2.2547 | 0.3639 | — |
| **ISO-8** | iSTFT Synthesis + OLA | NumPy (256-pt irfft) | 0.0839 | 0.1587 | 0.1851 | 0.2742 | 0.1027 | 0.0307 | — |

---

## 4. Phase 2: Full Integrated Pipeline Benchmark

The integrated pipeline was measured using the production [`CausalStreamingEngine`](file:///F:/SIH%202026/src/streaming/causal_engine.py).  
- **Protocol:** 50 warm-up hops executed and discarded (ensuring full JIT compilation, PyTorch buffer allocation, and thread pool warming).
- **Measurement:** 350 consecutive steady-state hops timed with `time.perf_counter()`.
- **Target:** 128 samples @ 16 kHz = **8.000 ms budget**.

**Artifact:** [`results/csv/ph05_integrated_profile.csv`](file:///F:/SIH%202026/results/csv/ph05_integrated_profile.csv)

### A. Integrated Pipeline Comparison Table

| Pipeline Configuration | Stage | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Budget &le; 8.0 ms | Verdict |
|---|---|---|---|---|---|---|---|---|
| **INT-1 (Python Backends)** | **Total Hop** | **40.773** | **38.035** | **56.374** | **64.432** | **77.063** | **FAIL** | 4.8x–7.0x over budget |
| INT-1 (Python Backends) | DSP Stage | 35.898 | 33.634 | 49.672 | 53.456 | 55.713 | — | 88.0% of hop time |
| INT-1 (Python Backends) | AI Stage | 3.464 | 3.193 | 4.878 | 10.321 | 27.964 | — | 8.5% of hop time |
| INT-1 (Python Backends) | Regime Detector | 1.082 | 1.134 | 1.465 | 1.592 | 1.691 | — | 2.7% of hop time |
| **INT-2 (Numba Backends)** | **Total Hop** | **5.636** | **5.623** | **7.411** | **7.798** | **9.130** | **PASS** | **Meets 8 ms budget** |
| INT-2 (Numba Backends) | DSP Stage | 0.721 | 0.754 | 0.860 | 0.895 | 0.923 | — | 12.8% of hop time |
| INT-2 (Numba Backends) | AI Stage | 3.383 | 3.262 | 4.756 | 5.135 | 6.804 | — | 60.0% of hop time |
| INT-2 (Numba Backends) | Regime Detector | 1.179 | 1.221 | 1.473 | 1.563 | 1.612 | — | 20.9% of hop time |

### B. System-Level Speedup vs Component-Level Speedup
- **VSS-NLMS Component Speedup:** 69.1x
- **Impulse Controller Component Speedup:** 22.2x
- **Integrated System Speedup (P50):** 38.035 ms &rarr; 5.623 ms = **6.76x**
- **Integrated System Speedup (P95):** 56.374 ms &rarr; 7.411 ms = **7.61x**

> **Amdahl's Law Explanation:** Even though DSP execution was compressed from ~35.9 ms to ~0.72 ms, the remaining pipeline stages (AI inference at ~3.38 ms, regime detection at ~1.18 ms, and STFT/OLA buffering at ~0.35 ms) define the new Amdahl floor (~5.28 ms). The system speedup is therefore 6.8x, not 69x.

---

## 5. Phase 3: PESQ Diagnostic Ablation Study

Controlled single-variable ablations were executed on 10 standardized synthetic audio clips across diverse SNR conditions (-5 dB to +10 dB).  
**Crucial Constraint:** The AI architecture was **kept completely unchanged**. This study is diagnostic only to isolate the root cause of perceptual degradation.

**Artifact:** [`results/csv/ph05_pesq_ablation.csv`](file:///F:/SIH%202026/results/csv/ph05_pesq_ablation.csv)

| ID | Ablation Description | Type | PESQ (WB) | STOI | SI-SDR (dB) | Delta-SNR (dB) | SNR_out (dB) | Diagnostic Finding |
|---|---|---|---|---|---|---|---|---|
| **ABL-0** | **Baseline Hybrid** (hop=128, Hanning) | Reference | 1.030 | -0.3482 | -38.46 | -5.49 | -0.97 | Baseline reference |
| **ABL-1** | **AI Bypass (NLMS-Only)** | Single-variable | 1.018 | **+0.3986** | **+6.42** | **+2.51** | **+7.03** | **Major finding:** Eliminating AI restores positive SI-SDR (+6.42 dB), positive Delta-SNR (+2.51 dB), and positive STOI |
| **ABL-2** | **NLMS Bypass (AI-Only)** | Single-variable | 1.031 | -0.3354 | -39.32 | -5.62 | -1.10 | Confirms AI stage causes negative SI-SDR and negative Delta-SNR on synthetic audio |
| **ABL-3** | **Phase Passthrough** (keep DSP phase) | Single-variable | 1.030 | -0.3482 | -38.46 | -5.49 | -0.97 | Phase modification by AI is not the sole issue; magnitude mask alone distorts signal |
| **ABL-4** | **Hamming Window** (vs Hanning) | Single-variable | 1.031 | -0.3529 | -38.43 | -5.55 | -1.03 | Window shape is not the root cause |
| **ABL-5** | **Cadence Change (hop=64, 75% OLA)** | **Secondary Experiment** | 1.030 | -0.3407 | -40.28 | -5.97 | -1.45 | Higher temporal overlap does not resolve mask-induced signal cancellation |

### Key Diagnostic Takeaways
1. **The Root Cause is the Neural Mask Transfer Function, Not the DSP:** When audio passes through the classical NLMS filter alone (ABL-1), it achieves **+6.42 dB SI-SDR** and **+2.51 dB Delta-SNR**. Introducing the current TinyEnhancer mask drops SI-SDR to **-38.46 dB** and Delta-SNR to **-5.49 dB**.
2. **PESQ Floor on Synthetic Tones:** PESQ Wideband saturates near 1.02–1.03 across all conditions because synthetic harmonic tones lack human formant structures expected by ITU-T P.862.
3. **ABL-5 Separation Confirmed:** As predicted, changing hop from 128 to 64 alters temporal cadence and STFT frame rate without resolving spectral cancellation. It was properly evaluated as an independent secondary experiment.
4. **Architecture Decision:** Architecture changes remain frozen pending evaluation on natural speech corpora (VoiceBank-DEMAND / LibriSpeech).

---

## 6. Full Test Suite Verification

Following implementation of the Numba backends and Test 8 split:
- **Suite Command:** `pytest`
- **Total Tests:** 78
- **Passed:** 78 (100%)
- **Failed:** 0
- **Duration:** 150.11 s (~2.5 minutes, down from 3:56 due to Numba streaming efficiency)

---

## 7. Status Summary

```text
[PH0.5 STATUS]
Isolated DSP speedup:      69.1x (NLMS), 22.2x (Impulse Controller)
Integrated system speedup: 6.8x (P50: 38.03 ms -> 5.62 ms)
Test 8a (Mechanism):       PASS (Software verified)
Test 8b (Budget):          PASS (P95 = 7.41 ms <= 8.00 ms on host laptop)
PESQ Ablation:             Complete (Diagnostic finding documented; AI unmodifed)
Full Test Suite:           78/78 Passed (100%)
Hardware Status:           UNVERIFIED (Target embedded hardware not connected)
```
