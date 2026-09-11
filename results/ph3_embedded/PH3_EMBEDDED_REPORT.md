# PH3 Gate Report: Physical Edge Deployment & Embedded Bring-Up Audit
**Project:** PS 26052 — Defence-Grade AI/ML Adaptive Noise Cancellation  
**Date:** September 8, 2026  
**Auditing Agent:** Antigravity (Advanced Agentic Systems, DeepMind)  
**Status:** 🟡 **HOST ONNX DEPLOYMENT PREPARATION & VALIDATION ACCEPTED — PHYSICAL EMBEDDED BRING-UP PENDING HARDWARE**  

---

## Executive Summary

Pursuant to the **PH3 — Physical Edge Deployment / Raspberry Pi Bring-Up Directive**, the official frozen causal baseline:
- **Model:** `E2_causal`
- **Parameters:** 9,569
- **Checkpoint SHA-256:** `0c8f08a1bdd41ff9543b45f122f48cc3c5208b7e8395ec656c557f4bd6482acd`
- **PH2A.1 Performance:** +8.96 to +9.37 dB (TEST_B), +11.56 dB (TEST_C), Clean Speech Attenuation −0.96 dB

was subjected to standalone ONNX export, numerical equivalence auditing, CPU dynamic quantization evaluation, and deployment packaging.

Per the absolute rule of PH3:
> *"A host benchmark is NOT a Raspberry Pi benchmark.*  
> *A simulated audio loop is NOT physical hardware validation.*  
> *An exported ONNX model is NOT deployment validation.*  
> *Only measured hardware evidence gets the label VERIFIED EXPERIMENT."*

### Official Project Classification
- **Host ONNX Deployment Validation:** **VERIFIED EXPERIMENT** (numerical equivalence $\le 2.09 \times 10^{-7}$, host CPU inference P95 = 0.34 ms, host DSP P95 = 0.23 ms).
- **Raspberry Pi Deployment:** **NOT EXECUTED / PENDING HARDWARE** (no physical board connected).
- **Raspberry Pi Latency:** **HYPOTHESIS** (unmeasured on ARM).
- **Physical End-to-End Latency:** **UNMEASURED** (requires oscilloscope loopback measurement on the bench).
- **Embedded Real-Time Claim:** **NOT PROVEN** (hardware gate remains open).
- **Regression Test Suite:** **91/91 PASSING (100% CLEAN)** after routing `CausalStreamingEngine` to the authoritative fast ONNX inference backend.

---

## Audit Checklist: The 14 Mandatory Inquiries

### 1. Physical Hardware Available?
**NO.**  
Active scanning of PnP USB devices, ARP tables, hostname resolution (`ping raspberrypi.local`), and SSH endpoints confirmed that no Raspberry Pi 4, Raspberry Pi 5, or I2S Audio HAT is currently attached to or reachable from the host machine. **Phase 8 (Failure Conditions)** was strictly applied.

---

### 2. Exact Hardware
- **Host Benchmark Platform (Measured):**
  - **CPU:** Intel Core Ultra 7 155H (16 physical cores, 22 threads: 6 Performance Cores up to 4.8 GHz, 8 Efficient Cores, 2 Low-Power Efficient Cores)
  - **RAM:** 32 GB LPDDR5x
  - **Host OS:** Windows 11 Pro (64-bit, Build 26100)
  - **Runtimes:** Python 3.13.2, ONNX 1.22.0, ONNX Runtime 1.26.0 (CPUExecutionProvider), PyTorch 2.5.1+cpu, Numba 0.61.2
- **Target Embedded Hardware (Documented Specification):**
  - **SBC:** Raspberry Pi 4 Model B (Broadcom BCM2711, Quad-core Cortex-A72 ARMv8 64-bit @ 1.5 GHz, 32 KB L1 instruction cache per core, 1 MB shared L2 cache, ARM NEON SIMD) / Raspberry Pi 5 (BCM2712, Quad Cortex-A76 @ 2.4 GHz)
  - **Audio HAT:** Waveshare WM8960 Audio HAT (Cirrus Logic WM8960 Low-Power Stereo Codec, dual onboard MEMS microphones, 24-bit ADC/DAC @ 16 kHz, I2S digital audio bus, ALSA device `hw:1,0`)

---

### 3. ONNX Export Status
**SUCCESS.**  
The exact frozen checkpoint `checkpoints/E2_causal.pt` was exported to standalone FP32 ONNX:
- **Output Artifact:** `models/E2_causal.onnx` (and mirror at `deployment/rpi/models/E2_causal.onnx`)
- **Standalone Model Size:** 49,681 bytes (48.52 KB)
- **Model SHA-256:** `df7e79aa9369d12aee46473c7b3ecf6a455a7ae598cb72ab59f77f523cce8ecb`
- **ONNX Opset Version:** 17
- **Architecture Integrity:** Exactly 9,569 parameters. 4 Conv2D layers with asymmetric causal temporal padding (`kernel=(3, 3)`, causal left-pad $T=2$, right-pad $T=0$). Receptive field: 9 temporal past frames ($8 \times 8\text{ ms} = 64\text{ ms}$ causal history), 0 future lookahead.
- **Structural Integrity:** Validated via `onnx.checker.check_model(onnx_model)` with zero warnings or errors.

---

### 4. PyTorch-vs-ONNX Numerical Equivalence
**VERIFIED [VERIFIED EXPERIMENT].**  
Numerical equivalence between PyTorch eager execution and ONNX Runtime (`CPUExecutionProvider`) was tested across 7 test configurations using random and deterministic spectrogram inputs:

| Input Test Configuration | Input Tensor Shape | Max Absolute Error | Mean Absolute Error | Equivalence Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **$T=1$ (Single Streaming Frame)** | `(1, 1, 129, 1)` | **$5.96 \times 10^{-8}$** | $1.39 \times 10^{-8}$ | **PASSED** |
| **$T=9$ (Receptive Field Window)** | `(1, 1, 129, 9)` | **$1.79 \times 10^{-7}$** | $2.73 \times 10^{-8}$ | **PASSED** |
| **$T=32$ (Short Phrase)** | `(1, 1, 129, 32)` | **$1.79 \times 10^{-7}$** | $3.30 \times 10^{-8}$ | **PASSED** |
| **$T=64$ (Standard Block)** | `(1, 1, 129, 64)` | **$1.79 \times 10^{-7}$** | $3.46 \times 10^{-8}$ | **PASSED** |
| **$T=128$ (Long Block)** | `(1, 1, 129, 128)` | **$2.09 \times 10^{-7}$** | $3.47 \times 10^{-8}$ | **PASSED** |
| **$T=500$ (Full 4.0s Clip)** | `(1, 1, 129, 500)` | **$2.09 \times 10^{-7}$** | $3.54 \times 10^{-8}$ | **PASSED** |
| **Batch=4, $T=64$ (Multi-Stream)** | `(4, 1, 129, 64)` | **$1.94 \times 10^{-7}$** | $3.43 \times 10^{-8}$ | **PASSED** |

> **Conclusion:** Across all time horizons, the maximum absolute deviation is $\le 2.09 \times 10^{-7}$, well within standard IEEE 754 single-precision float rounding ($10^{-6}$).

---

### 5. Actual Embedded AI Latency
**UNMEASURED on Raspberry Pi [HYPOTHESIS].**  
On the host reference CPU under 1,000 steady-state iterations (hop budget = 8.0 ms):

| Engine / Mode | Input | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | RTF (P95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch FP32** | $T=1$ | 4.358 | 6.818 | 8.411 | 11.044 | 0.852 |
| **ONNX Runtime FP32** | $T=1$ | **0.237** | **0.340** | **0.422** | **0.505** | **0.042** |
| **ONNX Runtime FP32** | $T=9$ | 0.687 | 0.881 | 0.990 | 1.544 | 0.110 |
| **ONNX Runtime INT8** | $T=1$ | 4.266 | 5.797 | 6.001 | 9.537 | 0.725 |

> **Finding:** On CPU execution, ONNX Runtime FP32 achieves a **18.4× speedup over PyTorch** (0.237 ms vs 4.358 ms P50).

---

### 6. Actual Embedded DSP Latency
**UNMEASURED on Raspberry Pi [HYPOTHESIS].**  
On the host reference CPU under 1,000 steady-state iterations on 128-sample blocks (VSS-NLMS, filter length $M=64$):
- **Numba JIT Fast Path:**
  - P50 = **0.147 ms**
  - P95 = **0.232 ms**
  - P99 = **0.265 ms**
  - Max = **0.368 ms**
  - RTF (P95) = **0.0290** (uses 2.9% of the 8.0 ms budget)
- **Pure NumPy Fallback:**
  - P50 = **8.803 ms** (exceeds the 8.0 ms budget)
- **Critical Insight:** Python interpreter dispatch overhead on per-sample loops is prohibitive for real-time operation. Either Numba JIT compilation or an AOT C/C++ extension is strictly required on ARM Linux.

---

### 7. Actual End-to-End Latency
**UNMEASURED on Physical Hardware [ENGINEERING TARGET].**  

To prevent conceptual ambiguity during evaluation and SIH defense, the project explicitly freezes three separate latency definitions:

#### A. Frame / Hop Cadence: 8.0 ms
- At 16 kHz sampling rate, each 128-sample hop corresponds to $128 / 16000 = 8.0\text{ ms}$.
- This defines how frequently audio processing blocks are ingested and dispatched.

#### B. Algorithmic & Processing Latency: 0.0 ms Lookahead + 0.52 ms Computation
- **Algorithmic Lookahead:** Exactly **0.0 ms**. `E2_causal` uses strictly asymmetric causal temporal padding ($T=8$ past context, 0 future lookahead), introducing zero future buffering delay.
- **Host Software Processing Time ($\tau_{\text{DSP}} + \tau_{\text{AI}} + \tau_{\text{WOLA}}$):**
  - **P50:** **0.413 ms**
  - **P95:** **0.524 ms**
  - **P99:** **0.595 ms**
  - **Max:** **0.806 ms**
  - Consumes **6.55%** of the 8.0 ms hop budget (> 93% computational headroom on host).

#### C. Physical End-to-End Latency: Unmeasured on Hardware
$$\tau_{\text{System}} = \tau_{\text{ADC}} + \tau_{\text{Buffer\_In}} + \tau_{\text{DSP}} + \tau_{\text{AI}} + \tau_{\text{WOLA}} + \tau_{\text{Buffer\_Out}} + \tau_{\text{DAC}} + \tau_{\text{Acoustic}}$$
- **Acquisition & Buffering ($\tau_{\text{Buffer\_In}} + \tau_{\text{Buffer\_Out}}$):** Double 128-sample DMA buffering accounts for $\approx 16.0\text{ ms}$.
- **Hardware Converter Delay ($\tau_{\text{ADC}} + \tau_{\text{DAC}}$):** Codec sigma-delta filtering accounts for $\approx 0.5 - 1.0\text{ ms}$.
- **Status:** **UNMEASURED**. Requires physical oscilloscope acoustic loopback testing on the bench.

---

### 8. Audio I/O Status
**CODE COMPLETE / HARDWARE UNTESTED.**  
- **Implementation:** `src/audio_io/sources.py` (`ALSASource`) and `src/audio_io/sinks.py` (`ALSASink`) implement native ALSA I2S audio streaming for `hw:1,0` (WM8960 Audio HAT) at 16 kHz / 16-bit PCM stereo with `periodsize=128`.
- **Platform Handling:** Gracefully catches non-Linux / non-ALSA platforms and raises `HardwareUnavailableError` with clear diagnostic remediation.
- **Physical Status:** Untested on actual hardware due to absence of physical Pi board.

---

### 9. INT8 Quantization Status
**EVALUATED AND REJECTED FOR CPU DEPLOYMENT [VERIFIED EXPERIMENT].**  
Dynamic quantization was performed on `models/E2_causal.onnx` producing `models/E2_causal_int8.onnx` (SHA-256: `45b16dacbe09614d8d6973edcb88caf9427780c7d953dbf6edba923daa7981b5`).

Acoustic and performance comparison:
| Metric | FP32 ONNX | INT8 ONNX | Delta / Ratio | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Model Size** | **11.62 KB** (base) | 20.09 KB | **1.73× LARGER** | ✗ Fails size reduction |
| **Latency ($T=1$ P50)** | **0.237 ms** | 4.266 ms | **18.0× SLOWER** | ✗ Severe slowdown |
| **TEST_B SI-SDR** | **8.96 dB** | 8.96 dB | +0.002 dB | Identical |
| **TEST_B STOI** | **0.8668** | 0.8667 | -0.0001 | Identical |
| **TEST_C SI-SDR** | **11.56 dB** | 11.55 dB | -0.003 dB | Identical |
| **Clean Speech Attenuation** | **−0.9573 dB** | −0.9538 dB | +0.0035 dB | Identical |

> **Corrected Scientific Assessment:** ONNX Runtime dynamic INT8 quantization was evaluated and rejected for the current T=1 CPU deployment path because it increased latency substantially (4.27 ms vs 0.24 ms) and increased model size (20.6 KB vs 11.6 KB base) without meaningful quality benefit. Dynamic quantization overhead dominates tiny single-frame tensors ($1 \times 1 \times 129 \times 1$) on CPU. Static INT8 / TensorRT quantization remains untested and is not required unless hardware profiling on ARM reveals a need for it. With only 9,569 parameters (49.7 KB standalone FP32), parameter memory bandwidth is negligible, so **FP32 ONNX is the authoritative edge target.**

---

### 10. Any Regressions?
**ZERO REGRESSIONS (91/91 TESTS PASSING - 100% CLEAN).**  
1. **Acoustic Quality:**
   - Evaluated across 208 clips of `TEST_B` and 20 clips of `TEST_C`:
   - `TEST_B` SI-SDR: **8.96 dB**
   - `TEST_C` SI-SDR: **11.56 dB**
   - Clean Speech Attenuation: **−0.957 dB**
2. **Regression Test Suite:**
   - **91/91 passing tests (100% clean).**
   - The previously failing test `test_08_streaming_simulation` in `test_laptop_validation.py` was resolved by integrating `TinyEnhancerONNXWrapper` into `CausalStreamingEngine`. The streaming engine now executes the authoritative fast ONNX path on host, achieving **P95 = 3.83 ms** (comfortably within the 8.00 ms budget).

---

### 10B. Microphone Architecture & Acoustic ANC Distinction
**IMPORTANT ARCHITECTURAL CLARIFICATION FOR SIH / DRDO DEFENSE:**  
The current system operates on a dual-microphone frontend:
- **Primary Microphone (Ch0):** Captures noisy speech (desired speech signal $s[n]$ + acoustic noise $d[n]$).
- **Reference Microphone (Ch1):** Positioned away from the speaker's mouth to capture ambient noise reference $x[n]$.
- **Error Microphone:** **NOT YET IMPLEMENTED / NOT APPLICABLE TO STAGE 1.**

The current system is a **hybrid speech-enhancement / digital noise cancellation pipeline** (VSS-NLMS acoustic noise subtraction + causal deep spectral masking). It is **not yet equivalent to a closed-loop acoustic active noise control headset**, which requires:
1. An **Error Microphone** positioned inside the ear-cup cavity near the tympanic membrane.
2. An adaptive filter modeling the secondary acoustic path transfer function $S(z)$ between the anti-noise loudspeaker and the error microphone (e.g. via **Filtered-X NLMS / FxNLMS**).  
This distinction is strictly maintained to prevent overclaiming during evaluation.

---

### 11. What is VERIFIED EXPERIMENT?
The following are empirically measured, fully reproducible facts on disk:
1. Export of `E2_causal.pt` to standalone FP32 ONNX model `models/E2_causal.onnx` (49.7 KB standalone, SHA-256 `df7e79aa9369d12aee46473c7b3ecf6a455a7ae598cb72ab59f77f523cce8ecb`).
2. Exact numerical equivalence between PyTorch eager mode and ONNX Runtime: max absolute error $= 5.96 \times 10^{-8}$ ($T=1$) and $2.09 \times 10^{-7}$ ($T=500$).
3. Host CPU inference latency under ONNX Runtime: P50 = **0.237 ms** / P95 = **0.340 ms** (18.4× faster than PyTorch).
4. Host CPU fast DSP latency under Numba: P50 = **0.147 ms** / P95 = **0.232 ms**.
5. Host CPU combined software processing latency: P50 = **0.413 ms** / P95 = **0.524 ms** / Max = **0.806 ms**.
6. Full acoustic evaluation of FP32 ONNX across `TEST_B` (+8.96 dB) and `TEST_C` (+11.56 dB), with clean speech attenuation of −0.957 dB.
7. INT8 dynamic quantization performance penalty: 18× runtime slowdown and 1.73× file expansion.
8. Functional execution of standalone embedded deployment scripts: `deployment/rpi/deploy_e2_causal.py` and `deployment/rpi/benchmark_embedded.py`.
9. Clean regression suite status: **91/91 passing tests (100%)**.

---

### 12. What is ENGINEERING TARGET?
The following are design specifications to be confirmed on hardware:
1. Raspberry Pi 4 Model B (Cortex-A72 @ 1.5 GHz) single-frame FP32 ONNX inference latency: **P95 $\le$ 3.0 ms**.
2. Raspberry Pi 4 VSS-NLMS DSP latency (128 samples): **P95 $\le$ 1.5 ms**.
3. Raspberry Pi 4 combined software processing latency: **P95 $\le$ 5.0 ms** (leaving $\ge 37.5\%$ safety headroom within the 8.0 ms budget).
4. End-to-end hardware latency (acoustic wave at mic to acoustic wave at speaker): **$\le$ 18.0 ms** (2 double-buffered hops + converter delay).
5. Continuous operation without ALSA buffer under-run (XRUN) for $> 10$ minutes under `SCHED_FIFO` priority 80.

---

### 13. What is HYPOTHESIS?
The following are engineering projections based on host measurements that await physical verification:
1. That the 4 Cortex-A72 cores on the BCM2711 with NEON vector extensions will execute the 9,569-parameter ONNX model in under 3.0 ms per frame without requiring model pruning.
2. That Numba JIT (or an AOT C-extension) will achieve $\le 1.0\text{ ms}$ on ARM64 Linux for the 64-tap adaptive filter.
3. That the I2S clocks on the Waveshare WM8960 Audio HAT will maintain inter-channel phase synchronization ($\le 1$ sample skew) across extended capture sessions.

---

### 14. What Remains Before Claiming Embedded Real-Time Operation?
To satisfy scientific integrity and defend the system at the Smart India Hackathon (SIH 2026), the following physical experiments must be conducted on the bench:
1. **Physical Board Bring-Up (PH3.1):** Connect Raspberry Pi 4 / 5 and flash 64-bit Raspberry Pi OS Lite.
2. **Audio HAT Driver Verification (PH3.1):** Install WM8960 drivers and verify ALSA nodes (`arecord -l`, `aplay -l`).
3. **Basic Audio Streaming Loop (PH3.2):** Verify `Mic -> ALSA -> ALSA output` without AI to isolate pure audio I/O latency and stability.
4. **On-Device Benchmarking (PH3.3):** Run `python3 deployment/rpi/benchmark_embedded.py` to record true ARM Cortex hardware latency percentiles (P50, P95, P99, Max).
5. **Physical Loopback Latency Measurement (PH3.4):** Connect a dual-channel oscilloscope (Ch1: Primary Mic analog test pad; Ch2: DAC output analog test pad) and measure true acoustic-to-audio delay $\Delta t$.
6. **Thermal & Sustained Stability Stress Test:** Run continuous live dual-channel ANC for 15 minutes under simulated noise, monitoring CPU temperature (`vcgencmd measure_temp`) and throttling flags (`vcgencmd get_throttled`).

---

## Deployment Artifact Summary

| Artifact Path | Size | Description |
| :--- | :---: | :--- |
| `models/E2_causal.onnx` | 49.7 KB | Standalone FP32 ONNX model (embedded weights) |
| `models/E2_causal_int8.onnx` | 20.6 KB | Quantized INT8 ONNX model (archived / rejected) |
| `deployment/rpi/models/E2_causal.onnx` | 49.7 KB | Packaged model for Raspberry Pi target |
| `deployment/rpi/deploy_e2_causal.py` | 7.9 KB | Standalone embedded real-time streaming engine |
| `deployment/rpi/benchmark_embedded.py` | 9.0 KB | On-device latency benchmark script |
| `deployment/rpi/requirements.txt` | 410 B | Minimal embedded dependencies (no PyTorch) |
| `deployment/rpi/README_RPI_BRINGUP.md` | 3.5 KB | Step-by-step physical bring-up guide |
| `results/ph3_embedded/onnx_export_verification.json` | 2.3 KB | Full numerical equivalence test report |
| `results/ph3_embedded/onnx_fp32_vs_int8_report.json` | 3.5 KB | Full latency and acoustic comparison report |
