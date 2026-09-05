# Laptop Testing vs. Embedded Hardware Boundaries
## Adaptive-Defence ANC — Smart India Hackathon (SIH) 2026 (PS ID: 26052)
**DRDO / Department of Defence Production / iDEX — Smart Vehicles**

---

## Executive Principle

> **"Laptop testing can prove that your algorithm works. It cannot prove that your embedded hardware implementation works."**

A fundamental engineering failure mode in edge audio and Active Noise Control (ANC) projects is confusing mathematical/simulated execution with physical hardware readiness. The table below formalizes the non-negotiable boundary of what our laptop test suite verifies versus what strictly requires the physical Raspberry Pi 4 + WM8960 DAC/ADC HAT.

---

## The Boundary Matrix

| Engineering Question | Laptop Validation? | Physical Hardware (Pi 4 + WM8960)? | Method of Laptop Proof | Reason Laptop Cannot Prove Hardware |
|---|:---:|:---:|---|---|
| **Does NLMS mathematically converge?** | **YES** | YES | Known correlated noise at 0 dB SNR; error energy decreases monotonically. | N/A (Mathematical property). |
| **Does AI model improve noisy speech?** | **YES** | YES | Objective metrics ($\Delta\text{SNR}$, SI-SDR, STOI, PESQ) on clean/noisy mixtures. | N/A (Algorithmic function). |
| **Does Hybrid beat AI-only & NLMS-only?** | **YES** | YES | Paired ablation testing ($N=30$) and hypothesis testing ($p < 0.05$). | N/A (Algorithmic function). |
| **Does the filter handle impulsive noise?** | **YES** | YES | Energy spike detection and post-impulse recovery time measurement. | N/A (Signal processing logic). |
| **Does the filter track non-stationary noise?** | **YES** | YES | Multi-regime sequence (engine &rarr; broadband &rarr; tonal &rarr; burst). | N/A (Adaptive tracking logic). |
| **Is the pipeline strictly causal?** | **YES** | YES | Future mutation test ($t > T_0$ mutated; $|y_A - y_B| = 0$ for $t \le T_0$). | N/A (Data dependency property). |
| **Does hop-by-hop streaming work?** | **YES** | YES | 128-sample ring buffer simulation with state continuity across hops. | N/A (Buffer architecture). |
| **Algorithmic compute profiling?** | **YES** | PARTIAL | Per-stage Python/Torch timing, FLOPs count, and memory allocation. | Laptop CPU (x86_64, 4+ GHz) $\ne$ Pi 4 (ARM Cortex-A72, 1.5 GHz). |
| **Physical end-to-end audio latency?** | **NO** | **YES** | *Cannot be tested on laptop.* | Requires physical I2S DMA transfers, WM8960 FIFO buffers, and ALSA ringbuffers. |
| **Physical WM8960 ADC/DAC latency?** | **NO** | **YES** | *Cannot be tested on laptop.* | Hardware converter group delay ($~0.5\text{--}1.0\text{ ms}$). |
| **Raspberry Pi 4 CPU load & thermal throttling?** | **NO** | **YES** | *Cannot be tested on laptop.* | Cortex-A72 cache hierarchy, memory bus bandwidth, and passive thermal dynamics. |
| **ALSA buffer underruns (XRUNs)?** | **NO** | **YES** | *Cannot be tested on laptop.* | Linux kernel scheduling jitter, PREEMPT_RT kernel latency, and real-time audio threads. |
| **I2S clock synchronization & jitter?** | **NO** | **YES** | *Cannot be tested on laptop.* | Physical BCLK/LRCK clocks between BCM2711 and WM8960 codec. |
| **Physical dual-mic acoustic isolation?** | **NO** | **YES** | *Cannot be tested on laptop.* | Physical distance, microphone polar patterns, acoustic baffling, and head shadow effect. |
| **Acoustic secondary-path FxLMS?** | **NO** | **YES** | *Cannot be tested on laptop.* | Physical speaker-to-ear acoustic impulse response $S(z)$. |
| **Physical in-ear acoustic noise reduction?** | **NO** | **YES** | *Cannot be tested on laptop.* | Decibels of sound pressure level (SPL) reduction measured inside an artificial ear/coupler. |

---

## The 8 Laptop Verification Protocols (PH0-A to PH0-H)

### Test 1 (PH0-A): Known Noise Cancellation
- **Goal:** Confirm classical adaptive filter convergence on clean synthetic noise before any AI complexity.
- **Input:** Speech + 0 dB pure tonal sine wave / white Gaussian noise. Reference mic = correlated noise.
- **Criterion:** NLMS residual must show monotonic error energy reduction ($> 3.0\text{ dB } \Delta\text{SNR}$).

### Test 2 (PH0-B): Quantitative Improvement
- **Goal:** Comprehensive quantitative evaluation of speech quality.
- **Metrics Calculated:**
  - $\text{SNR}_{\text{in}} = 10\log_{10}(P_{\text{speech}} / P_{\text{noise}})$
  - $\text{SNR}_{\text{out}} = 10\log_{10}(P_{\text{speech}} / P_{\text{residual}})$
  - $\Delta\text{SNR} = \text{SNR}_{\text{out}} - \text{SNR}_{\text{in}}$
  - SI-SDR, STOI, PESQ.
- **Systems Compared:** Clean, Noisy, NLMS, AI-Only, Hybrid (`NLMS -> AI`).

### Test 3 (PH0-C): Isolated AI Contribution
- **Goal:** Verify that the neural network provides genuine incremental improvement rather than coasting on NLMS output.
- **Criterion:** Hybrid $\Delta\text{SI-SDR} > \text{NLMS-only } \Delta\text{SI-SDR}$ and Hybrid STOI $\ge \text{NLMS-only STOI}$.

### Test 4 (PH0-D): Reference-Mic Speech Leakage Stress Test
- **Model:** $x[n] = n_r[n] + \alpha s[n]$
- **Sweep Range:** $\alpha \in [0.0, 0.01, 0.05, 0.10, 0.20, 0.30]$.
- **Target:** Determine the engineering threshold $\alpha_{\max}$ beyond which speech distortion becomes unacceptable ($STOI < 0.70$).

### Test 5 (PH0-E): Impulsive Noise & Filter Recovery Time
- **Input:** Continuous background noise ($0\text{ dB}$) with high-energy $+40\text{ dB}$ gunfire spike.
- **Measurements:** Post-impulse filter recovery time (ms to return within $1\text{ dB}$ of baseline error floor) and peak transient overshoot.

### Test 6 (PH0-F): Non-Stationary Noise Tracking
- **Input:** 10-second multi-regime sequence:
  - 0–2 s: Low-frequency engine rumble (120 Hz + harmonics)
  - 2–4 s: Broadband pink noise
  - 4–6 s: High-frequency tonal whistle (2.4 kHz)
  - 6–8 s: Artillery impulse burst
  - 8–10 s: Broadband pink noise
- **Observations:** Adaptive step-size $\mu(t)$, error energy, and coefficient trajectory tracking $w_i(t)$.

### Test 7 (PH0-G): Strict Anti-Cheating Causality Audit
- **Protocol:** Process signal $s_1[n]$ up to time $T_0$. Create $s_2[n]$ where samples for $t > T_0$ are completely replaced with high-amplitude noise.
- **Pass Criterion:** $\max_{t \le T_0} |y_1[t] - y_2[t]| < 10^{-5}$.

### Test 8 (PH0-H): Real-Time Hop-by-Hop Streaming Simulation
- **Architecture:** 128-sample ingestion hops ($8.0\text{ ms}$ real-time budget), maintaining internal state across frames.
- **Metrics:** P50, P95, P99, and MAX processing times per hop.
- **Budget Target:** Laptop frame time $< 8.0\text{ ms}$ (design target; hardware validation deferred to Pi 4).

---

## Defensive Presentation Guidance for SIH 2026 Jury

When presenting our results:
1. **Never say:** *"We have achieved 20 dB real-time acoustic noise cancellation on physical hardware."*
2. **Always say:** *"Our laptop streaming simulation proves our hybrid algorithm achieves $+7.0\text{ dB } \Delta\text{SNR}$ and $+13.2\text{ dB}$ SI-SDR in causal hop-by-hop execution within our $8.0\text{ ms}$ computational budget. Physical acoustic cancellation and ALSA latency will be measured on our Raspberry Pi 4 + WM8960 testbench in Phase 1."*
3. **Show the X-Ray graphs:** Demonstrate deep understanding through convergence curves, coefficient tracking, spectrograms, and AI suppression masks.
