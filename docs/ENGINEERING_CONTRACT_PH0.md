# DRDO SIH 2026: Pre-Hardware Engineering Contract (PH0)
**Problem Statement ID:** 26052 — AI/ML-Enabled Adaptive Noise Cancellation for Defence Vehicles  
**Milestone:** PH0 Baseline Freeze  
**Status:** ACTIVE & FROZEN (2026-09-05)

---

## 1. Executive Summary & Purpose

This contract formalizes and freezes the algorithmic, physical, and architectural parameters of the Adaptive ANC system. In accordance with PH0 pre-hardware discipline, **these parameters are frozen and shall not be modified during downstream experiments**.

All algorithmic and software implementations must strictly adhere to the contracts defined herein.

---

## 2. Parameter Contract Table

| Parameter | Frozen Value | Engineering Rationale |
|---|---|---|
| **Sampling Frequency ($f_s$)** | **16,000 Hz** (16 kHz) | Defence standard voice communication rate (STANAG 4285 / ITU-T G.722 compatible). |
| **STFT FFT Size ($N_{\text{fft}}$)** | **512 samples** (32.0 ms) | Provides 257 discrete frequency bins with 31.25 Hz frequency resolution. |
| **Hop Size ($R$)** | **128 samples** (**8.0 ms**) | 75% window overlap. Crucial: enforces an 8.0 ms frame cadence for low-latency streaming. |
| **Analysis Window** | **512 samples** (Hann window) | Zero DC leakage, spectral side-lobe suppression &gt; 31 dB. |
| **Synthesis Window** | **512 samples** (Normalized OLA) | Exact perfect reconstruction under 75% overlap without energy gain. |
| **Frequency Bins ($F$)** | **257 bins** ($0 \text{ to } 8,000\text{ Hz}$) | Matches `TinyEnhancer` input dimensions natively without zero-padding or slicing. |
| **Hop Compute Budget** | **&le; 8.0 ms per hop** | Processing per hop (DSP + AI + iSTFT) must be strictly &lt; 8 ms to avoid ALSA audio underruns (XRUNs). |
| **Algorithmic Latency** | **32.0 ms** (16.0 ms with causal asymmetric window) | Governed by analysis window buffer length before overlap-add emission. |
| **Target Total Latency** | **&le; 20.0 ms target** (&le; 30.0 ms DRDO ceiling) | Frame latency + physical I/O DMA buffers on target audio codec. |
| **Target Hardware Platform** | **Raspberry Pi 4 Model B + Waveshare WM8960 Audio HAT** | Quad-core ARM Cortex-A72 @ 1.5 GHz, dual I2S MEMS mic input, stereo DAC output. |
| **Active Baseline AI Model** | **TinyEnhancer V3** (9,569 parameters) | 4-layer 2D Real-Valued ConvNet (input `[1, 1, 257, T]`, output real mask $M(f,t) \in [0, 1]$). |
| **Adaptive Filter Algorithm** | **VSS-NLMS** ($L=64$, $\mu_{\max}=0.10$, $\mu_{\min}=0.01$) | Variable step-size normalized LMS with cross-correlation speech leakage guard and crest-factor impulse freeze. |
| **Acoustic Operation Mode** | **Dual-Microphone Speech Enhancement** | Coarse acoustic noise removal via reference mic &rarr; fine spectral residual suppression via AI. **No loudspeaker acoustic wave cancellation claimed.** |

---

## 3. Latency & Budget Decomposition

Under the frozen 128-sample (8.0 ms) hop configuration:

```text
Time per Hop: 8.0 ms (128 samples @ 16,000 Hz)
├─ 1. STFT Analysis Framing:                ~0.25 ms
├─ 2. Reference TDE Delay Alignment:        ~0.15 ms
├─ 3. VSS-NLMS Filter & Error Computation:  ~0.30 ms
├─ 4. TinyEnhancer PyTorch/INT8 Inference:  ~1.80 ms (target on Cortex-A72)
├─ 5. Spectral Mask Application & iSTFT:    ~0.35 ms
├─ 6. RingBuffer Read/Write & ALSA DMA:     ~0.15 ms
└─ TOTAL PROCESSING PER HOP:               ~3.00 ms (Margin: 5.0 ms headroom under 8.0 ms hop)
```

> **Critical Distinction**:
> - **Hop Processing Time (~3.0 ms)**: Time required by CPU to compute one 128-sample block. Must be $< 8.0\text{ ms}$.
> - **Algorithmic Latency (16–32 ms)**: Time delay between a physical sound entering the microphone and being reconstructed in the output buffer due to STFT windowing.

---

## 4. Target Metric Requirements

The system performance will be benchmarked against the following formal targets:

| Metric | DRDO Requirement | Current Offline Reality | PH0 Gate Focus |
|---|---|---|---|
| **STOI** | &gt; 0.85 | **0.896 (PASS)** | Maintain intelligibility across all SNR levels. |
| **PESQ (Wideband)** | &gt; 2.50 | **1.50 (FAIL)** | Deep investigation of mask clipping & noisy phase preservation. |
| **Output SNR** | &gt; 15 dB | **Delta SNR +7.0 dB** | Measure absolute output SNR across -10 dB to +20 dB input range. |
| **Real-Time Factor (RTF)** | &lt; 0.50 | **~0.02 (PASS on PC)** | Profile computational budget for embedded ARM target. |
| **Causality** | Zero future lookahead | Strictly causal ($center=False$) | Future-perturbation anti-cheating audit. |
