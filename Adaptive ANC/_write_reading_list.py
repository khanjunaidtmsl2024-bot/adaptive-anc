import pathlib

content = r"""# PS 26052 - RESEARCH PAPER READING ORDER
## Adaptive Noise Cancellation: From First Principles to Defence-Grade Systems

**Purpose:** Structured reading list to build understanding from basics to advanced
**Organized:** Level-by-level, read in order within each level
**Current project stage:** You are here -> **Level 3-4** (concurrent with execution)

---

## LEVEL 0 - THE FOUNDATION (Start Here)

These papers define the field. Read all of them before anything else.

### 0.1 The Original - Adaptive Noise Cancellation

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **Adaptive Noise Cancelling: Principles and Applications** | Widrow, Glover, et al. | 1975 | THE paper that started everything. Defines the two-microphone adaptive filter framework. Read Sections I-III carefully. This is where d[n], x[n], e[n], y[n] come from. |
| **Adaptive Signal Processing (textbook, Ch. 6-8)** | Widrow and Stearns | 1985 | The textbook companion. Chapters on LMS derivation, convergence proof, and noise cancellation applications. |

**What you should understand after this:**
- What d[n] (primary), x[n] (reference), y[n] (output), e[n] (error) mean
- Why LMS works as a noise canceller
- What convergence means and why step size matters
- The physical meaning of the adaptive filter coefficients

**Source:** Widrow et al., Proc. IEEE, Vol. 63, No. 12, Dec 1975
**Available:** IEEE Xplore, widely cited (>15,000 citations)

---

### 0.2 The Normalized Variant - NLMS

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **Adaptive Filter Theory (Ch. 6)** | Haykin | 2013 | NLMS normalizes step size by input power. Robust to signal level changes. |
| **Acoustic Noise Cancellation by NLMS and RLS** | Abhishek and Ramesh | 2020 | Practical NLMS vs RLS comparison on real audio. |

**What you should understand:**
- NLMS update rule: w(n+1) = w(n) + [mu / (eps + ||x(n)||^2)] * e(n) * x(n)
- Why normalization helps with non-stationary signals
- When NLMS beats LMS and when it does not

---

### 0.3 Spectral Subtraction - The Classical Baseline

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **Spectral Subtraction** | Boll, S.F. | 1979 | The original. Simple, fast, produces musical noise artifacts. Your baseline to beat. |
| **Speech Enhancement Based on Noise Reduction** | Singh and Gu | 2014 | Student-level overview comparing LMS, NLMS, spectral subtraction. |

---

## LEVEL 1 - DSP FUNDAMENTALS

### 1.1 STFT and Time-Frequency Processing

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **Discrete-Time Signal Processing** (Ch. on DFT/STFT) | Oppenheim and Schafer | 2010 | Math foundation. Windowing, overlap, and reconstruction. |
| **Window Functions for STFT-based Speech Enhancement** | Various | 2020+ | Why Hann, Hamming windows matter for speech. |

**What you should understand:**
- Why we window before FFT
- Hop size and FFT size
- iSTFT reconstruction, OLA/NOLA conditions
- Algorithmic latency = FFT size - hop size + inference time

---

### 1.2 FxLMS - Active Noise Control (Physical Anti-Noise)

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **Active Noise Control: A Tutorial Review** | Kuo and Morgan | 1996 | THE survey for physical ANC. Secondary path model, FxLMS. |
| **FxLMS-based ANC: A Quick Review** | Ardekani and Sharif-Bakhtiar | 2011 | Shorter modern review of FxLMS. |
| **Active Noise Control Using FxLMS** (MathWorks) | MathWorks | Current | Practical implementation with code. |

**What you should understand:**
- Feedforward vs feedback ANC
- Why you need a secondary-path model S(z)
- Why full physical ANC is a Phase 2 problem

---

## LEVEL 2 - DEEP LEARNING FOR SPEECH ENHANCEMENT

### 2.1 Neural Network Fundamentals for Audio

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **Deep Learning for Monaural Speech Enhancement** | Various survey | 2019-2021 | How CNNs process spectrograms. Masks (binary, complex, ratio). |
| **Deep Noise Suppression Challenge (DNS Challenge)** | Microsoft | 2020 | Benchmark challenge. Also your **dataset source**. |

---

### 2.2 Model Architectures - The Candidates

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **DeepFilterNet2: Real-Time SE on Embedded Devices** | Schroeter et al. | 2022 | PRIMARY MODEL. Two-stage: ERB + deep filtering. RTF 0.04. YOUR MAIN CANDIDATE. |
| **DCCRN: Deep Complex Convolution Recurrent Network** | Hu et al. | 2020 | Complex-domain processing. Why phase matters. |
| **DTLN: Dual-signal Transformation LSTM Network** | Westhausen and Vikter | 2019 | Lightweight alternative. Embedded benchmark. |
| **RNNoise: Real-Time Noise Suppression** | Valin (Xiph.org) | 2017 | Extremely lightweight. Zero-GPU option. |
| **Compact CRN for Monaural SE** | Chuan et al. | 2020 | Build-your-own-model reference. |

---

### 2.3 DeepFilterNet Deep Dive

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **DeepFilterNet2 (full paper)** | Schroeter et al. | 2022 | Read Sections III-V. ERB envelope + deep filtering. |
| **DeepFilterNet3** | Schroeter et al. | 2023 | Full-band extension. |
| **pDeepFilterNet2 (personalized)** | Wang et al. | 2024 | Personalize DFN for specific noise types - relevant to defence noise. |

---

## LEVEL 3 - HYBRID SYSTEMS (Where Your Project Lives)

### 3.1 AI + Adaptive Filter Hybrids

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **Hybrid SFANC-FxNLMS for ANC Based on Deep Learning** | Luo et al. | 2022 | DIRECTLY RELEVANT. NN selects filter, FxLMS adapts online. Closest to Config A. |
| **HAD-ANC: Hybrid Adaptive Filter + DNNs** | Park et al. (Interspeech) | 2023 | DNN + adaptive filter. 12 citations. Validates hybrid. |
| **Latent FxLMS: Accelerating ANC with Deep Learning** | arXiv | 2025 | Cutting-edge: latent representations speed up FxLMS. |

**What you should understand:**
- WHY hybrids exist: AI = noise classification, adaptive filters = online adaptation
- The convergence problem: classical FxLMS is slow, AI helps
- Config A (NLMS -> AI) is simpler but needs experimental validation

---

### 3.2 AI + DSP for Defence/Military

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **AI Driven Advances in Noise Cancellation for Military Operations** | Narain, Singh et al. (DRDO) | 2026 | CRITICAL. Defence Science Journal. From DRDO itself. |
| **Neural Network-Based ANC Algorithms: A Review** | Liu et al. | 2025 | Survey of neural networks applied to ANC (2020-2025). |
| **Deep Learning Approaches to Selective Noise Cancellation** | arXiv 2507.07043 | 2025 | Systematic review of AI-driven selective noise cancellation. |

---

## LEVEL 4 - COMPETITIVE CHALLENGES AND BENCHMARKS

### 4.1 DNS Challenge (Your Dataset Source)

| Resource | Why Read It |
|----------|-------------|
| **ICASSP 2021 DNS Challenge** | Dataset generation scripts, clean/noise/RIR. Your dataset starter stack. |
| **DNS Challenge Leaderboard** | State-of-the-art PESQ/STOI/SI-SNR numbers. |

### 4.2 URGENT Challenge (Universal Speech Enhancement)

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **Interspeech 2025 URGENT Challenge** | Saijo et al. | 2025 | Latest universal SE challenge. Current frontier. |
| **ICASSP 2026 URGENT Challenge** | Li et al. | 2026 | What SOTA looks like in 2026. |

### 4.3 Where Your Project Fits

| Dimension | DNS SOTA | URGENT | Your Target |
|-----------|----------|--------|-------------|
| Noise types | General | Universal | Defence + impulsive |
| Real-time | Some | No | **Required** |
| Two-microphone | No | No | **Yes (NLMS)** |
| Impulsive handling | No | No | **Yes (controller)** |
| Embedded | Some | No | **Required** |
| PESQ | >3.0 | Varies | >2.5 |
| STOI | >0.90 | Varies | >0.85 |

**Your novelty:** Nobody in DNS/URGENT does defence-specific impulsive noise + two-mic hybrid + embedded
