# PS 26052 — RESEARCH PAPER READING ORDER
## Adaptive Noise Cancellation: From First Principles to Defence-Grade Systems

**Purpose:** Structured reading list to build understanding from basics to advanced
**Organized:** Level-by-level, read in order within each level
**Current project stage:** You are here → **Level 3-4** (concurrent with execution)

---

## LEVEL 0 — THE FOUNDATION (Start Here)

These papers define the field. Read all of them before anything else.

### 0.1 The Original — Adaptive Noise Cancellation

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **"Adaptive Noise Cancelling: Principles and Applications"** | Widrow, Glover, et al. | 1975 | **THE paper that started everything.** Defines the two-microphone adaptive filter framework. Read Sections I-III carefully. This is where d[n], x[n], e[n], y[n] come from. |
| **"Adaptive Signal Processing" (textbook, Ch. 6-8)** | Widrow & Stearns | 1985 | The textbook companion. Chapters on LMS derivation, convergence proof, and noise cancellation applications. The math proof that LMS converges. |

**What you should understand after this:**
- What d[n] (primary), x[n] (reference), y[n] (output), e[n] (error) mean
- Why LMS works as a noise canceller
- What "convergence" means and why step size matters
- The physical meaning of the adaptive filter coefficients

**Source:** Widrow et al., Proc. IEEE, Vol. 63, No. 12, Dec 1975
**Available:** IEEE Xplore, widely cited (>15,000 citations)

---

### 0.2 The Normalized Variant — NLMS

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **"A Modified LMS Algorithm Based on Orthogonal Projections"** (establishes NLMS foundation) | Haykin | 2013 (textbook) | NLMS normalizes step size by input power. This makes it robust to signal level changes — essential for real audio. |
| **"Acoustic Noise Cancellation by NLMS and RLS Algorithms: A Comparative Study"** | Abhishek & Ramesh | 2020 | Practical comparison of NLMS vs RLS on real audio. Shows when NLMS is sufficient and when you need RLS. |

**What you should understand after this:**
- The NLMS update rule: w(n+1) = w(n) + [μ / (ε + ||x(n)||²)] · e(n) · x(n)
- Why normalization helps with non-stationary signals
- When NLMS beats LMS and when it doesn't
- The trade-off: NLMS = stability, RLS = faster convergence

---

### 0.3 Spectral Subtraction — The Classical Baseline

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **"Spectral Subtraction — A Survey of Modern Methods"** | Boll, S.F. | 1979 | The original spectral subtraction. Simple, fast, produces musical noise artifacts. This is your baseline to beat. |
| **"Speech Enhancement Based on Noise Reduction"** | Singh & Gu (Rochester) | 2014 | Good student-level overview comparing LMS, NLMS, spectral subtraction. Short and readable. |

**What you should understand after this:**
- How FFT-based noise estimation works
- Why spectral subtraction creates "musical noise"
- The difference between spectral subtraction (frequency domain) and LMS/NLMS (time domain)
- Why neither is sufficient alone for defence noise

---

## LEVEL 1 — DSP FUNDAMENTALS (Build Your Math)

### 1.1 STFT and Time-Frequency Processing

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **"A High-Resolution Time-Frequency Distribution Based on the STFT"** | Oppenheim & Schafer (textbook) | 2010 | **"Discrete-Time Signal Processing"** — Chapter on DFT/STFT. This is your math foundation. You cannot do speech enhancement without understanding windowing, overlap, and reconstruction. |
| **"A Review of Window Functions for STFT-based Speech Enhancement"** | Various | 2020+ | Explains why Hann, Hamming, and other windows matter for speech. |

**What you should understand after this:**
- Why we window before FFT
- What hop size and FFT size mean
- How iSTFT reconstructs audio from STFT
- The overlap-add (OLA) and NOLA conditions
- Algorithmic latency = FFT size - hop size + inference time

---

### 1.2 FxLMS — Active Noise Control (Physical Anti-Noise)

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **"Active Noise Control: A Tutorial Review"** | Kuo & Morgan | 1996 | THE survey paper for physical ANC. Explains why you need a secondary path model and how FxLMS modifies LMS to account for it. |
| **"FxLMS-based Active Noise Control: A Quick Review"** | Ardekani & Sharif-Bakhtiar | 2011 | Shorter, more modern review. Good for understanding the secondary path problem. |
| **"Active Noise Control Using a Filtered-X LMS FIR Adaptive Filter"** (MathWorks example) | MathWorks | Current | Practical implementation guide with code. Shows the exact signal flow for FxLMS. |

**What you should understand after this:**
- The difference between feedforward and feedback ANC
- Why you need a secondary-path model (S(z))
- What filtered-x means and why it's necessary
- Why full physical ANC is a Phase 2 problem for your project
- How this relates to your NLMS-before-AI architecture

---

## LEVEL 2 — DEEP LEARNING FOR SPEECH ENHANCEMENT (The AI Layer)

### 2.1 Neural Network Fundamentals for Audio

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **"A Deep Learning Approach to Monaural Speech Enhancement"** (DCUNet/FullSubNet context) | Various survey | 2019-2021 | How CNNs process spectrograms. The shift from hand-crafted features to learned representations. |
| **"Deep Noise Suppression Challenge" (DNS Challenge)** | Microsoft | 2020 | The benchmark challenge that spawned most modern SE models. Read the overview paper and baseline description. DNS-Challenge is also your **dataset source**. |

**What you should understand after this:**
- How neural networks see spectrograms
- What a mask is (binary mask, complex mask, ratio mask)
- What SI-SNR loss means and why it's used
- The difference between time-domain and frequency-domain models

---

### 2.2 Model Architectures — The Candidates

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **"DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices"** | Schröter, Escalante, et al. | 2022 | **PRIMARY MODEL PAPER.** Two-stage: ERB envelope + deep filtering. RTF 0.04 on laptop. Designed for embedded. THIS IS YOUR MAIN CANDIDATE. |
| **"DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement"** | Hu et al. | 2020 | Complex-domain processing. Your main comparison model. Shows why phase matters. |
| **"DTLN: Dual-signal Transformation LSTM Network for Real-Time Speech Enhancement"** | Westhausen & Vik台er | 2019 | Lightweight alternative. Good benchmark for embedded deployment. |
| **"RNNoise: Real-Time Noise Suppression"** | Valin, Jean-Marc (Xiph.org) | 2017 | Extremely lightweight. Hornschur/Mozilla context. Good for "can we do it with zero GPU?" |
| **"Condern-UNet for Monaural Speech Enhancement"** | Chuan et al. | 2020 | Compact CRN architecture. Your "build your own model" reference. |

**What you should understand after this:**
- Why DeepFilterNet is the leading candidate (ERB + deep filtering + embedded-focused)
- The difference between magnitude-only, complex, and time-domain processing
- What "causal" means for real-time inference
- Model size vs quality trade-offs

---

### 2.3 DeepFilterNet Deep Dive (Read Thoroughly)

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **"DeepFilterNet2" (full paper)** | Schröter et al. | 2022 | Read Sections III-V carefully. The two-stage architecture: Stage 1 = ERB envelope enhancement, Stage 2 = complex deep filtering for periodic components. |
| **"DeepFilterNet3"** | Schröter et al. | 2023 | Extension to full-band audio. Better quality but slightly higher compute. |
| **"pDeepFilterNet2"** (personalized DFN) | Wang et al. | 2024 | Shows how to personalize/fine-tune DFN for specific noise types — directly relevant to defence noise. |

**What you should understand after this:**
- The ERB (Equivalent Rectangular Bandwidth) filter bank and why it is perceptually motivated
- How deep filtering differs from standard masking
- The RTF measurement methodology
- What you would need to modify for 48 kHz vs 16 kHz

---

## LEVEL 3 - HYBRID SYSTEMS (Where Your Project Lives)

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

**Your novelty:** Nobody in DNS/URGENT does defence-specific impulsive noise + two-mic hybrid + embedded.

---

## LEVEL 5 - EMBEDDED AND REAL-TIME

### 5.1 Real-Time Speech Enhancement

| Paper | Year | Why Read It |
|-------|------|-------------|
| **Real-Time Noise Suppression Using Deep Learning** (NVIDIA) | 2018 | Frame-by-frame inference, streaming state, latency. |
| **Deep-Learning Framework for Efficient Real-Time SE** | 2025 | Updated streaming architecture. |

### 5.2 Embedded AI Deployment

| Resource | Why Read It |
|----------|-------------|
| **ONNX Runtime documentation** | Export and run models on edge |
| **NVIDIA TensorRT documentation** | INT8/FP16 for Jetson |
| **DeepFilterNet GitHub README** | On-device deployment instructions |
| **pDeepFilterNet2** (Wang 2024) | Make DFN2 smaller and personalized |

---

## LEVEL 6 - ADVANCED TOPICS

### 6.1 Impulsive Noise

| Paper | Year | Why Read It |
|-------|------|-------------|
| **Adaptive Kalman Filter for Impulsive Noise** (Liu) | 2025 | Kalman filter for impulse noise. Detector design reference. |
| **MAD Military Audio Dataset** | 2023+ | Defence acoustic dataset. Gunshots, artillery, rotor. |
| **C3GD (Gunshot Dataset)** | Various | Controlled impulsive recordings. |

### 6.2 Evaluation Methodology

| Resource | Why Read It |
|----------|-------------|
| **PESQ standard (ITU-T P.862)** | How PESQ is defined. Sample rate requirements. |
| **STOI paper (Taal et al., 2010)** | What STOI > 0.85 actually means. |
| **SI-SNR definition (Le Roux et al., 2018)** | Scale-invariant SNR. Better than plain SNR. |
| **NOISEX-92** | Classic noise database. Held-out test noise. |

---

## READING SCHEDULE

| Week | Level | Papers | Time |
|------|-------|--------|------|
| Week 1, Day 1 | 0 | Widrow 1975 (Sections I-III) | 2 hours |
| Week 1, Day 2 | 0 | NLMS basics (Haykin Ch. 6) | 2 hours |
| Week 1, Day 3 | 0 | Spectral subtraction (Boll 1979) | 1 hour |
| Week 1, Day 4 | 1 | STFT/Oppenheim Ch. on DFT | 2 hours |
| Week 1, Day 5 | 1 | FxLMS (Kuo and Morgan) | 2 hours |
| Week 2 | 2 | DeepFilterNet2 full paper | 3 hours |
| Week 2 | 2 | DNS Challenge overview | 1 hour |
| Week 3 | 3 | SFANC-FxNLMS hybrid | 2 hours |
| Week 3 | 3 | DRDO Narain 2026 | 2 hours |
| Week 3 | 3 | HAD-ANC (Interspeech 2023) | 1 hour |
| Week 4 | 4-5 | DNS leaderboard + URGENT | 2 hours |
| Week 4 | 5 | NVIDIA real-time SE + DFN README | 2 hours |
| Ongoing | 6 | As needed during implementation | - |

---

## MUST READ (Directly affects architecture decisions)

1. **Widrow 1975** - defines the adaptive cancellation framework
2. **DeepFilterNet2 (2022)** - your primary AI model
3. **SFANC-FxNLMS (Luo 2022)** - closest published hybrid to your Config A
4. **DRDO Narain 2026** - your problem statement issuer own research
5. **HAD-ANC (Park 2023)** - validates the hybrid DNN+adaptive approach

## SHOULD READ (Informs implementation)

6. Kuo and Morgan (1996) - FxLMS and secondary path
7. DNS Challenge - dataset generation methodology
8. Boll 1979 - spectral subtraction baseline
9. Haykin Ch. 6 - NLMS convergence proof
10. DCCRN (Hu 2020) - phase-aware processing comparison

## NICE TO READ (Deeper context)

11. URGENT Challenge - state-of-the-art benchmarking
12. Liu 2025 - neural network ANC survey
13. Wang 2024 - personalized DeepFilterNet
14. Liu 2025 - Kalman filter for impulsive noise
15. Oppenheim and Schafer - DSP textbook

---

## WHERE TO FIND THESE PAPERS

| Source | Papers |
|--------|--------|
| **arXiv.org** | DeepFilterNet2, SFANC-FxNLMS, URGENT Challenge, all recent |
| **IEEE Xplore** | Widrow 1975, Kuo and Morgan, DNS Challenge, ICASSP/Interspeech |
| **DRDO Publications** | Narain 2026 (publicationsdrdo.in) |
| **GitHub repos** | DeepFilterNet, DNS Challenge, SFANC-FxNLMS implementations |
| **Semantic Scholar** | Free access, good citation graphs |
| **Google Scholar** | Start here - finds free versions |

---

## NOTE ON YOUR EXISTING PAPERS

Your folder already contains:
- deepfilternet2.pdf - The DeepFilterNet2 paper (Paper #2 above)
- DRDO_ANC_Deep_Research most updated.pdf - Your research synthesis

The 12 DOCX files are your own research iterations. Download the actual papers listed above.

---

**Rule:** Read Level 0-1 in Week 1 before writing any code. Read Level 2-3 during first two weeks. Read Level 4-6 as needed.

**The team that understands Widrow 1975 before touching DeepFilterNet will make better architectural decisions than the team that skips to the shiny neural network paper.**
