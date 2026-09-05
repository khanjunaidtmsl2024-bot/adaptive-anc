# PAPER LIBRARY INDEX
## PS 26052 — Adaptive Defence ANC Research Papers

**Location:** `papers/` folder
**Total papers:** 17
**Status:** All verified as valid PDFs

---

## READING PRIORITY

### MUST READ (Architecture decisions depend on these)

| # | File | Paper | Authors | Year | Level | Why |
|---|------|-------|---------|------|-------|-----|
| 0 | `00_DeepFilterNet2_Schroeter_2022_original.pdf` | DeepFilterNet2: Towards Real-Time SE on Embedded Devices | Schroeter, Escalante et al. | 2022 | L2 | PRIMARY MODEL. Your main AI candidate. |
| 1 | `01_SFANC_FxNLMS_Luo_2022.pdf` | Hybrid SFANC-FxNLMS for ANC Based on Deep Learning | Luo et al. | 2022 | L3 | Closest published hybrid to your Config A. |
| 7 | `07_DRDO_AI_ANC_Narain_2026.pdf` | AI Driven Advances in Noise Cancellation for Military Operations | Narain, Singh (DRDO) | 2026 | L3 | From DRDO itself. Your problem statement issuer. |
| 8 | `08_HAD_ANC_Park_2023.pdf` | HAD-ANC: Hybrid Adaptive Filter + Deep Neural Networks | Park et al. | 2023 | L3 | Validates hybrid DNN+adaptive approach. |
| 13 | `13_DeepFilterNet_Schroeter_2022.pdf` | DeepFilterNet2 (arXiv version) | Schroeter et al. | 2022 | L2 | Backup copy of primary model paper. |

### SHOULD READ (Informs implementation choices)

| # | File | Paper | Authors | Year | Level | Why |
|---|------|-------|---------|------|-------|-----|
| 2 | `02_DeepFilterNet3_Schroeter_2023.pdf` | DeepFilterNet3: Low Complexity SE for Full-Band Audio | Schroeter et al. | 2023 | L2 | Full-band extension. Better quality. |
| 3 | `03_DCCRN_Hu_2020.pdf` | DCCRN: Deep Complex Convolution Recurrent Network | Hu et al. | 2020 | L2 | Complex-domain comparison model. Phase matters. |
| 4 | `04_DTLN_Westhausen_2020.pdf` | DTLN: Dual-signal Transformation LSTM Network | Westhausen, Meyer | 2020 | L2 | Lightweight embedded benchmark. |
| 5 | `05_RNNoise_Valin_2017.pdf` | RNNoise: A Hybrid DSP/Deep Learning Approach | Valin (Xiph.org) | 2017 | L2 | Ultra-lightweight. Runs on Raspberry Pi. |
| 11 | `11_Selective_NC_Review_2025.pdf` | Deep Learning Approaches to Selective Noise Cancellation | Various | 2025 | L3 | Systematic review of AI-driven SNC. |
| 15 | `15_Deep ANC_Zhang_2021.pdf` | Deep ANC: A Deep Learning Approach to ANC | Zhang et al. | 2021 | L3 | Supervised learning formulation of ANC. |

### NICE TO READ (Deeper context and cutting-edge)

| # | File | Paper | Authors | Year | Level | Why |
|---|------|-------|---------|------|-------|-----|
| 6 | `06_Latent_FxLMS_Sarkar_2025.pdf` | Latent FxLMS: Accelerating ANC with Neural Adaptive Filters | Sarkar et al. | 2025 | L3 | Cutting-edge: latent representations + FxLMS. |
| 9 | `09_pDeepFilterNet2_Serre_2024.pdf` | Lightweight Dual-Stage Personalized SE Based on DFN2 | Serre et al. | 2024 | L2 | How to personalize DFN for specific noise types. |
| 10 | `10_Kalman_Impulsive_Liu_2025.pdf` | Adaptive Kalman Filter-Based Impulsive Noise Suppression | Liu et al. | 2025 | L6 | Kalman approach for impulse noise. Detector reference. |
| 12 | `12_GFANC_FxNLMS_Luo_2026.pdf` | Stabilized Hybrid GFANC-FxNLMS Algorithm | Luo et al. | 2026 | L3 | Latest hybrid: generalized FANC + FxNLMS. |
| 14 | `14_URGENT_Challenge_2025.pdf` | Interspeech 2025 URGENT Speech Enhancement Challenge | Saijo et al. | 2025 | L4 | Current state-of-the-art benchmark. |
| 16 | `16_DRDO_Research_Synthesis.pdf` | DRDO ANC Research Synthesis | Your team | 2026 | - | Your own research document. Reference. |

---

## PAPERS BY LEVEL (Match to Reading Schedule)

### Level 0 — Foundation (Week 1)
- **Widrow 1975** — NOT DOWNLOADED (IEEE paywalled). Read on Semantic Scholar or IEEE Xplore with institutional access.
- **Haykin Ch. 6** — Textbook chapter. Not a standalone PDF.
- **Boll 1979** — NOT DOWNLOADED (IEEE paywalled). Read on IEEE Xplore.

### Level 1 — DSP Fundamentals (Week 1)
- **Oppenheim & Schafer** — Textbook. Not a standalone PDF.
- **Kuo & Morgan 1996** — NOT DOWNLOADED (IEEE paywalled). Read on IEEE Xplore.

### Level 2 — Deep Learning for SE (Week 2)
- `00_DeepFilterNet2_Schroeter_2022_original.pdf` — HAVE
- `02_DeepFilterNet3_Schroeter_2023.pdf` — HAVE
- `03_DCCRN_Hu_2020.pdf` — HAVE
- `04_DTLN_Westhausen_2020.pdf` — HAVE
- `05_RNNoise_Valin_2017.pdf` — HAVE
- `09_pDeepFilterNet2_Serre_2024.pdf` — HAVE
- `13_DeepFilterNet_Schroeter_2022.pdf` — HAVE (arXiv copy)

### Level 3 — Hybrid Systems (Week 3)
- `01_SFANC_FxNLMS_Luo_2022.pdf` — HAVE
- `08_HAD_ANC_Park_2023.pdf` — HAVE
- `06_Latent_FxLMS_Sarkar_2025.pdf` — HAVE
- `07_DRDO_AI_ANC_Narain_2026.pdf` — HAVE
- `11_Selective_NC_Review_2025.pdf` — HAVE
- `12_GFANC_FxNLMS_Luo_2026.pdf` — HAVE
- `15_Deep ANC_Zhang_2021.pdf` — HAVE

### Level 4 — Benchmarks (Week 4)
- `14_URGENT_Challenge_2025.pdf` — HAVE
- **DNS Challenge** — NOT A PAPER. Get dataset from GitHub: https://github.com/microsoft/DNS-Challenge

### Level 5 — Embedded (Week 4)
- **NVIDIA Real-Time SE** — Blog post, not a paper. URL: https://developer.nvidia.com/blog/nvidia-real-time-noise-suppression-deep-learning/
- **DeepFilterNet GitHub** — Repository, not paper. URL: https://github.com/rikorose/deepfilternet

### Level 6 — Advanced (Ongoing)
- `10_Kalman_Impulsive_Liu_2025.pdf` — HAVE
- **MAD Dataset** — Dataset, not paper. URL: https://github.com/goodwink/mad
- **C3GD** — Dataset. Search for "C3GD gunshot dataset"
- **PESQ (ITU-T P.862)** — Standard document. Available from ITU.
- **STOI (Taal 2010)** — IEEE paywalled.
- **SI-SNR (Le Roux 2018)** — arXiv available.

---

## TOTAL SIZE

| Category | Papers | Total Size |
|----------|--------|------------|
| Model architectures (L2) | 7 | ~5.9 MB |
| Hybrid systems (L3) | 7 | ~19.3 MB |
| Benchmarks (L4) | 1 | ~200 KB |
| Advanced (L6) | 1 | ~6.8 MB |
| Reference | 1 | ~479 KB |
| **Total** | **17** | **~32.7 MB** |

---

## WHAT IS NOT DOWNLOADED (Paywalled)

These papers require IEEE Xplore institutional access or textbook purchase:

1. **Widrow 1975** — "Adaptive Noise Cancelling: Principles and Applications" (Proc. IEEE)
2. **Boll 1979** — "Spectral Subtraction" (IEEE)
3. **Haykin 2013** — "Adaptive Filter Theory" (textbook, Ch. 6)
4. **Kuo & Morgan 1996** — "Active Noise Control: A Tutorial Review" (IEEE)
5. **Taal 2010** — STOI paper (IEEE)
6. **Le Roux 2018** — SI-SNR definition (check arXiv)

**Action:** Check if your college has IEEE Xplore access. If yes, download these from the campus network.

---

## QUICK START

For the team member starting today:

1. Read `00_DeepFilterNet2_Schroeter_2022_original.pdf` (your primary model)
2. Read `01_SFANC_FxNLMS_Luo_2022.pdf` (your hybrid architecture reference)
3. Read `07_DRDO_AI_ANC_Narain_2026.pdf` (what DRDO says about military ANC)

Then follow the reading schedule in `docs/00_RESEARCH_PAPER_READING_ORDER.md`.
