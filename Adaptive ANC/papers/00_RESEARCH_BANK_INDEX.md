# RESEARCH BANK V1 - COMPLETE INDEX
## PS 26052 Adaptive Defence ANC

**Location:** papers/ folder
**Total unique papers:** 25 (50 MB, all verified valid PDFs)

---

## TIER 1 - CORE AI SPEECH ENHANCEMENT (8 papers)

| # | File | Paper | Year | Topic |
|---|------|-------|------|-------|
| 03 | 03_DCCRN_Hu_2020.pdf | DCCRN: Deep Complex Convolution Recurrent Network | 2020 | Complex-domain, 3.7M params |
| 00 | 00_DeepFilterNet2_Schroeter_2022_original.pdf | DeepFilterNet2: Real-Time SE on Embedded | 2022 | Primary model candidate |
| 17 | 17_FullSubNet_Gao_2020.pdf | FullSubNet: Full-Band and Sub-Band Fusion | 2020 | Sub-band processing |
| 18 | 18_FullSubNet_Plus_Li_2022.pdf | FullSubNet+: Channel Attention + Complex | 2022 | Sub-band + complex |
| 04 | 04_DTLN_Westhausen_2020.pdf | DTLN: Dual-Signal Transformation LSTM | 2020 | Less than 1M params |
| 20 | 20_Deep_Complex_UNet_Choi_2019.pdf | Phase-aware SE with Deep Complex U-Net | 2019 | Complex spectral mapping |
| 27 | 27_CRN_Speech_Enhancement_Park_2020.pdf | CRN-based Speech Enhancement | 2020 | Convolutional-recurrent |
| 05 | 05_RNNoise_Valin_2017.pdf | RNNoise: Hybrid DSP/DL Real-Time SE | 2017 | Ultra-lightweight |

## TIER 2 - ARCHITECTURE ALTERNATIVES (5 papers)

| # | File | Paper | Year | Topic |
|---|------|-------|------|-------|
| 19 | 19_Conv_TasNet_Luo_2018.pdf | Conv-TasNet: Time-Domain Speech Separation | 2018 | Time-domain alternative |
| 21 | 21_MetricGAN_Fu_2019.pdf | MetricGAN: GAN-based Metric Optimization | 2019 | Metric-oriented training |
| 22 | 22_MetricGAN_Plus_Fu_2021.pdf | MetricGAN+: Improved MetricGAN | 2021 | PESQ-oriented training |
| 02 | 02_DeepFilterNet3_Schroeter_2023.pdf | DeepFilterNet3: Full-Band Audio SE | 2023 | Full-band extension |
| 09 | 09_pDeepFilterNet2_Serre_2024.pdf | Personalized DeepFilterNet2 | 2024 | Noise-specific personalization |

## TIER 3 - ADAPTIVE ANC / FxLMS (5 papers)

| # | File | Paper | Year | Topic |
|---|------|-------|------|-------|
| 23 | 23_FxLMS_Review_MDPI_2018.pdf | Frequency-Domain FxLMS: A Review | 2018 | Major FxLMS reference |
| 25 | 25_Modified_FxLMS_MDPI_2021.pdf | Modified Filtered-X Hierarchical LMS | 2021 | Efficient ANC |
| 01 | 01_SFANC_FxNLMS_Luo_2022.pdf | Hybrid SFANC-FxNLMS for ANC | 2022 | AI + adaptive filter |
| 12 | 12_GFANC_FxNLMS_Luo_2026.pdf | Stabilized Hybrid GFANC-FxNLMS | 2026 | Latest hybrid |
| 06 | 06_Latent_FxLMS_Sarkar_2025.pdf | Latent FxLMS: Neural Adaptive Filters | 2025 | Cutting-edge |

## TIER 4 - HYBRID AI + ANC (3 papers)

| # | File | Paper | Year | Topic |
|---|------|-------|------|-------|
| 08 | 08_HAD_ANC_Park_2023.pdf | HAD-ANC: Adaptive Filter + DNNs | 2023 | DNN + adaptive for ANC |
| 15 | 15_Deep ANC_Zhang_2021.pdf | Deep ANC: Deep Learning for ANC | 2021 | Supervised ANC |
| 11 | 11_Selective_NC_Review_2025.pdf | AI-Driven Selective Noise Cancellation | 2025 | Systematic review |

## TIER 5 - DEFENCE / IMPULSIVE / MILITARY (4 papers)

| # | File | Paper | Year | Topic |
|---|------|-------|------|-------|
| 07 | 07_DRDO_AI_ANC_Narain_2026.pdf | AI Driven ANC for Military Operations | 2026 | DRDO own research |
| 10 | 10_Kalman_Impulsive_Liu_2025.pdf | Kalman Filter for Impulsive Noise | 2025 | Impulse noise |
| 16 | 16_DRDO_Research_Synthesis.pdf | DRDO ANC Research Synthesis | 2026 | Your team document |
| 14 | 14_URGENT_Challenge_2025.pdf | URGENT Speech Enhancement Challenge | 2025 | Universal SE benchmark |

---

## COVERAGE MATRIX

| Topic | Papers | Status |
|-------|--------|--------|
| Classical NC | 23, 25 | 2 papers |
| Deep Learning SE | 00, 02, 03, 04, 05, 09, 17, 18, 27 | 9 papers |
| Complex-Domain SE | 03, 18, 20 | 3 papers |
| Time-Domain SE | 19 | 1 paper |
| Low-Latency/Causal | 00, 04, 05, 17 | 4 papers |
| Sub-band SE | 17, 18 | 2 papers |
| Lightweight | 04, 05, 09 | 3 papers |
| AI+Adaptive Hybrid | 01, 06, 08, 12, 15 | 5 papers |
| Impulsive Noise | 10 | 1 paper |
| Non-Stationary | 23 | 1 paper |
| Multichannel | -- | NEED |
| AEC/Dereverb | -- | NEED |
| VAD | -- | NEED |
| Robustness | 11, 14 | 2 papers |
| Loss Functions | 21, 22 | 2 papers |
| Edge Deployment | 00, 04, 05 | 3 papers |
| Model Compression | -- | NEED |
| Controller | -- | NEED |
| Safety/Speech | -- | NEED |
| Defence | 07, 10, 16 | 3 papers |
| E2E Architecture | 01, 08, 12 | 3 papers |

---

## GAPS NEEDING PAPERS

### Critical
1. Multichannel/Beamforming - two-mic design
2. VAD - impulse detection and mode switching
3. Controller/Decision Layer - runtime controller refs
4. Safety/Speech Preservation - over-suppression prevention

### Important
5. Model Compression - quantization/pruning
6. AEC/Dereverberation - headset environment
7. Non-Stationary Noise - specific papers

---

## NEEDING INSTITUTIONAL ACCESS

| Paper | Source |
|-------|--------|
| Widrow 1975 | IEEE Xplore |
| Boll 1979 | IEEE Xplore |
| Kuo and Morgan 1996 | IEEE Xplore |
| Haykin 2013 Ch.6 | Textbook |
| Taal 2010 STOI | IEEE Xplore |
| Le Roux 2018 SI-SNR | Check arXiv |

---

**Coverage: ~70% of 25 topics. Remaining 30% needs dedicated searches.**