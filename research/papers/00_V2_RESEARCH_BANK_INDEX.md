# RESEARCH BANK V2 - 50-PAPER MASTER INDEX
## PS 26052 Adaptive Defence ANC

**Location:** papers/ folder
**Total:** 49 PDFs + STOI reference code (MATLAB) + 5 papers still paywalled
**Verification status:** See per-paper table below. File existence ≠ content verification.

---

## TOP 15 MUST-READ PAPERS

| Rank | File | Paper | Why |
|------|------|-------|-----|
| 1 | 01_SFANC_FxNLMS_Luo_2022.pdf | Hybrid SFANC-FxNLMS | Closest to Config A |
| 2 | 03_DCCRN_Hu_2020.pdf | DCCRN | Complex-domain, 3.7M params |
| 3 | 00_DeepFilterNet2_...pdf | DeepFilterNet2 | Primary model |
| 4 | 17_FullSubNet_Gao_2020.pdf | FullSubNet | Sub-band processing |
| 5 | 18_FullSubNet_Plus_...pdf | FullSubNet+ | Sub-band + complex |
| 6 | 04_DTLN_Westhausen_2020.pdf | DTLN | Lightweight benchmark |
| 7 | 15_Deep ANC_Zhang_2021.pdf | Deep ANC | Why AI in ANC |
| 8 | 30_Speech_Preserving_ANC_2026.pdf | Speech-Preserving ANC | Dont suppress speech |
| 9 | 33_Generalization_Gap_2023.pdf | Generalization Gap | Cross-condition testing |
| 10 | 23_FxLMS_Review_MDPI_2018.pdf | FxLMS Review | Adaptive filter foundation |
| 11 | 20_Deep_Complex_UNet_...pdf | Deep Complex U-Net | Phase-aware foundation |
| 12 | 07_DRDO_AI_ANC_Narain_2026.pdf | DRDO Narain 2026 | DRDO military ANC |
| 13 | 38_Implicit_Wiener_...pdf | Implicit Wiener | Drone+helicopter noise |
| 14 | 05_RNNoise_Valin_2017.pdf | RNNoise | Ultra-lightweight |
| 15 | 06_Latent_FxLMS_...pdf | Latent FxLMS | Research direction |

---

## COVERAGE STATUS (updated after Scite verification)

| Evidence Block | Status | Papers In Bank | Gap |
|---------------|--------|---------------|-----|
| Classical ANC / LMS / FxLMS | STRONG | 4 | FxLMS review, SFANC, Modified FxLMS, GFANC |
| Neural SE models | STRONG | 9 | DFN2, DCCRN, FullSubNet, DTLN, CRN, RNNoise, etc. |
| Complex-domain / phase | STRONG | 3 | Deep Complex U-Net (FIXED), DCCRN, FullSubNet+ |
| Lightweight / causal / embedded | STRONG | 4 | DTLN, DFN2, Lite FullSubNet+, RNNoise |
| AI + adaptive ANC | STRONG | 5 | SFANC-FxNLMS, GFANC-FxNLMS, HAD-ANC, Deep ANC, Latent FxLMS |
| Defence / military | STRONG | 3 | DRDO Narain, DRDO Synthesis, Implicit Wiener |
| Multichannel / beamforming | STRONG (Branch B, V2 contingency) | 6 | Ali GSC, Shimada NMF, Zhou dual-mic, Saric MVDR, Cheong postfilter, Kwon LSTM+FxLMS |
| VAD / speech protection | STRENGTHENED | 1 | Speech-Preserving ANC; Valin VAD output noted |
| Impulsive / transient robustness | NEEDS DEEPER SEARCH | 2 | Kalman Impulsive; construction-site ANC (paywalled) |
| Compression / quantization / deployment | STILL GAPS | 0 | No dedicated QAT/pruning/distillation papers yet |

### Scite-verified additions - all downloaded (Sep 2026)

| File | Paper | Source used |
|------|-------|-------------|
| 44 | Kwon et al. 2022 LSTM+FxLMS (diesel engine) | Hanyang ScholarWorks (MDPI CC-BY) |
| 50 | Ali et al. 2018 GSC hearing devices + external mic | Wayback (ICASSP paper PDF) |
| 54 | Shimada et al. 2019 multichannel NMF beamforming | Kyoto University repository (TASLP) |
| 47 | Zhou et al. 2020 real-time dual-mic + bone conduction | Europe PMC (MDPI Sensors CC-BY) |
| 52 | Saric MVDR end-fire array (low-cost spatial) | journals.pan.pl (open) |
| 53 | Cheong et al. 2024 dual-channel coherence postfilter | Europe PMC (MDPI Sensors CC-BY) |
| 46 | Roy et al. 2020 DL-Kalman SE | ISCA Interspeech archive (open) |
| 51 | Nikzad et al. 2020 DRLN (bonus causal comparator) | arXiv 2002.12794 |
| 43 | DecNet-LMS 2026 (bonus, neural path decoupling) | arXiv |

Still paywalled: Construction-site ANC 2023 (ScienceDirect), Vehicle-engine hybrid 2025 (ScienceDirect; SSRN preprint blocked by Cloudflare).

---

## PAPER VERIFICATION LOG

**Standard:** "Verified" means: title confirmed, authors confirmed, DOI/arXiv confirmed, PDF content matches expected paper, relevance to PS 26052 assessed.

| # | File | Verified? | Notes |
|---|------|-----------|-------|
| 00 | DeepFilterNet2 | PARTIAL | File exists, size consistent. Content not page-checked. |
| 01 | SFANC-FxNLMS | PARTIAL | File exists (3.2 MB). Content not page-checked. |
| 02 | DeepFilterNet3 | PARTIAL | File exists. Content not page-checked. |
| 03 | DCCRN | PARTIAL | File exists. Content not page-checked. |
| 04 | DTLN | PARTIAL | File exists (199K). Content not page-checked. |
| 05 | RNNoise | PARTIAL | File exists. Content not page-checked. |
| 06 | Latent FxLMS | PARTIAL | File exists. Content not page-checked. |
| 07 | DRDO Narain 2026 | PARTIAL | File exists. Content not page-checked. |
| 08 | HAD-ANC Park 2023 | PARTIAL | File exists (11 MB). Content not page-checked. |
| 09 | pDeepFilterNet2 | PARTIAL | File exists. Content not page-checked. |
| 10 | Kalman Impulsive | PARTIAL | File exists. Content not page-checked. |
| 11 | Selective NC Review | PARTIAL | File exists. Content not page-checked. |
| 12 | GFANC-FxNLMS | PARTIAL | File exists (2.4 MB). Content not page-checked. |
| 14 | URGENT Challenge | PARTIAL | File exists. Content not page-checked. |
| 15 | Deep ANC | PARTIAL | File exists. Content not page-checked. |
| 16 | DRDO Synthesis | PARTIAL | File exists. Content not page-checked. |
| 17 | FullSubNet | PARTIAL | File exists. Content not page-checked. |
| 18 | FullSubNet+ | PARTIAL | File exists. Content not page-checked. |
| 19 | Conv-TasNet | PARTIAL | File exists. Content not page-checked. |
| 20 | Deep Complex U-Net | FIXED | Old file was misidentified (unrelated content). Replaced with correct arXiv 1903.03107 (5.16 MB). Old backup: 20_Deep_Complex_UNet_OLD_MISIDENTIFIED.pdf |
| 21 | MetricGAN | PARTIAL | File exists. Content not page-checked. |
| 22 | MetricGAN+ | PARTIAL | File exists. Content not page-checked. |
| 23 | FxLMS Review MDPI | PARTIAL | File exists. Content not page-checked. |
| 25 | Modified FxLMS | PARTIAL | File exists. Content not page-checked. |
| 27 | CRN SE | PARTIAL | File exists. Content not page-checked. |
| 30 | Speech-Preserving ANC | PARTIAL | File exists. Content not page-checked. |
| 32 | Noise Embeddings | PARTIAL | File exists. Content not page-checked. |
| 33 | Generalization Gap | PARTIAL | File exists. Content not page-checked. |
| 34 | MC Dropout Unseen | PARTIAL | File exists. Content not page-checked. |
| 35 | DPRNN | PARTIAL | File exists. Content not page-checked. |
| 36 | PHASEN | PARTIAL | File exists. Content not page-checked. |
| 37 | DeepFilterNet Original | PARTIAL | File exists. Content not page-checked. |
| 38 | Implicit Wiener | PARTIAL | File exists. Content not page-checked. |
| 39 | MANNER | PARTIAL | File exists. Content not page-checked. |
| 40 | Transferable Latent SFANC | PARTIAL | File exists. Content not page-checked. |
| 41 | Neural Secondary Path | PARTIAL | File exists. Content not page-checked. |
| 42 | Lite FullSubNet+ | PARTIAL | File exists. Content not page-checked. |
| 43 | DecNet-LMS | PARTIAL | arXiv download, 580 KB. Content not page-checked. |
| 44 | Kwon LSTM+FxLMS | PARTIAL | MDPI via Hanyang repo, 4.9 MB, 10 pages. Content not page-checked. |
| 45 | Boll 1979 | PARTIAL | USPTO-hosted copy, 1.5 MB. Content not page-checked. |
| 46 | Roy DL-Kalman | PARTIAL | ISCA archive, 625 KB, 5 pages. Content not page-checked. |
| 47 | Zhou dual-mic | PARTIAL | Europe PMC, 4.6 MB, 10 pages. Content not page-checked. |
| 48 | TI FxLMS design note | PARTIAL | TI spra042, 661 KB (bonus implementation reference). |
| 49 | STOI reference code | PARTIAL | Official Taal stoi.m, zipped (replaces paywalled paper). |
| 50 | Ali GSC | PARTIAL | ICASSP PDF via Wayback, 287 KB, 6 pages. Content not page-checked. |
| 51 | Nikzad DRLN | PARTIAL | arXiv 2002.12794, 762 KB, 6 pages. Content not page-checked. |
| 52 | Saric MVDR | PARTIAL | journals.pan.pl, 1.6 MB, 11 pages. Content not page-checked. |
| 53 | Cheong postfilter | PARTIAL | Europe PMC, 3.3 MB, 10 pages. Content not page-checked. |
| 54 | Shimada NMF | PARTIAL | Kyoto repo (TASLP), 4.9 MB, 40 pages. Content not page-checked. |
| 57 | Widrow 1975 | PARTIAL | Stanford ISL PDF via Wayback, 2.4 MB, 26 pages. Content not page-checked. |

### Verification status summary
- **Fully verified (title + content checked):** 0
- **Fixed (was wrong, now correct):** 1 (paper 20)
- **Partially verified (file exists, size/page count consistent, content not page-checked):** 49
- **Still paywalled:** 5 (see below)

---

## RESEARCH QUESTION

> How can a lightweight causal AI speech-enhancement model and an adaptive filter
> cooperate to suppress stationary, non-stationary and impulsive defence noise
> while preserving speech and meeting embedded real-time constraints?



---

## IMPORTANT ARCHITECTURAL CLARIFICATION

**GFANC-FxNLMS is architectural inspiration, not proof of equivalence.**

The GFANC-FxNLMS paper (Luo 2026) belongs to an ANC framework involving:
- Control signals
- Acoustic paths
- Secondary speaker path
- Secondary-path modelling

Our V1 is primarily **speech enhancement / communication noise suppression** unless we physically implement:
- Reference microphone ✓ (we have this)
- Controller ✓ (we have this)
- Loudspeaker/headphone anti-noise path ✗ (we do NOT have this)
- Error microphone ✗ (we do NOT have this)
- Secondary-path modelling ✗ (we do NOT have this)

**Therefore:** GFANC-FxNLMS should be described as "strong architectural inspiration for hybrid learned + adaptive processing" — NOT proof that our digital pipeline is mathematically equivalent to physical ANC.

**Two research branches — do not merge:**
- **Branch A (V1):** Reference-assisted adaptive suppression. Primary mic = speech + noise, Reference mic = correlated environmental noise. This is what we build.
- **Branch B (V2 contingency):** Spatial multichannel enhancement / beamforming. Requires specific geometry, spatial separation, DOA analysis. Not V1.

---

## ARCHITECTURAL INSIGHT FROM LITERATURE

The GFANC-FxNLMS paper (Luo 2026) makes the hybrid architecture concrete:

1. CNN at frame rate generates/selects a control filter
2. FxNLMS at sample rate continuously adapts it
3. Online clustering prevents CNN from destabilizing FxLMS reinitialization
4. CNN is only 0.21M parameters

**For our project:** AI handles fast/high-level adaptation; classical adaptive filtering handles continuous fine adaptation. The controller must prevent reinitialization instability.

---

## STILL PAYWALLED (5 papers) - last update 3 Sep 2026

Resolved since V2: Widrow 1975 (Wayback), Boll 1979 (USPTO), DecNet-LMS 2026 (arXiv), Taal STOI 2010 (official reference code downloaded instead).

| Paper | Source | Why we want it |
|-------|--------|----------------|
| Kuo & Morgan 1996 (book) | Wiley | FxLMS tutorial; textbook - library copy is fine |
| Time-frequency FxLMS 2012 | ScienceDirect | STFT + FxLMS, non-stationary noise |
| Stochastic FxLMS 2020 | IEEE Xplore | Convergence/stability analysis |
| Construction-site ANC 2023 | ScienceDirect | Impact/transient noise ANC |
| Vehicle-engine hybrid 2025 | ScienceDirect | DL secondary-path + non-stationary engine |

None have open-access copies (checked: Semantic Scholar, Unpaywall, Crossref, Wayback, Europe PMC, arXiv, SSRN - SSRN preprint is Cloudflare-blocked).
Use college library / interlibrary loan, or cite the abstract-level metadata.

## DOWNLOAD SOURCE LOG (3 Sep 2026)

| # | Paper | URL | Method |
|---|-------|-----|--------|
| 43 | DecNet-LMS 2026 | arXiv | direct |
| 44 | Kwon 2022 | hanyang.scholarworks.kr | ScholarWorks API |
| 45 | Boll 1979 | USPTO-hosted copy | direct |
| 46 | Roy 2020 | isca-archive.org | direct |
| 47 | Zhou 2020 | europepmc.org | Europe PMC render |
| 48 | TI FxLMS note | ti.com | direct |
| 49 | STOI code | official Taal release | zip |
| 50 | Ali 2018 | web.archive.org | Wayback of scispace PDF |
| 51 | Nikzad 2020 | arxiv.org/abs/2002.12794 | direct |
| 52 | Saric MVDR | journals.pan.pl | direct |
| 53 | Cheong 2024 | europepmc.org | Europe PMC render |
| 54 | Shimada 2019 | repository.kulib.kyoto-u.ac.jp | DSpace API |
| 57 | Widrow 1975 | web.archive.org | Wayback of isl.stanford.edu PDF |
