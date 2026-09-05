# ADAPTIVE-DEFENCE ANC: Hybrid AI-DSP Edge Speech Enhancement
> **Smart India Hackathon (SIH) 2026 — Problem Statement ID: 26052**  
> **Organization:** Defence Research and Development Organisation (DRDO)  
> **Department:** Department of Defence Production / iDEX • **Category:** Hardware • **Theme:** Smart Vehicles  

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![SIH 2026](https://img.shields.io/badge/SIH-2026%20PS%2026052-orange.svg)](https://sih.gov.in/)
[![DRDO Defence](https://img.shields.io/badge/DRDO-Defence%20Hardware-green.svg)](https://drdo.gov.in/)
[![Real-Time Factor](https://img.shields.io/badge/RTF-0.011%20(Edge)-brightgreen.svg)]()
[![Latency Target](https://img.shields.io/badge/Algorithmic%20Latency-32ms%20(%3C30ms%20opt)-yellowgreen.svg)]()

---

## 📌 Executive Summary

Modern defence communication environments (armoured vehicle cockpits, combat aircraft, forward artillery posts, and naval engine rooms) present extreme acoustic challenges: **intense non-stationary background noise (tank engine rumble, propeller hum) interspersed with high-energy impulsive transients (gunfire, artillery blast, shockwaves) exceeding 110–130 dB SPL**.

Standard pure-DSP adaptive filters (such as classic FxLMS) suffer from stability collapse and divergence during impulsive shocks. Conversely, pure end-to-end Deep Learning models suffer from excessive parameter counts and prohibitive latency (>50–100 ms) unviable on tactical edge silicon.

**ADAPTIVE-DEFENCE ANC** solves this via a **Hybrid AI-DSP Two-Stage Edge Architecture**:
1. **Classical Adaptive DSP Stage (Pre-AI Reference Canceller - Config A):** Fast Normalized Least Mean Squares (NLMS) with dual-microphone reference cancellation, tracking and suppressing correlated stationary/engine noise before non-linear saturation.
2. **Deep Learning Enhancement Stage (Post-Filter / DeepFilterNet / CRN):** A compact deep neural network that models remaining non-linear residual noise, preserves speech spectral formants, and suppresses impulsive transients without speech distortion.

---

## 🎯 Target Specifications (DRDO PS 26052)

| Performance Parameter | Target Requirement | Current Baseline Status | Verification Method |
|:---|:---:|:---:|:---|
| **Signal-to-Noise Ratio (SNR)** | **> 15.0 dB** | Testing in Progress | `02_evaluation_harness.py` |
| **Speech Intelligibility (STOI)** | **> 0.85** | Measured on Baseline | `pystoi` benchmark suite |
| **Perceptual Speech Quality (PESQ)** | **> 2.50 (ITU-T P.862)** | In Validation | `pesq` automated suite |
| **Algorithmic Latency** | **< 30.0 ms** | **32.0 ms** (50% overlap STFT) | `03_streaming_skeleton.py` |
| **Real-Time Factor (RTF)** | **< 1.0 (Real-Time)** | **0.011** (Ultra-low compute) | Frame loop profiler |
| **Impulsive Transient Suppression** | **> 20 dB suppression** | Config A verified | `04_adaptive_filter_lab.py` |

---

## 🏗️ System Architecture

```
                                    +-----------------------------------------+
                                    |        DUAL MICROPHONE FRONT-END        |
                                    +-----------------------------------------+
                                             |                       |
                                     d(n)    |                       |    x(n)
                             Primary Mic     |                       | Reference Mic
                        (Speech + Noise d)   v                       v (Noise Reference x)
                                        +---------------------------------+
                                        |   STAGE 1: ADAPTIVE DSP NLMS    |
                                        | (Config A: Pre-AI Canceller)    |
                                        +---------------------------------+
                                                        |
                                                        v  e(n) = d(n) - y(n)
                                              Partially Enhanced Error
                                                        |
                                                        v
                                        +---------------------------------+
                                        |  STAGE 2: DEEP LEARNING MODEL   |
                                        | (DeepFilterNet / CRN Residual)  |
                                        +---------------------------------+
                                                        |
                                                        v
                                        +---------------------------------+
                                        |    TACTICAL AUDIO OUTPUT ŝ(n)    |
                                        |  (Formant-Preserved Speech)     |
                                        +---------------------------------+
```

---

## 📂 Repository Organization

```
SIH 2026/
├── README.md                            # Master repository documentation (this file)
├── .gitignore                           # Excludes >100MB archives, binaries & raw WAVs
├── Adaptive ANC/
│   ├── README.md                        # Project technical quickstart
│   ├── requirements.txt                 # Core dependencies (numpy, scipy, pystoi, pesq)
│   ├── artifacts/                       # Executable engineering scripts
│   │   ├── 01_dataset_generator.py      # Defence noise generator & metadata provenance
│   │   ├── 02_evaluation_harness.py     # Automated SNR, STOI, PESQ, SI-SNR benchmarker
│   │   ├── 03_streaming_skeleton.py     # Low-latency ring-buffer & STFT streaming engine
│   │   └── 04_adaptive_filter_lab.py    # NLMS adaptive filter lab (Config A/B/C comparison)
│   ├── docs/                            # Deep engineering documents & build specifications
│   │   ├── 00_RESEARCH_PAPER_READING_ORDER.md  # 60+ paper reading progression
│   │   ├── 01_V1_CONTRACT.md            # Hardware & software contract interface
│   │   ├── 02_BUILD_GUIDE.md            # Reproduction & build guide
│   │   ├── 03_REFERENCE_LIBRARY.md      # Consolidated literature library
│   │   ├── 04_EVIDENCE_LOG.md           # Experimental evidence & measurement proofs
│   │   ├── 05_DEFENCE_PACKAGE.md        # DRDO defence pitch & jury justification
│   │   ├── 06_REBUILD_GUIDE.md          # Clean rebuild procedure
│   │   └── 07_PROTOTYPE_DEMO_PLAN.md    # Real-time hardware demonstration plan
│   ├── papers/                          # 60+ curated research papers & indexing dossiers
│   ├── configs/                         # Model & DSP YAML hyperparameters
│   ├── data/
│   │   ├── clean/                       # Clean speech directory
│   │   ├── noise/                       # Tactical defence noise directory
│   │   ├── mixed/                       # Synthesized mixtures
│   │   └── metadata/                    # Dataset metadata (CSV & JSON)
│   └── results/                         # Benchmarks & evaluation logs
```

---

## 🔗 Collaboration & Connection with [ichigo137/anc](https://github.com/ichigo137/anc)

This repository is designed to integrate symbiotically with the working AI repository [`ichigo137/anc`](https://github.com/ichigo137/anc):

```
+------------------------------------+           +------------------------------------+
|     OUR REPO (SIH 2026 / ANC)      |           |     ICHIGO137/ANC REPO             |
| (khanjunaidtmsl2024-bot/adaptive-anc)|          | (github.com/ichigo137/anc)         |
+------------------------------------+           +------------------------------------+
| • Real-time STFT/iSTFT Streaming   |           | • PyTorch Model Definitions        |
| • Dual-Mic NLMS Pre-AI Canceller   |  =====>   | • Deep Learning Training Pipeline  |
| • Standardized ITU-T Evaluation    |  <=====   | • Checkpoint Generation (.pt/.pth) |
| • DRDO Defence Specs & Dossiers    |           | • GPU Performance Benchmarking     |
+------------------------------------+           +------------------------------------+
```

### Integration Workflow:
1. **Model Checkpoint Sharing:** Weights trained by `ichigo137/anc` (`train.py` -> `models/`) drop straight into our `artifacts/03_streaming_skeleton.py` callback for real-time edge streaming.
2. **Dataset Format Unification:** `01_dataset_generator.py` produces calibrated SNR mixtures with JSON/CSV provenance for `ichigo137/anc/src/generate_dataset.py`.
3. **Rigorous Defense Validation:** Our `02_evaluation_harness.py` computes the authoritative PESQ, STOI, and SI-SNR scores required for DRDO jury presentations.
4. **Git Sync:** We will link `ichigo137/anc` as an `upstream` remote (`git remote add upstream https://github.com/ichigo137/anc.git`), enabling clean cross-repo pull requests and branch merges.

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/khanjunaidtmsl2024-bot/adaptive-anc.git
cd adaptive-anc

# Install dependencies
pip install -r "Adaptive ANC/requirements.txt"
```

### 2. Generate Synthetic Defence Dataset
```bash
python "Adaptive ANC/artifacts/01_dataset_generator.py" --preset demo --max-mixes 50
```

### 3. Run Adaptive Filter Laboratory (Config A Pre-AI Canceller)
```bash
python "Adaptive ANC/artifacts/04_adaptive_filter_lab.py" --mode compare --duration 3.0
```

### 4. Benchmark Streaming Latency & Real-Time Factor
```bash
python "Adaptive ANC/artifacts/03_streaming_skeleton.py" --mode benchmark
```

### 5. Multi-Metric Evaluation
```bash
python "Adaptive ANC/artifacts/02_evaluation_harness.py" \
    --clean "Adaptive ANC/data/clean/demo.wav" \
    --enhanced "Adaptive ANC/results/enhanced.wav"
```

---

## 👥 Project Team

- **Junaid Khan** ([@khanjunaidtmsl2024-bot](https://github.com/khanjunaidtmsl2024-bot)) — *Project Lead, System Architecture & Hybrid AI-DSP Design*
- **Bitan** — *Deep Learning & Neural Model Optimization*
- **Pabi** — *Embedded DSP & Hardware Interfacing*
- **Payel** — *Dataset Provenance, Acoustic Data & Testing*
- **Shristy** — *Evaluation Metrics, Benchmarking & ITU-T Validation*
- **Shruti** — *Hardware Integration & Tactical Communications*

---

## 📜 License & Compliance

Developed for **Smart India Hackathon (SIH) 2026** under Problem Statement **26052** for the **Defence Research and Development Organisation (DRDO)**.
Internal Project — All Rights Reserved.
